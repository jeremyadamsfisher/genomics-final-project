import random
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn.metrics as metrics
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence
from tqdm import tqdm, trange

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ONTOLOGY_SEQ_DATASET_FP = "data/intermediary/drosophila_protein_ontology_and_seqs.csv"

# load sequence and ontology data into memory
df_original = pd.read_csv(ONTOLOGY_SEQ_DATASET_FP)
relevant_subset = df_original[df_original.qualifier.isin(["enables", "involved_in"])].dropna()
interesting_go_names = [
    name for (name, freq)
    in relevant_subset.go_name.value_counts().to_dict().items()
    if 100 < freq
]  
relevant_subset = relevant_subset[relevant_subset.go_name.isin(interesting_go_names)]
df = pd.DataFrame(index=relevant_subset.seq.unique(), columns=interesting_go_names).fillna(0)
for _, row in relevant_subset.iterrows():
    df.loc[row.seq, row.go_name] = 1
df = df.reset_index().rename(columns={"index": "seqs"})
vocab = set()
for seq in df.seqs:
    vocab.update(seq)
vocab.add("<pad>")
to_ix = {char: i for i, char in enumerate(vocab)}

# determine/define dataset and model parameters
N_EXAMPLES, _ = df.shape
N_ONTOLOGICAL_CATEGORIES = len(interesting_go_names)
VOCAB_SIZE = len(vocab)
EMBEDDING_DIM = 8
HIDDEN_DIM = 16
BATCH_SIZE = 1

print("-"*25)
print(f"Dataset consists of:")
print(f"\t{N_EXAMPLES} examples")
print(f"\t{N_ONTOLOGICAL_CATEGORIES} ontological categories")
print(f"\t{N_EXAMPLES*N_ONTOLOGICAL_CATEGORIES} total prediction tasks")
print("-"*25)

# convert data into tensors
class SeqOntologyDataset(torch.utils.data.Dataset):
    def __init__(self):
        self.X = list(df.seqs)
        self.y = df.loc[:, interesting_go_names]
    def __len__(self):
        return len(self.X)
    def __getitem__(self, i):
        seq = self.X[i]
        seq_tensor = torch.tensor([to_ix[residue] for residue in seq])
        label = self.y.iloc[i,:].values.T
        return seq_tensor, torch.tensor(label[np.newaxis,:], dtype=torch.double)
ds_train, ds_test = torch.utils.data.random_split(
    SeqOntologyDataset(),
    lengths=[N_EXAMPLES-(N_EXAMPLES//4), N_EXAMPLES//4]
)
dl_args = {"batch_size": BATCH_SIZE, "shuffle": True}
dl = {"train": torch.utils.data.DataLoader(ds_train, **dl_args),
      "test": torch.utils.data.DataLoader(ds_test, **dl_args)}

# define model(s)
class OntologyLSTM(nn.Module):
    def __init__(self):
        super(OntologyLSTM, self).__init__()
        self.seq_embedding = nn.Embedding(VOCAB_SIZE, EMBEDDING_DIM).to(device)
        self.lstm = nn.LSTM(EMBEDDING_DIM, HIDDEN_DIM).to(device)
        self.fc = nn.Linear(HIDDEN_DIM, N_ONTOLOGICAL_CATEGORIES).to(device)
        self.sigmoid = nn.Sigmoid().to(device)
        
    def reset(self, seq_len):
        """clear gradients from earlier examples"""
        self.zero_grad()
        self.h0 = torch.zeros(1, seq_len, HIDDEN_DIM).to(device)
        self.c0 = torch.zeros(1, seq_len, HIDDEN_DIM).to(device)
        
    def forward(self, seq):
        seq_embedded = self.seq_embedding(seq).view(len(seq), -1, EMBEDDING_DIM)
        _, (self.h0, self.c0) = self.lstm(seq_embedded, (self.h0, self.c0))
        logits = self.fc(self.c0[:,-1,np.newaxis,:])
        likelihoods = self.sigmoid(logits)
        return logits, likelihoods

clf = OntologyLSTM()
criterion = torch.nn.BCEWithLogitsLoss()
optimizer = optim.Adam(clf.parameters())

losses = {"train": [], "test": []}
f1s = []
accuracies = []
N_EPOCHS = 100
for epoch in trange(N_EPOCHS, unit="epoch"):
    for phase in ["train", "test"]:
        clf.train() if phase == "train" else clf.eval()
        running_loss = 0
        ontology_valid_truth = []
        ontology_valid_pred = []
        for seq, ontology in dl[phase]:
            seq, ontology = seq.to(device), ontology.to(device)
            _, seq_len = seq.shape
            clf.reset(seq_len)
            with torch.set_grad_enabled(phase == "train"):
                ontology_logits, ontology_likelihoods = clf(seq)
                loss = criterion(ontology_logits, ontology)
                if phase == "train":
                    loss.backward()
                    optimizer.step()
                if phase == "test":
                    ontology_valid_truth.extend(ontology.squeeze())
                    ontology_valid_pred.extend((ontology_likelihoods.squeeze()) > 0.5)
            running_loss += loss.item() * BATCH_SIZE
        losses[phase].append(running_loss/len(dl[phase]))
        if phase == "test":
            accuracies.append(metrics.accuracy_score(ontology_valid_truth, ontology_valid_pred))
            f1s.append(metrics.fbeta_score(ontology_valid_truth, ontology_valid_pred, beta=1))

with plt.style.context("ggplot"):
    fig, (ax0, ax1, ax2) = plt.subplots(1, 3, figsize=(15,5))
    ax0.plot(range(N_EPOCHS), losses["train"], label="training")
    ax0.plot(range(N_EPOCHS), losses["test"], label="testing", c="black")
    ax0.set(xlabel="Epoch", ylabel="Loss (Binary Cross Entropy)")
    ax1.plot(range(N_EPOCHS), f1s, c="black")
    ax1.set(xlabel="Epoch", ylabel="F1")
    ax2.plot(range(N_EPOCHS), accuracies, c="black")
    ax2.set(xlabel="Epoch", ylabel="Accuracy")
    fig.legend()
    fig.suptitle("LSTM")

with open("metrics.json", "wt") as f:
    json.dump({
        "accuracy": accuracies[-1],
        "f1": f1s[-1],
    }, f)