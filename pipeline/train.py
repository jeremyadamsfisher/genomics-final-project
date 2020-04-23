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

ONTOLOGY_SEQ_DATASET_FP = "data/intermediary/drosophila_full_protein_ontology_and_seqs.csv"

# load sequence and ontology data into memory
df_original = pd.read_csv(ONTOLOGY_SEQ_DATASET_FP)
relevant_subset = df_original[df_original.qualifier.isin(["enables", "involved_in"])].dropna()
interesting_go_names = [
    name for (name, freq)
    in relevant_subset.go_name.value_counts().to_dict().items()
    if 500 < freq
]
relevant_subset = relevant_subset[relevant_subset.go_name.isin(interesting_go_names)]
df = pd.DataFrame(index=relevant_subset.seq.unique(), columns=interesting_go_names).fillna(0)
for _, row in relevant_subset.iterrows():
    df.loc[row.seq, row.go_name] = 1
df = df.reset_index().rename(columns={"index": "seqs"})

# include **only** genes with 1 annotation
df = df[df.sum(axis=1) == 1]
interesting_go_names = [
    go for go in interesting_go_names if df[go].sum() > 0
]
df = df[["seqs"] + interesting_go_names]
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
BATCH_SIZE = 8
N_EPOCHS = 100

print("="*40)
print(f"Dataset consists of:")
print(f"\t - {N_EXAMPLES} examples")
print(f"\t - {N_ONTOLOGICAL_CATEGORIES} ontological categories")
print(f"\t - {N_EXAMPLES*N_ONTOLOGICAL_CATEGORIES} total prediction tasks")
print(f"Using {device}")
print("="*40)

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
def collate(batch):
    Xs, ys = zip(*batch)
    X = pad_sequence(Xs, batch_first=True, padding_value=to_ix["<pad>"])
    y = torch.cat(ys, 0)
    return X, y
dl_args = {"batch_size": BATCH_SIZE, "shuffle": True, "collate_fn": collate}
dl = {"train": torch.utils.data.DataLoader(ds_train, **dl_args),
      "test": torch.utils.data.DataLoader(ds_test, **dl_args)}

# define model(s)
class OntologyLSTM(nn.Module):
    def __init__(self):
        super(OntologyLSTM, self).__init__()
        self.seq_embedding = nn.Embedding(VOCAB_SIZE, EMBEDDING_DIM).to(device)
        self.lstm = nn.LSTM(EMBEDDING_DIM, HIDDEN_DIM).to(device)
        self.fc = nn.Linear(HIDDEN_DIM, N_ONTOLOGICAL_CATEGORIES).to(device)
        
    def reset(self, seq_len):
        """clear gradients from earlier examples"""
        self.zero_grad()
        self.c0 = torch.zeros(1, seq_len, HIDDEN_DIM).to(device)
        self.h0 = torch.zeros(1, seq_len, HIDDEN_DIM).to(device)
        
    def forward(self, seq):
        seq_embedded = self.seq_embedding(seq).view(len(seq), -1, EMBEDDING_DIM)
        X, (self.h0, self.c0) = self.lstm(seq_embedded, (self.h0, self.c0))
        logits = self.fc(X[:,-1,:])
        return logits

clf = OntologyLSTM()
criterion = torch.nn.BCEWithLogitsLoss()
optimizer = optim.Adam(clf.parameters())

losses = {"train": [], "test": []}
f1s = []
accuracies = []
rocs_aucs = []
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
                logits = clf(seq)
                loss = criterion(logits, ontology)
                if phase == "train":
                    loss.backward()
                    optimizer.step()
                if phase == "test":
                    # breakpoint()
                    ontology_valid_truth.extend(ontology.view(-1).tolist())
                    likelihoods = torch.sigmoid(logits)
                    ontology_valid_pred.extend((likelihoods.squeeze() > 0.5).view(-1).tolist())
            running_loss += loss.item() * BATCH_SIZE
        losses[phase].append(running_loss/len(dl[phase]))
        if phase == "test":
            # TODO: should I be flattening out these categories before calculating metrics?
            accuracies.append(metrics.accuracy_score(ontology_valid_truth, ontology_valid_pred))
            f1s.append(metrics.fbeta_score(ontology_valid_truth, ontology_valid_pred, beta=1))
            rocs_aucs.append(metrics.roc_auc_score(ontology_valid_truth, ontology_valid_pred))

with plt.style.context("ggplot"):
    fig, (ax0, ax1, ax2, ax3) = plt.subplots(1, 4, figsize=(20,5))
    ax0.plot(range(N_EPOCHS), losses["train"], label="training")
    ax0.plot(range(N_EPOCHS), losses["test"], label="testing", c="black")
    ax0.set(xlabel="Epoch", ylabel="Loss (Binary Cross Entropy)")
    ax1.plot(range(N_EPOCHS), f1s, c="black")
    ax1.set(xlabel="Epoch", ylabel="F1")
    ax2.plot(range(N_EPOCHS), accuracies, c="black")
    ax2.set(xlabel="Epoch", ylabel="Accuracy")
    ax3.plot(range(N_EPOCHS), rocs_aucs, c="black")
    ax3.set(xlabel="Epoch", ylabel="AUC-ROC")
    fig.legend()
    fig.suptitle("LSTM")
    fig.savefig("DELETEME_performance.png")
    plt.close()

with open("metrics.json", "wt") as f:
    json.dump({
        "accuracy": accuracies[-1],
        "f1": f1s[-1],
        "auc_roc": rocs_aucs[-1],
    }, f)
