import random
import json
from pathlib import Path
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

# inputs
ONTOLOGY_SEQ_DATASET_FP = Path("data/intermediary/drosophila_full_protein_ontology_and_seqs.csv")

# outputs
MODEL_ARTIFACTS_DIR = Path("data/model_artifacts")
FILTERED_DATASET_FP = MODEL_ARTIFACTS_DIR/"drosophila_subset.csv.gz"
RUNNING_METRICS_FP = MODEL_ARTIFACTS_DIR/"running_metrics.json"
LSTM_WEIGHTS_FP = MODEL_ARTIFACTS_DIR/"lstm.pth"
RNN_WEIGHTS_FP = MODEL_ARTIFACTS_DIR/"rnn.pth"
LSTM_ATTN_WEIGHTS_FP = MODEL_ARTIFACTS_DIR/"lstm_attn.pth"

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
df.to_csv(FILTERED_DATASET_FP)

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
HIDDEN_DIM = 64
BATCH_SIZE = 8
N_EPOCHS = 200

print("="*40)
print(f"Dataset consists of:")
print(f"- {N_EXAMPLES} examples")
print(f"- {N_ONTOLOGICAL_CATEGORIES} ontological categories")
print(f"- {N_EXAMPLES*N_ONTOLOGICAL_CATEGORIES} total prediction tasks")
print(f"- {df.sum(axis=1).sum()/(N_EXAMPLES*N_ONTOLOGICAL_CATEGORIES):.2f} positives")
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
class OntologyClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.seq_embedding = nn.Embedding(VOCAB_SIZE, EMBEDDING_DIM).to(device)
        self.fc = nn.Linear(HIDDEN_DIM, N_ONTOLOGICAL_CATEGORIES).to(device)
        
    def forward(self, seq):
        seq_embedded = self.seq_embedding(seq).view(len(seq), -1, EMBEDDING_DIM)
        hidden = self.forward_(seq, seq_embedded)
        logits = self.fc(hidden)
        likelihood = torch.softmax(logits, -1).double()
        return logits, likelihood


class OntologyLSTM(OntologyClassifier):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(EMBEDDING_DIM, HIDDEN_DIM).to(device)

    def init_hidden(self, seq_len):
        return (torch.zeros(1, seq_len, HIDDEN_DIM).to(device), 
                torch.zeros(1, seq_len, HIDDEN_DIM).to(device))

    def forward_(self, seq, seq_embedded):
        _, seq_len = seq.shape
        X, _ = self.lstm(seq_embedded, self.init_hidden(seq_len))
        return X[:,-1,:]

class OntologyAttnLSTM(OntologyLSTM):
    """https://github.com/prakashpandey9/Text-Classification-Pytorch/blob/master/models/selfAttention.py"""
    def __init__(self):
        super().__init__()
        self.w1 = nn.Linear(HIDDEN_DIM, 350).to(device)
        self.w2 = nn.Linear(350, 30).to(device)
        self.fc = nn.Linear(30*HIDDEN_DIM, N_ONTOLOGICAL_CATEGORIES).to(device)
    
    def forward_(self, seq, seq_embedded):
        _, seq_len = seq.shape
        X, _ = self.lstm(seq_embedded, self.init_hidden(seq_len))
        attention = self.w1(X.to(device))
        attention = torch.tanh(attention)
        attention = self.w2(attention)
        attention = attention.permute(0, 2, 1)
        attention = torch.softmax(attention, dim=2)
        hidden = torch.bmm(attention, X)
        return hidden.view(-1, 30*HIDDEN_DIM)


class OntologyRNN(OntologyClassifier):
    def __init__(self):
        super().__init__()
        self.rnn = nn.RNN(EMBEDDING_DIM, HIDDEN_DIM).to(device)
    
    def forward_(self, _, seq_embedded):
        X, _ = self.rnn(seq_embedded)
        return X[:,-1,:]


weight = (df[interesting_go_names].sum()/N_EXAMPLES).pow(-1)  # weight loss by inverse frequency
weight = (weight / weight.mean()).apply(lambda x: max((1,x)))  # make this less extreme
criterion = nn.CrossEntropyLoss(weight=torch.tensor(weight).float().to(device))

metrics = {}
for model, model_name, model_out_fp in [
    (OntologyAttnLSTM(), "AttentionLSTM", LSTM_ATTN_WEIGHTS_FP),
    (OntologyRNN(),      "RNN",           RNN_WEIGHTS_FP),
    (OntologyLSTM(),     "LSTM",          LSTM_WEIGHTS_FP),
    ]:
    optimizer = optim.Adam(model.parameters())
    losses = {"train": [], "test": []}
    accuracies = {"train": [], "test": []}
    for epoch in trange(N_EPOCHS, unit="epoch"):
        for phase in ["train", "test"]:
            model.train() if phase == "train" else model.eval()
            running_loss = 0
            running_correct = 0
            for seq, ontology in dl[phase]:
                seq, ontology = seq.to(device), ontology.to(device)
                _, seq_len = seq.shape
                model.zero_grad()
                with torch.set_grad_enabled(phase == "train"):
                    logits, likelihood = model(seq)
                    loss = criterion(logits, ontology.argmax(axis=1))
                    if phase == "train":
                        loss.backward()
                        optimizer.step()
                running_loss += loss.item() / BATCH_SIZE
                running_correct += (likelihood.argmax(axis=1) == ontology.argmax(axis=1)).sum().item() / BATCH_SIZE
            losses[phase].append(running_loss/len(dl[phase]))
            accuracies[phase].append(running_correct/len(dl[phase]))
        if epoch % 50 == 0:
            print(f"{model_name} @ epoch {epoch+1}:")
            for phase in ["train", "test"]:
                print(f"=> {phase} accuracy:  {accuracies[phase][-1]:.1%}")
                print(f"=> {phase} avg. loss: {losses[phase][-1]:.2f}")
    metrics[model_name] = {"losses": losses, "accuracies": accuracies}
    torch.save(model, model_out_fp)

with RUNNING_METRICS_FP.open("wt") as f:
    json.dump(metrics, f)
