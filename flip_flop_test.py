import datasets
import torch
from torch.utils.data import DataLoader

import torch.nn as nn
import math
import torch.nn.functional as F

import matplotlib.pyplot as plt
import seaborn as sns

import numpy as np


dataset = datasets.load_dataset("synthseq/flipflop")
flip_flop_dict = {'0': 0, "1": 1, "w": 2,"r": 3, "i": 4}

def tokenize_raw(batch):
    tokenized = [[flip_flop_dict[char] for char in s] for s in batch["text"]]
    return {"tokens": torch.tensor(tokenized, dtype=torch.int64)}

dataset.set_transform(tokenize_raw)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")

class NextTokenDataset(torch.utils.data.Dataset):
    def __init__(self, hf_dataset):
        self.data = []
        for item in hf_dataset:
            tokens = item["tokens"]
            tokens = torch.tensor(tokens, dtype=torch.long)
            x = tokens[:-1]
            y = tokens[1:]
            self.data.append((x, y))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx] 

# train_small = dataset["train"].select(range(20000))
# val_small = dataset["val"].select(range(1000))
train_small = dataset["val_dense"].select(range(3200))
val_small = dataset["val_dense"].select(range(3200, 4000))
train_dataset = NextTokenDataset(train_small)
val_dataset = NextTokenDataset(val_small)

train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=128)



class Sinusoidal_Embedding(nn.Module):
    def __init__(self, embed_dim, max_len=512):
        super().__init__()
        pe = torch.zeros(max_len, embed_dim)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, embed_dim, 2) * (-math.log(10000.0) / embed_dim)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe) 

    def forward(self, x):
        return self.pe[:x.size(1)].unsqueeze(0).expand(x.size(0), -1, -1)
    
# https://stackoverflow.com/questions/71998978/early-stopping-in-pytorch idea and implementation from stack-overflow
class EarlyStopper:
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = float('inf')

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False
    

class Transformer(nn.Module):
    def __init__(
            self,
            max_seq_len:int,
            dictionary_size:int,
            num_attn_layer:int=2,
            num_attn_heads:int=1,
            attn_dim:int=8,
            
        ):
        super().__init__()
        self.word_embedding = nn.Embedding(
            num_embeddings=dictionary_size,
            embedding_dim=attn_dim,
            )
        self.positional_embedding = Sinusoidal_Embedding(
            embed_dim=attn_dim,
            max_len=max_seq_len
        )
        self.attention_layers = nn.ModuleList()

        for _ in range(num_attn_layer):
            self.attention_layers.append(nn.ModuleDict({
                "mha": nn.MultiheadAttention(embed_dim=attn_dim, num_heads=num_attn_heads, batch_first=True),
                "ffn": nn.Sequential(
                    nn.Linear(attn_dim, attn_dim),
                    nn.ReLU(),
                    nn.Linear(attn_dim, attn_dim)
                ),
            }))

        
        self.attn_weights = [] 

        self.classification = nn.ModuleList([
            nn.Linear(in_features=attn_dim, out_features=attn_dim),
            nn.ReLU(),
            nn.Linear(in_features=attn_dim, out_features=dictionary_size)
        ])

    def forward(self, x, need_weights=False):
        """
        shape of x should be: (batch_size, seq_len)
        """
        seq_len = x.size()[1]
        word_emb = self.word_embedding(x)
        pos_emb = self.positional_embedding(x)
        emb = word_emb + pos_emb

        attn_mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool().to(emb.device)

        for layer in self.attention_layers:
            residual = emb
            attn_output, attn_w = layer["mha"](emb, emb, emb, attn_mask=attn_mask, need_weights=need_weights)
            emb = residual + attn_output

            if need_weights:
                self.attn_weights.append(attn_w.detach().cpu())

            residual = emb

        
        for layer in self.classification:
            emb = layer(emb)

        return emb
    

def train_epoch(model, loader, optimizer, loss_fn):
    model.train()
    total_loss = 0
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        logits = model(x)  # (batch, seq_len, vocab_size)

        loss = loss_fn(logits.view(-1, logits.size(-1)), y.view(-1))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

def eval_epoch(model, loader, loss_fn):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            logits = model(x)
            loss = loss_fn(logits.view(-1, logits.size(-1)), y.view(-1))
            total_loss += loss.item()
    return total_loss / len(loader)
vocab_size = 5


model = Transformer(max_seq_len=512, num_attn_layer=2, dictionary_size=vocab_size).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=5e-3)
loss_fn = nn.CrossEntropyLoss()

early_stopper = EarlyStopper(patience=20)

for epoch in range(600):
    train_loss = train_epoch(model, train_loader, optimizer, loss_fn)
    val_loss = eval_epoch(model, val_loader, loss_fn)
    print(f"Epoch {epoch+1}: train loss = {train_loss:.4f}, val loss = {val_loss:.4f}")
    if early_stopper.early_stop(val_loss):
        break


# def visualize_attention(attn_weights, token_labels=None):
#     for layer_idx, layer_attn in enumerate(attn_weights):
#         layer_attn = layer_attn[0].detach().numpy()
#         plt.figure(figsize=(6, 5))
#         sns.heatmap(layer_attn, annot=False, cmap="Blues", xticklabels=token_labels, yticklabels=token_labels)
#         plt.title(f"Layer {layer_idx}")
#         plt.xlabel("Key positions")
#         plt.ylabel("Query positions")
            
#         plt.show()


def visualize_r_attention(attn_weights, token_seq):
    seq_len = len(token_seq)
    
    r_positions = [i for i, t in enumerate(token_seq) if t == 'r']
    r_pairs = [(i, i+1) for i in r_positions if i + 1 < seq_len]

    query_indices = []
    query_labels = []
    color_indices = []

    for color_id, (r_idx, next_idx) in enumerate(r_pairs):
        query_indices.append(r_idx)
        query_labels.append(f"r@{r_idx}")
        color_indices.append(color_id)

        query_indices.append(next_idx)
        query_labels.append(f"{token_seq[next_idx]}@{next_idx}")
        color_indices.append(color_id)

    # Predefine some colors
    palette = sns.color_palette("husl", n_colors=len(r_pairs))
    num_layers = len(attn_weights)

    for layer_idx in range(num_layers):
        weights = attn_weights[layer_idx].mean(dim=0)  # (seq_len, seq_len)

        # Rows for r and number after r
        selected_rows = weights[query_indices]  # (num_queries, seq_len)
        selected_np = selected_rows.cpu().numpy()

        # Setup plot
        plt.figure(figsize=(12, len(query_indices) * 0.6 + 1))
        ax = sns.heatmap(
            selected_np,
            cmap="YlOrRd",
            xticklabels=token_seq,
            yticklabels=query_labels,
            cbar=True,
            linewidths=0.3,
            linecolor='gray',
        )

        ax.set_title(f"Attention Map (Layer {layer_idx + 1})", fontsize=14)
        ax.set_xlabel("Key Positions", fontsize=12)
        ax.set_ylabel("Query Tokens", fontsize=12)

        # Color and bold the yticklabels (queries)
        for i, label in enumerate(ax.get_yticklabels()):
            label.set_color(palette[color_indices[i]])
            label.set_weight("bold")

        # Color and bold the xticklabels (keys)
        xtick_tokens = token_seq
        for i, label in enumerate(ax.get_xticklabels()):
            for j, (r_idx, next_idx) in enumerate(r_pairs):
                if i == r_idx or i == next_idx:
                    label.set_color(palette[j])
                    label.set_weight("bold")

        # Annotate the highest attention in each row
        for row_idx, row in enumerate(selected_np):
            max_col = np.argmax(row)
            ax.text(
                max_col + 0.5,
                row_idx + 0.5,
                f"{row[max_col]:.2f}",
                ha="center",
                va="center",
                fontsize=8,
                fontweight="bold",
                color="black",
                bbox=dict(facecolor='white', edgecolor='none', boxstyle='round,pad=0.2'),
            )

        plt.tight_layout()
        plt.show()

sentence = "w0w1w1r1r1w0w1w1w0r0"
ids = torch.tensor([[flip_flop_dict[c] for c in sentence]]).to(device)  # [1, seq_len]
model.eval()
_ = model(ids, need_weights=True)
attn_weights = model.attn_weights

# Visualize
visualize_r_attention(attn_weights, token_seq=list(sentence))