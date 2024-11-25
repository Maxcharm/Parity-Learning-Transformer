# %%
import torch.nn as nn
import torch
from torchmetrics import HingeLoss
import random
import math
from hard_attention import Attention
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from matplotlib import pyplot as plt
import pickle
from datetime import datetime
# %%


def simple_embedding(x):
    length = len(x)
    token_embedding = torch.stack([torch.tensor(x), 1 - torch.tensor(x)], dim=1)

    positions = torch.arange(length, dtype=torch.float32) * (2 * math.pi / length)
    positional_embeddings = torch.stack([torch.cos(positions), torch.sin(positions)], dim=1)
    encoded_tensor = torch.cat([token_embedding, positional_embeddings], dim=1)
    return encoded_tensor


def data_generator(
        num_data: int,
        k: int = 4,
        n: int = 20,
        ):
    parity_bits = random.sample(range(n), k)
    # parity_bits = [2, 4, 15]
    num = 2 ** n
    x = torch.zeros((num, n), dtype=torch.float32)
    for i in range(num):
        x[i] = torch.tensor(list(map(int, bin(i)[2:].zfill(n))), dtype=torch.float)
    y = x[:, parity_bits].sum(dim=1) % 2
    # y = - 2 * y + 1
    y = y.reshape(-1, 1)
    x_embeddings = torch.stack([simple_embedding(row) for row in x])
    indices = torch.randperm(num)[:num_data]
    data_embeddings = x_embeddings[indices]
    labels = y[indices]

    return x_embeddings, y, data_embeddings, labels, parity_bits


def collect_attention(heads, true_bits, sample):
    length = sample.shape[1]
    v0 = torch.zeros(length)
    v0[0] = 1
    weights = []
    for head in heads:
        Av0 = torch.matmul(v0, head.A)
        scores = torch.matmul(sample, Av0.unsqueeze(-1)).squeeze(-1)
        attention_weights = F.softmax(scores, dim=-1)
        weights.append(attention_weights)
    return weights


class parity_NN(nn.Module):
    def __init__(self, k: int = 3) -> None:
        super().__init__()
        self.attention_heads = [Attention(dim=4) for _ in range(k)]
        self.network = nn.Sequential(
            nn.Linear(k*4, k),
            nn.ReLU(),
            nn.Linear(k, 1),
            )
        self.freeze_params()
        self.initialize_params(k)

    def initialize_params(self, k):
        with torch.no_grad():
            first_layer = self.network[0]
            weight_pattern = torch.zeros(k, 4 * k)

            for i in range(k):
                weight_pattern[i, :] = torch.tensor([1, 0, 0, 0] * k)

            first_layer.weight.data = weight_pattern
            first_layer.bias.data = -torch.arange(k).float() - 0.5

            second_layer = self.network[2]
            weights = torch.tensor([((-1) ** i) * (2 + 4 * i) for i in range(k)], dtype=torch.float32)
            second_layer.weight.data = weights.view(1, -1)  # Shape: (k, 1)
            second_layer.bias.data.zero_()

    def freeze_params(self):
        for param in self.network.parameters():
            param.requires_grad = False

    def forward(self, x):
        attention_vectors = []
        for head in self.attention_heads:
            attn_output = head(x)
            attention_vectors.append(attn_output)
        attention_vectors = torch.concat(attention_vectors, dim=1)
        return self.network(attention_vectors)


def visualize_weights(weights, true_bits):
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    weights = np.array(weights)
    num_heads = weights.shape[1]
    fig, axes = plt.subplots(1, num_heads, figsize=(15, 5), sharey=True)
    for head_idx in range(num_heads):
        ax = axes[head_idx]
        head_scores = weights[:, head_idx, :]
        heatmap = ax.imshow(head_scores, aspect="auto", cmap="viridis",
                            origin="lower")
        ax.set_title(f"Head {head_idx + 1}")
        ax.set_xlabel("Attention score for each position")
        ax.set_ylabel("Epoch")
        shifted_labels = [pos + 0.5 for pos in true_bits]
        ax.set_xticks(shifted_labels)
        ax.set_xticklabels([f"{pos}" for pos in true_bits], rotation=0)
        ax.tick_params(axis="x", which="both", length=0)
        cbar = fig.colorbar(heatmap, ax=ax, orientation="vertical")
        cbar.set_label("Attention Score")
    plt.tight_layout()
    plt.savefig(f"{len(true_bits)}_bits_{current_time}.jpg")


def test(model, x, y):
    pred = model(x)
    predicted_classes = (pred >= 0.5).float()
    correct_predictions = (predicted_classes == y).sum()
    accuracy = correct_predictions / y.size(0)
    return accuracy.item()


if __name__ == "__main__":
    number_of_data = int(2**20 * 0.8)
    k = 5
    epochs = 30
    batch_size = 8000
    loss_fn = HingeLoss(task="binary")
    # loss_fn = nn.MSELoss()
    x, y, data, label, bits = data_generator(number_of_data, k)
    dataset = TensorDataset(data, label)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    parity_network = parity_NN(k)
    attention_params = []
    for i in range(k):
        attention_params += list(parity_network.attention_heads[i].parameters())
    total_weights = []
    with torch.no_grad():
        total_weights.append(collect_attention(parity_network.attention_heads, bits, data[0]))
    optimizer = torch.optim.Adam(attention_params, lr=8e-2)
    parity_network.train()
    for i in range(epochs):
        total_loss = 0
        for batch_idx, (inputs, targets) in enumerate(dataloader):
            pred = parity_network(inputs)
            loss = loss_fn(pred, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        # if i % 2 == 0:
        print(f"epoch: {i}, loss: {loss:>7f}.")
        with torch.no_grad():
            total_weights.append(collect_attention(parity_network.attention_heads, bits, data[0]))
    with open("weights_test.pkl", "wb") as f:
        pickle.dump(total_weights, f)
    visualize_weights(weights=total_weights, true_bits=bits)
    parity_network.eval()
    print(test(parity_network, x[:2000], y[:2000]))
