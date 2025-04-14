# %%
import torch.nn as nn
import torch
from torchmetrics import HingeLoss
from torchmetrics.classification import BinaryHingeLoss
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
    token_embedding = torch.stack(
        [torch.tensor(x), 1 - torch.tensor(x)], dim=1
        )

    positions = torch.arange(
        length, dtype=torch.float32
        ) * (2 * math.pi / length)
    positional_embeddings = torch.stack(
        [torch.cos(positions), torch.sin(positions)], dim=1
        )
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
        x[i] = torch.tensor(
            list(map(int, bin(i)[2:].zfill(n))), dtype=torch.float
            )
    y = x[:, parity_bits].sum(dim=1) % 2
    # y = - 2 * y + 1
    y = y.reshape(-1, 1)
    x_embeddings = torch.stack([simple_embedding(row) for row in x])
    indices = torch.randperm(num)[:num_data]
    data_embeddings = x_embeddings[indices]
    labels = y[indices]

    return x_embeddings, y, data_embeddings, labels, parity_bits


def collect_attention(heads, sample):
    length = sample.shape[1]
    v0 = torch.zeros(length)
    v0[0] = 1
    weights = []
    for head in heads:
        Av0 = torch.matmul(v0, head.A)
        scores = torch.matmul(sample, Av0.unsqueeze(-1)).squeeze(-1)
        # temperature = 1 / 20
        # scores /= temperature
        attention_weights = F.softmax(scores, dim=-1)
        weights.append(attention_weights)
    return weights

def print_initialisation_information(heads, sample, true_bits):
    print("-------------init information------------")
    print(f"the true bits are {true_bits}.")
    length = sample.shape[1]
    v0 = torch.zeros(length)
    v0[0] = 1
    for i, head in enumerate(heads):
        Av0 = torch.matmul(v0, head.A)
        scores = torch.matmul(sample, Av0.unsqueeze(-1)).squeeze(-1)
        attention_weights = F.softmax(scores, dim=-1)
        indexed_scores = [(weight, i) for i, weight in enumerate(attention_weights)]
        indexed_scores.sort(reverse=True, key=lambda x: x[0])
        rank_map = {idx: rank for rank, (_, idx) in enumerate(indexed_scores, start=1)}
        ranked_positions = sorted(true_bits, key=lambda pos: rank_map[pos])
        print(f"the rank of head {i} is: {ranked_positions}.")
    
    return ranked_positions
class parity_NN(nn.Module):
    def __init__(self, k: int = 3) -> None:
        super().__init__()
        self.attention_heads = [Attention(dim=4) for _ in range(k)]
        self.network = nn.Sequential(
            nn.Linear(4, k),
            nn.GELU(),
            nn.Linear(k, 1),
            )
        self.freeze_params()
        self.initialize_params(k)

    def initialize_params(self, k):
        with torch.no_grad():
            first_layer = self.network[0]
            weight_pattern = torch.zeros(k, 4)

            for i in range(k):
                weight_pattern[i, :] = torch.tensor([k, 0, 0, 0])

            first_layer.weight.data = weight_pattern
            first_layer.bias.data = -torch.arange(k).float() - 0.5

            second_layer = self.network[2]
            weights = torch.tensor(
                [((-1) ** i) * (2 + 2 * i) for i in range(k)],
                dtype=torch.float32
                )
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
        stacked_tensors = torch.stack(attention_vectors)
        attention_vectors = stacked_tensors.mean(dim=0)
        # attention_vectors = torch.concat(attention_vectors, dim=1)
        return self.network(attention_vectors)


# %%
def visualize_weights(weights, true_bits):
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    weights = np.array(weights)
    num_heads = weights.shape[1]
    fig, axes = plt.subplots(num_heads, 1, figsize=(14, 10), sharex=True)
    # Create a shared color bar axis
    cbar_ax = fig.add_axes([0.90, 0.15, 0.02, 0.7])

    for head_idx, ax in enumerate(axes):
        head_scores = weights[:, head_idx, :].T
        heatmap = ax.imshow(
            head_scores, aspect="auto", cmap="viridis", origin="lower"
            )
        ax.set_title(f"Head {head_idx + 1}", fontsize=25, loc="right")
        shifted_positions = [pos + 0.5 for pos in true_bits]
        ax.set_yticks(shifted_positions)
        ax.set_yticklabels([str(pos) for pos in true_bits], fontsize=14)
    axes[-1].set_xlabel("Epoch", fontsize=25)
    for ax in axes:
        ax.tick_params(axis="x", labelsize=20)
    cbar = fig.colorbar(
        heatmap, cax=cbar_ax, orientation="vertical", label="Attention Score"
        )
    cbar.ax.tick_params(labelsize=14)
    cbar.ax.set_ylabel("Attention Score", fontsize=25)
    # fig.text(
    #     0.02, 0.5, 'Attention score for each position', va='center',
    #     ha='center', rotation='vertical', fontsize=25)
    plt.tight_layout(rect=[0, 0, 0.9, 1])
    plt.savefig(f"pictures/{len(true_bits)}_bits_{current_time}.jpg")
# %%


def test(model, x, y):
    pred = model(x)
    print(y[:5])
    print(pred[:5])
    predicted_classes = (pred >= 0.5).float()
    correct_predictions = (predicted_classes == y).sum()
    accuracy = correct_predictions / y.size(0)
    return accuracy.item()


if __name__ == "__main__":
    length = 30
    number_of_data = int(2**length * 0.8)
    k = 2
    epochs = 30
    batch_size = 12000
    loss_fn = HingeLoss(task="binary")
    # loss_fn = BinaryHingeLoss(squared=True)
    # loss_fn = nn.MSELoss()
    x, y, data, label, bits = data_generator(number_of_data, k)
    dataset = TensorDataset(data, label)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    parity_network = parity_NN(k)
    attention_params = []
    for i in range(k):
        attention_params += list(
            parity_network.attention_heads[i].parameters()
            )
    total_weights = []
    with torch.no_grad():
        print_initialisation_information(parity_network.attention_heads, data[0], bits)
        total_weights.append(
            collect_attention(parity_network.attention_heads, data[0])
            )
    optimizer = torch.optim.Adam(attention_params, lr=8e-2)
    parity_network.train()
    for i in range(epochs):
        total_loss = 0
        for batch_idx, (inputs, targets) in enumerate(dataloader):
            pred = parity_network(inputs)
            loss = loss_fn(pred, targets)
            # loss = loss_fn(pred, targets, parity_network)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        # if i % 2 == 0:
        print(f"epoch: {i}, loss: {loss:>7f}.")
        with torch.no_grad():
            total_weights.append(
                collect_attention(parity_network.attention_heads, data[0])
                )
    # with open("weights_test.pkl", "wb") as f:
    #     pickle.dump(total_weights, f)
    visualize_weights(weights=total_weights, true_bits=bits)
    parity_network.eval()
    print(test(parity_network, x[:2000], y[:2000]))
    # print("optimal params for A saved")
    sum_of_norms = 0
    for head in parity_network.attention_heads:
        sum_of_norms += torch.norm(head.A)
    print(f"the sum of the norms of the attention heads is: {sum_of_norms}.")
        
