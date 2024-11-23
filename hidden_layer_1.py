# %%
import torch.nn as nn
import torch
from torchmetrics import HingeLoss
import random
import math
from hard_attention import Attention
import torch.nn.functional as F
# %%


def simple_embedding(x):
    length = len(x)
    token_embedding = torch.stack([torch.tensor(x), 1 - torch.tensor(x)], dim=1)

    positions = torch.arange(length, dtype=torch.float32) * (2 * math.pi / length)
    positional_embeddings = torch.stack([torch.cos(positions), torch.sin(positions)], dim=1)
    encoded_tensor = torch.cat([token_embedding, positional_embeddings], dim=1) 
    return encoded_tensor

# %%


def data_generator(
        num_data: int,
        k: int = 3,
        n: int = 16,
        ):
    # parity_bits = random.sample(range(n), k)
    parity_bits = [2, 4, 15]
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


# %%
def visualize_attention(heads, true_bits, sample):
    length = sample.shape[1]
    v0 = torch.zeros(length)
    v0[0] = 1
    print(f"the true bits are {true_bits}.")
    for i, head in enumerate(heads):
        Av0 = torch.matmul(v0, head.A)
        scores = torch.matmul(sample, Av0.unsqueeze(-1)).squeeze(-1)
        attention_weights = F.softmax(scores, dim=-1) 
        maximum_position = torch.argmax(scores, dim=-1)
        print(f"The {i+1}-th head attend: {maximum_position}-th bit with score {attention_weights[maximum_position]:.2f}.")
    print("--------------------------------")


# %%
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
# %%


def test(model, x, y):
    pred = model(x)
    predicted_classes = (pred >= 0.5).float()
    correct_predictions = (predicted_classes == y).sum()
    accuracy = correct_predictions / y.size(0)
    return accuracy.item()


if __name__ == "__main__":
    number_of_data = 50000
    k = 3
    epochs = 40000
    loss_fn = HingeLoss(task="binary")
    # loss_fn = nn.MSELoss()
    x, y, data, label, bits = data_generator(number_of_data, k)
    parity_network = parity_NN(k)
    attention_params = (
        list(parity_network.attention_heads[0].parameters()) +
        list(parity_network.attention_heads[1].parameters()) +
        list(parity_network.attention_heads[2].parameters())
    )
    ffnn_params = list(parity_network.network.parameters())
    with torch.no_grad():
        visualize_attention(parity_network.attention_heads, bits, data[0])
    optimizer = torch.optim.Adam(attention_params, lr=8e-2)
    parity_network.train()
    for i in range(epochs):
        pred = parity_network(data)
        loss = loss_fn(pred, label)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        loss = loss.item()
        if i % 2000 == 0:
            print(f"epoch: {i}, loss: {loss:>7f}.")
            with torch.no_grad():
                visualize_attention(parity_network.attention_heads, bits, data[0])
            # for param in attention_params:
            #     print(param)
    parity_network.eval()
    print(test(parity_network, x, y))
# %%
