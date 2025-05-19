# %%
import torch.nn as nn
import torch
import random
import math
from src.model_architectures.transformer import transformer_SL
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from matplotlib import pyplot as plt
from datetime import datetime
import os
from fire import Fire

class MyHingeLoss(nn.Module):

    def __init__(self):
        super(MyHingeLoss, self).__init__()

    def forward(self, output, target):
        y_hat = output * 2 - 1
        y_true = target * 2 - 1
        hinge_loss = 1 - torch.mul(y_hat, y_true)
        hinge_loss = torch.clamp(hinge_loss, min=0)
        return (hinge_loss.mean())**2
    
def simple_embedding(x):
    length = len(x)
    token_embedding = torch.stack(
        [x.clone(), (1 - x).clone()], dim=1
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
    x = torch.randint(0, 2, (num_data, n), dtype=torch.float32)
    y = x[:, parity_bits].sum(dim=1) % 2
    y = y.reshape(-1, 1)
    x_embeddings = torch.stack([simple_embedding(row) for row in x])

    return x_embeddings, y, parity_bits

def collect_attention(heads, sample):
    device = sample.device
    length = sample.shape[1]
    v0 = torch.zeros(length).to(device)
    v0[0] = 1
    weights = []
    for head in heads:
        Av0 = torch.matmul(v0, head.A)
        scores = torch.matmul(sample, Av0.unsqueeze(-1)).squeeze(-1)
        attention_weights = F.softmax(scores, dim=-1).detach().cpu()
        weights.append(attention_weights)
    return weights

def visualize_weights(weights, true_bits, log_dir):
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
    fig.tight_layout(rect=[0, 0, 0.9, 1])
    # plt.show()
    file_path = os.path.join(log_dir, f"{len(true_bits)}_bits_{current_time}.jpg")
    plt.savefig(file_path)

def test(model, x, y):
    pred = model(x)
    predicted_classes = (pred >= 0.5).float()
    correct_predictions = (predicted_classes == y).sum()
    accuracy = correct_predictions / y.size(0)
    return accuracy.item()

def main(
        number_of_samples:int=None,
        device:str = None,
        length:int = 16,
        k:int = 3,
        batch_size:int = 80000,
        early_stop:bool = False,
        learning_rate:float = 1e-1,
        patience:int = 10,
        min_delta:float = 1e-4,
        epochs:int = 2000,
        visualize_attention:bool = True,
        save_loss:bool = True,
        log_dir:str = "log/",
):
    '''
        I don't recommend using early stop here because the loss can drop slowly in the first few epochs.
    '''
    # first make the directory if we want to save the loss or the plot.
    if visualize_attention or save_loss:
        os.makedirs(log_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = os.path.join(log_dir, f"run_{timestamp}")
        os.makedirs(run_dir)

    # take half of the instance space for training.
    if number_of_samples is None:
        number_of_data = int(2**length * 0.8)
    
    # 3:1 split for train and validation, only to speed up the validation with a small set size, so potential overlap can be excused.
    data, label, bits = data_generator(number_of_data, k, length)
    train_data, train_label = data[:int(number_of_data * 0.75)], label[:int(number_of_data * 0.75)]
    test_data, test_label = data[int(number_of_data * 0.75) + 1:], label[int(number_of_data * 0.75) + 1:]

    dataset = TensorDataset(train_data, train_label)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    parity_network = transformer_SL(k)

    # specify that only attention matrices should have gradient prop
    attention_params = []
    for i in range(k):
        attention_params += list(
            parity_network.attention_heads[i].parameters()
            )
        
    # self-explanatory from the if statements
    if visualize_attention:
        total_weights = []
        with torch.no_grad():
            total_weights.append(
                collect_attention(parity_network.attention_heads, data[0])
            )
    if save_loss:
        loss_records = []

    # initialize the early stopper
    best_loss = float("inf")
    epochs_wo_improv = 0
    optimizer = torch.optim.Adam(attention_params, lr=learning_rate)
    '''
        best_lr config: (20, 3): 2e-1
    '''
    if device is not None:
        device = torch.device(device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Starting the training on {device} for {epochs} epochs.")
    parity_network.to(device)
    test_data = test_data.to(device)
    test_label = test_label.to(device)
    # loss_fn = HingeLoss(task="binary").to(device)
    loss_fn = MyHingeLoss().to(device)

    for i in range(epochs):
        parity_network.train()
        total_loss = 0
        for batch_idx, (inputs, targets) in enumerate(dataloader):
            inputs = inputs.to(device)
            targets = targets.to(device)
            pred = parity_network(inputs)
            loss = loss_fn(pred, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(dataloader)
        
        if i % 20 == 0:
            parity_network.eval()
            accuracy = test(parity_network, test_data, test_label)
            print(f"epoch: {i}, train loss: {avg_loss:>4f}, test accuracy: {accuracy:>4f}")
        else:
            print(f"epoch: {i}, train loss: {avg_loss:>4f}.")
        if visualize_attention:
            with torch.no_grad():
                total_weights.append(
                    collect_attention(parity_network.attention_heads, data[0].to(device))
                )

        if save_loss:
            with torch.no_grad():
                loss_records.append(avg_loss)

        if early_stop and i > 50:
            if best_loss - avg_loss > min_delta:
                best_loss = avg_loss
                epochs_wo_improv = 0
            else:
                epochs_wo_improv += 1
            if epochs_wo_improv >= patience:
                print(f"Early stopping at epoch {i} — no improvement for {patience} epochs.")
                break
    
    if visualize_attention:
        visualize_weights(weights=total_weights, true_bits=bits, log_dir=run_dir)

    if save_loss:
        time_str = datetime.now().strftime("%Y%m%d-%H%M%S")
        loss_records = np.array(loss_records)
        file_path = os.path.join(run_dir, f"length_{length}_k{k}_{time_str}.npy")
        np.save(file_path, loss_records)

if __name__ == "__main__":
    Fire(main)