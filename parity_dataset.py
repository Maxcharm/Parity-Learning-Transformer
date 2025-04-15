# %%
import torch
from torch.utils.data import Dataset, DataLoader
import random
import math

# %%
def simple_embedding(x):
    length = len(x)
    token_embedding = torch.stack([torch.tensor(x), 1 - torch.tensor(x)], dim=1)

    positions = torch.arange(length, dtype=torch.float32) * (2 * math.pi / length)
    positional_embeddings = torch.stack([torch.cos(positions), torch.sin(positions)], dim=1)
    encoded_tensor = torch.cat([token_embedding, positional_embeddings], dim=1)
    return encoded_tensor

# %%
class DynamicBinaryStringDataset(Dataset):
    def __init__(
            self,
            length,
            parity_bits=None,
            parity_size=None
            ):
        self.length = length
        if parity_bits is not None:
            self.parity_bits = parity_bits
        else:
            assert parity_size is not None, "either specify the parity bits or the parity size."
            self.parity_bits = random.sample(range(length), parity_size)

    def __len__(self):
        return int(1e12)

    def __getitem__(self, idx):
        binary_tensor = torch.randint(0, 2, (self.length,))
        embedding = simple_embedding(binary_tensor)
        label = binary_tensor[self.parity_bits].sum(dim=1) % 2
        label = label.reshape(-1, 1)
        return embedding, label


# %%
binary_length = 10
batch_size = 2

dataset = DynamicBinaryStringDataset(length=binary_length, parity_size=3)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

for embeddings, labels in dataloader:
    print(f"Batch embeddings shape: {embeddings.shape}")
    print(f"Batch labels shape: {labels.shape}")
    break
# %%
