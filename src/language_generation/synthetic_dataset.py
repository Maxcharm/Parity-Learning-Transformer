import random
import torch
from torch.utils.data import Dataset

class BalancedSparseDNADataset(Dataset):
    def __init__(self, n_samples=2000, seq_len=50, num_rules=4, seed=None):
        self.bases = ['A', 'C', 'G', 'T']
        self.base2idx = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
        self.seq_len = seq_len
        self.data = []

        if seed is not None:
            random.seed(seed)

        # Select random positions for the rule
        self.positions = sorted(random.sample(range(seq_len), num_rules))
        self.rule_description = (
            f"Even number of 'A' or 'T' in positions {self.positions}"
        )

        # Rule function
        def rule_fn(seq):
            count = sum(seq[i] in {'A', 'T'} for i in self.positions)
            return int(count % 2 == 0)
        self.rule_fn = rule_fn

        # Generate balanced dataset
        positives = []
        negatives = []
        max_each = n_samples / 2
        while len(positives) < max_each or len(negatives) < max_each:
            seq = [random.choice(self.bases) for _ in range(seq_len)]
            label = rule_fn(seq)
            if label == 1 and len(positives) < max_each:
                positives.append((seq, label))
            elif label == 0 and len(negatives) < max_each:
                negatives.append((seq, label))
        
        self.data = positives + negatives
        random.shuffle(self.data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        seq, label = self.data[idx]
        encoded_seq = torch.tensor([self.base2idx[base] for base in seq], dtype=torch.long)
        label = torch.tensor(label, dtype=torch.float)
        return encoded_seq, label


def generate_polynomial_dataset(n, k, num_samples=1000, seed=None):
    coeffs = torch.Tensor([1, -1] * int(k/2)).long()
    if seed is not None:
        torch.manual_seed(seed)
    
    X = torch.randint(0, 10, (num_samples, n))
    indices = torch.randperm(n)[:k].long() 
    y = X[:, indices] @ coeffs               

    CLS_ID = 10
    cls_column = torch.full((num_samples, 1), CLS_ID, dtype=torch.long)
    X = torch.cat([cls_column, X], dim=1)

    return X, y

