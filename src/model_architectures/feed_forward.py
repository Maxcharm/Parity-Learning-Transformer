import torch.nn as nn

class FFNN_regressor(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        x = x.float()
        return self.net(x).squeeze(1)


class FFNN_classifier(nn.Module):
    def __init__(self, vocab_size, seq_len, hidden_dim, d_model=4):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.mlp = nn.Sequential(
            nn.Linear(seq_len * d_model, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.embedding(x)
        x = x.flatten(start_dim=1) 
        return self.mlp(x).squeeze(1)