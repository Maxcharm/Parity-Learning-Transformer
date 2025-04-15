#%%
import torch
import torch.nn as nn
import torch.nn.functional as F

#%%
class Attention(nn.Module):
    def __init__(self, dim):
        super(Attention, self).__init__()
        self.A = nn.Parameter(torch.zeros(dim, dim))
        self.v0 = torch.zeros(dim)
        self.v0[0] = 1
        theta = torch.rand(1) * 2 * torch.pi
        with torch.no_grad():
            self.A[0, 2] = torch.cos(theta)
            self.A[0, 3] = torch.sin(theta) 
        self.mask = torch.zeros(dim, dim)
        self.mask[0, 2] = 1
        self.mask[0, 3] = 1

    def forward(self, v):
        A_masked = self.A * self.mask
        Av0 = torch.matmul(self.v0, A_masked)
        scores = torch.matmul(v, Av0.unsqueeze(-1)).squeeze(-1)
        attention_weights = F.softmax(scores, dim=-1)
        weighted_sum = torch.matmul(attention_weights.unsqueeze(1), v).squeeze(1)
        return weighted_sum
# %%
