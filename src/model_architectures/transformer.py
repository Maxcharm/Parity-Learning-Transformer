import torch
import torch.nn as nn
import torch.nn.functional as F

def add_cls_token(x, cls_idx, vocab_size):
    if cls_idx is None:
        cls_idx = vocab_size
    cls_column = torch.full((x.size(0), 1), cls_idx, dtype=torch.long, device=x.device)
    return torch.cat([cls_column, x], dim=1)

class Attention(nn.Module):
    def __init__(self, dim):
        super(Attention, self).__init__()
        self.A = nn.Parameter(torch.zeros(dim, dim))
        theta = torch.rand(1) * 2 * torch.pi
        with torch.no_grad():
            self.A[0, 2] = torch.cos(theta)
            self.A[0, 3] = torch.sin(theta)

        # Non-trainable buffers
        v0 = torch.zeros(dim)
        v0[0] = 1
        self.register_buffer("v0", v0)

        mask = torch.zeros(dim, dim)
        mask[0, 2] = 1
        mask[0, 3] = 1
        self.register_buffer("mask", mask)
        

    def forward(self, v):
        # print(f"v0 is on {self.v0.device} and A is on {self.A.device}.")
        A_masked = self.A * self.mask
        Av0 = torch.matmul(self.v0, A_masked)
        temperature = v.shape[1] / 4
        scores = torch.matmul(v, Av0.unsqueeze(-1)).squeeze(-1)
        
        scores /= temperature
        attention_weights = F.softmax(scores, dim=-1)
        weighted_sum = torch.matmul(attention_weights.unsqueeze(1), v).squeeze(1)
        return weighted_sum


class transformer_SL(nn.Module):
    def __init__(self, k: int = 3) -> None:
        super().__init__()
        self.attention_heads = nn.ModuleList([Attention(dim=4) for _ in range(k)])
        self.network = nn.Sequential(
            nn.Linear(4, k),
            nn.ReLU(),
            nn.Linear(k, 1),
            )
        self.initialize_params(k)
        self.freeze_params()
        

    def initialize_params(self, k):
        with torch.no_grad():
            first_layer = self.network[0]
            weight_pattern = torch.zeros(k, 4)

            for i in range(k):
                weight_pattern[i, :] = torch.tensor([k, 0, 0, 0])

            first_layer.weight.data = weight_pattern
            first_layer.bias.data = - torch.arange(k).float() - 0.5

            second_layer = self.network[2]
            weights = torch.tensor(
                [((-1) ** i) * (2 + 4 * i) for i in range(k)],
                dtype=torch.float32
                )
            second_layer.weight.data = weights.view(1, -1)
            second_layer.bias.data = torch.Tensor([0])

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
        results = self.network(attention_vectors)
        return results

class transformer_encoder(nn.Module):
    def __init__(self, d_model, nhead, ff_dim):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, batch_first=True)
        self.linear1 = nn.Linear(d_model, ff_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.1)
        self.linear2 = nn.Linear(ff_dim, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(0.1)
        self.dropout2 = nn.Dropout(0.1)

    def forward(self, src, src_mask=None, src_key_padding_mask=None, is_causal=None):
        attn_output, attn_weights = self.self_attn(
            src, src, src,
            attn_mask=src_mask,
            key_padding_mask=src_key_padding_mask,
            need_weights=True,
            is_causal=is_causal,
            average_attn_weights=False
        )
        self.attn_weights = attn_weights 
        src = src + self.dropout1(attn_output)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.relu((self.linear1(src)))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src

class transformer_ml(nn.Module):
    def __init__(
        self,
        vocab_size,
        seq_len,
        d_model=32,
        nhead=2,
        ff_dim=64,
        num_layers=1,
        task="classification",
        cls_token_idx=None
    ):
        super().__init__()
        self.task = task
        self.cls_token_idx = cls_token_idx if cls_token_idx is not None else vocab_size

        self.embedding = nn.Embedding(vocab_size + 1, d_model)
        self.pos_embedding = nn.Parameter(torch.randn(1, seq_len + 1, d_model))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=ff_dim,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        head = [
            nn.Linear(d_model, ff_dim),
            nn.ReLU(),
            nn.Linear(ff_dim, 1)
        ]
        if task == "classification":
            head.append(nn.Sigmoid())

        self.head = nn.Sequential(*head)

    def forward(self, x):
        x = add_cls_token(x, self.cls_token_idx, self.embedding.num_embeddings - 1)
        x = self.embedding(x) + self.pos_embedding[:, :x.size(1), :]
        x = self.encoder(x)
        cls_repr = x[:, 0, :]
        return self.head(cls_repr).squeeze(1)
