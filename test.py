import torch
import torch.nn as nn
import torch.optim as optim

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Parameters
seq_len = 8
batch_size = 4
vocab_size = 10
embedding_dim = 16
num_heads = 2

# Sample data: batch of sequences of token IDs (just random for demo)
torch.manual_seed(0)
x = torch.randint(0, vocab_size, (seq_len, batch_size)).to(device)
y = x.clone()  # Predict the next token (for simplicity we use the same as input)

# Embedding layer
embedding = nn.Embedding(vocab_size, embedding_dim).to(device)
embedded_x = embedding(x)  # Shape: (seq_len, batch, embed_dim)

# Multihead Attention
mha = nn.MultiheadAttention(embed_dim=embedding_dim, num_heads=num_heads, batch_first=False).to(device)

# Output layer to predict logits over vocab
output_layer = nn.Linear(embedding_dim, vocab_size).to(device)

# Causal mask (prevents attention to future tokens)
# Shape: (seq_len, seq_len), upper triangle filled with -inf
mask = torch.triu(torch.ones(seq_len, seq_len) * float('-inf'), diagonal=1).to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(list(mha.parameters()) + list(embedding.parameters()) + list(output_layer.parameters()), lr=0.01)

# Training loop (very minimal)
for epoch in range(10):
    optimizer.zero_grad()
    
    # Forward pass through attention
    attn_output, attn_weights = mha(embedded_x, embedded_x, embedded_x, attn_mask=mask)
    
    # Predict token logits
    logits = output_layer(attn_output)  # Shape: (seq_len, batch, vocab_size)
    
    # Reshape for loss: merge batch and sequence dims
    logits_flat = logits.view(-1, vocab_size)
    targets_flat = y.view(-1)

    loss = criterion(logits_flat, targets_flat)
    loss.backward()
    optimizer.step()

    print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")
