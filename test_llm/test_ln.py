import torch
import torch.nn as nn

if __name__ == "__main__":
    batch, sentence_length, embedding_dim = 2, 2, 5
    embedding = torch.randn(batch, sentence_length, embedding_dim)
    print(embedding)
    layer_norm = nn.LayerNorm(embedding_dim)
    activate_ln = layer_norm(embedding)
    print(activate_ln)
