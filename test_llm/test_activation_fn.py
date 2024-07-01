import torch
import torch.nn as nn
import torch.nn.functional as F


class SwiGLU(nn.Module):
    def forward(self, x):
        x, gate = x.chunk(2, dim=-1)
        return F.silu(gate) * x


x = torch.tensor([[1, 2, 3, 1], [2, 3, 1, 2]]).to(torch.float16).to("cuda:0")

f = SwiGLU()
f(x)
