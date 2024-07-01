import torch

seed = 42
torch.manual_seed(seed)

weights_1 = torch.ones(2, 2, requires_grad=True, dtype=torch.float16)
print(weights_1)
weights_2 = torch.tensor([2.0], requires_grad=True)
print(weights_2)

y = weights_2 * weights_1 + 2
print(y)

z = torch.sum(torch.pow(y, 2))
print(z)
z.backward()
print(weights_1.grad)
print(weights_2.grad)
