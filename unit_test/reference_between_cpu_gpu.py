import torch
from torch import nn

device = "cpu"
# cpu
# x = nn.Parameter(torch.tensor([1, 2, 3, 4], device=device).to(torch.float))
x = torch.tensor([1, 2, 3, 4], device=device).to(torch.float)
print(x.requires_grad)
x.requires_grad_()
w = torch.tensor(3.0, requires_grad=True, device=device)

x_sub = x[1:3]
# print(x.is_leaf, x.grad)
# z = w * x_sub
# z.sum().backward()
# z = w * x_sub
# z.sum().backward()
# print(x.is_leaf, x.grad)

# move to gpu

# x = x.cuda()
# x.retain_grad()
print(x.is_leaf, x.device, x.requires_grad, x.grad)
# x.requires_grad_()
w = w.cuda()
# x_sub = x[1:3]
x_sub = x_sub.cuda()
x_sub.requires_grad_()
z = w * x_sub
# x.retain_grad()
# x_sub.retain_grad()
# w.retain_grad()
z.sum().backward()

print(
    x_sub.requires_grad,
    x.requires_grad,
    x_sub.device,
    w.device,
    x.device,
    x.grad,
    x_sub.grad,
)
b = x.cuda()
print(x.grad, b.grad)
z = w * b
z.sum().backward()
print(x.grad)
