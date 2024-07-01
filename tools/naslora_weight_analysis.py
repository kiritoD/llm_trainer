import torch

a = torch.tensor([2])
b_weights = torch.tensor([1.1, 1.2, 1.3, 1.4, 0.9])
b_1 = lambda x: torch.pow(x, 3)
b_2 = lambda x: torch.pow(x, 2)
b_3 = lambda x: torch.pow(x, 4)
b_4 = lambda x: torch.sigmoid(x)
b_5 = lambda x: torch.pow(x, 1.5)

b_weights_softmax = torch.nn.functional.softmax(b_weights)
print(f"softmax: {b_weights_softmax}")
b_1_results = [b_1(a), b_2(a), b_3(a), b_4(a), b_5(a)]
# print(b_1_results)

b_results = torch.cat(b_1_results)
print(f"results: {b_results}")

final_result = torch.sum(b_results * b_weights_softmax)
print(final_result)
b_grad = b_weights_softmax * (b_results - final_result)
print(b_grad)
