import torch

print(torch.cuda.is_available())

print("device is ", torch.device)


# Create a tensor on CPU
x = torch.randn(100, 100)

# Move the tensor to GPU
x = x.to('cuda')
print("done")
