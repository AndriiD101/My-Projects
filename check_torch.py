import torch

# print(torch.cuda.is_available())
# print(torch.cuda.device_count())
# print(torch.cuda.current_device())

print(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
