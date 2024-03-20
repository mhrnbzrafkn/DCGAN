import torch


device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(f'00- Using {device}.')

print(torch.cuda.device_count())
print(torch.cuda.get_device_name(0))