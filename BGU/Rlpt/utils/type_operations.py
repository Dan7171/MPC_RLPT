import torch
import numpy as np
def torch_tensor_to_ndarray(torch_tensor:torch.Tensor):
    return torch_tensor.cpu().numpy() if torch_tensor.is_cuda else torch_tensor.numpy()

def as_2d_tensor(l,device="cuda", dtype=torch.float64):
    return as_1d_tensor(l,device,dtype).unsqueeze(0)

def as_1d_tensor(l,device="cuda", dtype=torch.float64):
    return torch.tensor(l, device=device, dtype=dtype)
