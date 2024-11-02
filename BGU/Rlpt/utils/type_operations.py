import torch
import numpy as np
def torch_tensor_to_ndarray(torch_tensor:torch.Tensor):
    return torch_tensor.cpu().numpy() if torch_tensor.is_cuda else torch_tensor.numpy()