import torch

def inverse_sigmoid(x, eps=1e-5):
    return torch.logit(x, eps=eps)