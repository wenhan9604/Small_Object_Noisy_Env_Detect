import random, os
import numpy as np
import torch
import torchvision.utils as vutils


def get_device():
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using MPS.")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using CUDA (gpu: {torch.cuda.get_device_name(0)}).")
    else:
        device = torch.device("cpu")
        print("Using CPU")
    return device


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False