import torch

def get_device():
    if torch.backends.mps.is_available():
        print("Using GPUs with MPS (Apple Silicon)")
        return torch.device("mps")
    elif torch.cuda.is_available():
        print("Using GPUs with CUDA (NVIDIA)")
        return torch.device("cuda")
    else:
        print("Using CPU")
        return torch.device("cpu")