import torch

def check_tensor(name, x):
    print(f"\n--- {name} ---")
    print("Shape:", x.shape)
    print("Min:", x.min().item())
    print("Max:", x.max().item())
    print("Mean:", x.mean().item())
    print("Std:", x.std().item())
    print("NaN:", torch.isnan(x).any().item())
    print("Inf:", torch.isinf(x).any().item())