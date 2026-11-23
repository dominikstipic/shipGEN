import torch
import matplotlib.pyplot as plt

from lurssen.dataset import get_dataloader, RPlanDataset
import matplotlib.pyplot as plt
from torch.utils.data import Dataset

def test_dataset(dataset, n_samples=10):
    for i in range(min(n_samples, len(dataset))):
        img, house_mask, sizes, neighborhood = dataset[i]

        assert torch.isfinite(img).all(), f"Invalid image at {i}"
        assert torch.isfinite(house_mask).all(), f"Invalid mask at {i}"

        assert img.ndim == 3, f"Unexpected image dimensions: {img.shape}"
        assert house_mask.ndim in (2, 3)

        assert img.shape[-2:] == house_mask.shape[-2:], \
            f"Image and mask dimensions differ at {i}"

        assert (sizes >= 0).all(), f"Negative room sizes at {i}"

    print(f"Dataset test passed: {min(n_samples, len(dataset))} samples")


dataset = RPlanDataset(
        split="train",
        return_graph=True,
        hf_name="metindeder/rplan-floorplan-edited"
    )
test_dataset(dataset)