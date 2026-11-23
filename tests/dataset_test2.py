import matplotlib.pyplot as plt

from lurssen.dataset import get_dataloader, RPlanDataset
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from torchvision import transforms


dataset = RPlanDataset(
        split="train",
        return_graph=True,
        hf_name="metindeder/rplan-floorplan-edited"
    )


img, house_mask, sizes, neighborhood = dataset[0]
print(img.shape)
print(min(img.flatten()), max(img.flatten()))
print(min(house_mask.flatten()), max(house_mask.flatten()))