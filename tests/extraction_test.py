from lurssen.dataset import get_dataloader, RPlanDataset
import matplotlib.pyplot as plt
from torch.utils.data import Dataset

dataset = RPlanDataset(
        split="train",
        return_graph=True,
        hf_name="metindeder/rplan-floorplan-edited"
    )


img, house_mask, sizes, neighboorhood = dataset[0]


fig, axes = plt.subplots(1, 2, figsize=(12, 6))

# Original image
axes[0].imshow(img.permute(1, 2, 0))
axes[0].set_title("Original Image")
axes[0].axis("off")

# House mask
axes[1].imshow(house_mask.squeeze(), cmap="gray")
axes[1].set_title("House Mask")
axes[1].axis("off")


plt.tight_layout()
plt.show()

print(sizes)
print(neighboorhood)

