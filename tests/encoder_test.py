from torch_geometric.data import Data
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from lurssen.encoder import RoomGenEncoder 
from lurssen.dataset import RPlanDataset

dataset = RPlanDataset()
encoder = RoomGenEncoder()
img, house_mask, room_sizes, neighboorhood_graph = dataset[1]

print(room_sizes)
print(neighboorhood_graph)
print(f"img features: {img.shape}")
print(f"house features: {house_mask.shape}")
print(f"sizes features: {room_sizes.shape}")
print(f"neighboorhood features: {neighboorhood_graph.shape}")
cond = encoder(house_mask.unsqueeze(0), room_sizes, neighboorhood_graph)
print(f"out features: {cond.shape}")