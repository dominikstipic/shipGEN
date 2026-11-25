
from .gcn import RoomGraphEncoder
from .dataset import RPlanDataset
from .outershell import ShipGeometryEncoder
from .cross_attention import ImageGraphFusion
from torch_geometric.data import Data
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from .adapter import ConditionAdapter 


class RoomGenEncoder(nn.Module):
    
    def __init__(self):
        super().__init__()
        self.gcn_model = RoomGraphEncoder(input_dim=1, 
                                          hidden_dim=64, 
                                          output_dim=128)
        self.encoder = ShipGeometryEncoder(latent_dim=512)
        self.gcn_norm = nn.Sequential(
            nn.LayerNorm(128),
            nn.Tanh()
        )
        self.geometry_norm = nn.Sequential(
            nn.GroupNorm(num_groups=32, num_channels=512),
            nn.Tanh()
        )
        self.fusion = ImageGraphFusion()
        self.fusion_norm = nn.Sequential(
           nn.GroupNorm(32, 512),
           nn.Tanh()
        )
        self.adapter = ConditionAdapter(
                in_channels=512,
                out_channels=64)

    def forward(self, outershell_mask, node_features, edge_features):
        graph = Data(
            x=node_features,
            edge_index=edge_features
        )
        graph_features = self.gcn_model(graph.x, 
                                        graph.edge_index)
        #print("GCN before:", graph_features.min().item(), graph_features.max().item())
        graph_features = self.gcn_norm(graph_features)
        #print("GCN after:", graph_features.min().item(), graph_features.max().item())
        _, outtershell_features = self.encoder(outershell_mask)
        #print("MASK before:", graph_features.min().item(), graph_features.max().item())
        outtershell_features = self.geometry_norm(outtershell_features)
        #print("MASK after:", graph_features.min().item(), graph_features.max().item())
        output = self.fusion(outtershell_features, graph_features)
        #print("output BEFORE:", output.min().item(), output.max().item())
        output = self.fusion_norm(output)
        #print("output AFTER:", output.min().item(), output.max().item())
        condition_features = self.adapter(output,image_size=256)
        return condition_features


if __name__ == "__main__":
    dataset = RPlanDataset()
    encoder = RoomGenEncoder()
    img, house_mask, room_sizes, neighboorhood_graph = dataset[1]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].imshow(img.permute(1,2,0), cmap="gray")
    axes[0].set_title("Original Image")
    axes[0].axis("off")

    axes[1].imshow(house_mask.squeeze(), cmap="gray")
    axes[1].set_title("House Mask")
    axes[1].axis("off")

    plt.tight_layout()
    plt.show()

    print(room_sizes)
    print(neighboorhood_graph)
    print(f"img features: {img.shape}")
    print(f"house features: {house_mask.shape}")
    print(f"sizes features: {room_sizes.shape}")
    print(f"neighboorhood features: {neighboorhood_graph.shape}")
    cond = encoder(house_mask.unsqueeze(0), room_sizes, neighboorhood_graph)
    print(f"out features: {cond.shape}")

