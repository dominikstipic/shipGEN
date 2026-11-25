import torch
import torch.nn as nn
import torch.nn.functional as F


class ConditionAdapter(nn.Module):

    def __init__(self, in_channels=512, out_channels=64):
        super().__init__()

        self.projection = nn.Sequential(
            nn.Conv2d(in_channels, 256, kernel_size=1),
            nn.GroupNorm(8, 256),
            nn.SiLU(),

            nn.Conv2d(256, out_channels, kernel_size=1),
            nn.GroupNorm(8, out_channels),
            nn.SiLU()
        )

    def forward(self, condition, image_size=256):

        condition = self.projection(condition)

        condition = F.interpolate(
            condition,
            size=(image_size, image_size),
            mode="bilinear",
            align_corners=False
        )

        return condition

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    adapter = ConditionAdapter(
        in_channels=512,
        out_channels=64
    ).to(device)

    adapter.eval()

    # Simulate condition features from your encoder
    condition = torch.randn(1, 512, 8, 8).to(device)

    with torch.no_grad():

        condition_features = adapter(
            condition,
            image_size=256
        )

    print("Input:", condition.shape)
    print("Output:", condition_features.shape)
    print("Min:", condition_features.min().item())
    print("Max:", condition_features.max().item())