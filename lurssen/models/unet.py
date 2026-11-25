from diffusers import UNet2DModel
import torch
import torch.nn as nn

class ConditionalUNet(nn.Module):

    def __init__(
        self,
        condition_channels=64,
        image_channels=3
    ):
        super().__init__()

        self.unet = UNet2DModel(
            sample_size=256,

            in_channels=image_channels + condition_channels,
            out_channels=image_channels,

            layers_per_block=2,

            block_out_channels=(
                64,
                128,
                256,
                256
            ),

            down_block_types=(
                "DownBlock2D",
                "DownBlock2D",
                "AttnDownBlock2D",
                "DownBlock2D"
            ),

            up_block_types=(
                "UpBlock2D",
                "AttnUpBlock2D",
                "UpBlock2D",
                "UpBlock2D"
            ),

            norm_num_groups=8
        )

    def forward(self, noisy_images, timesteps, condition):

        x = torch.cat(
            [noisy_images, condition],
            dim=1
        )

        noise_pred = self.unet(
            x,
            timesteps
        ).sample

        return noise_pred

if __name__ == "__main__":
    import torch
    import matplotlib.pyplot as plt

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    # Initialize model
    model = ConditionalUNet(
        condition_channels=64,
        image_channels=3
    ).to(device)

    model.eval()

    # Define batch size
    batch_size = 1

    # Simulate noisy floor plans
    noisy_images = torch.randn(
        batch_size,
        3,
        256,
        256
    ).to(device)

    # Simulate encoder output
    condition = torch.randn(
        batch_size,
        64,
        256,
        256
    ).to(device)

    # Random diffusion timesteps
    timesteps = torch.randint(
        0,
        1000,
        (batch_size,),
        device=device
    ).long()

    # Forward pass
    with torch.no_grad():

        noise_pred = model(
            noisy_images,
            timesteps,
            condition
        )

    # Inspect results
    print("Noisy images:", noisy_images.shape)
    print("Condition:", condition.shape)
    print("Timesteps:", timesteps)
    print("Predicted noise:", noise_pred.shape)

    print("Min:", noise_pred.min().item())
    print("Max:", noise_pred.max().item())