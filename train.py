import torch
from diffusers import DDPMScheduler
from lurssen.models.unet import ConditionalUNet
import torch.nn.functional as F
from lurssen.encoder import RoomGenEncoder
from lurssen.dataset import get_dataloader

def train_diffusion_epoch(
    model,
    encoder,
    train_loader,
    noise_scheduler,
    optimizer,
    device
):

    model.train()
    encoder.train()

    total_loss = 0.0

    for floorplan, outershell_mask, node_features, edge_features in train_loader:

        floorplan = floorplan.to(device)
        outershell_mask = outershell_mask.to(device)

        node_features = [
            x.to(device) for x in node_features
        ]

        edge_features = [
            x.to(device) for x in edge_features
        ]
        # Add batch dimension if necessary
        if floorplan.ndim == 3:
            floorplan = floorplan.unsqueeze(0)

        if outershell_mask.ndim == 3:
            outershell_mask = outershell_mask.unsqueeze(0)

        # Generate condition
        with torch.no_grad():

            condition = encoder(
                outershell_mask,
                node_features[0],
                edge_features[0]
            )

        # Sample random noise
        noise = torch.randn_like(floorplan)

        # Sample random timesteps
        timesteps = torch.randint(
            0,
            noise_scheduler.config.num_train_timesteps,
            (floorplan.shape[0],),
            device=device
        ).long()

        # Add noise to clean images
        noisy_images = noise_scheduler.add_noise(
            floorplan,
            noise,
            timesteps
        )
        # Predict noise
        noise_pred = model(
            noisy_images,
            timesteps,
            condition
        )
        # Calculate loss
        loss = F.mse_loss(
            noise_pred,
            noise
        )
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(encoder.parameters()) +
            list(model.parameters()),
            max_norm=1.0
        )
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(train_loader)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ConditionalUNet(
    condition_channels=64,
    image_channels=3
).to(device)
encoder = RoomGenEncoder().to(device)
noise_scheduler = DDPMScheduler(
    num_train_timesteps=1000,
    beta_schedule="squaredcos_cap_v2",
    prediction_type="epsilon"
)
optimizer = torch.optim.AdamW(
    list(encoder.parameters()) +
    list(model.parameters()),
    lr=1e-4,
    weight_decay=1e-4
)

train_loader = get_dataloader(num_samples=1, batch_size=1, seed=41)


num_epochs = 100
for epoch in range(num_epochs):
    loss = train_diffusion_epoch(
        model=model,
        encoder=encoder,
        train_loader=train_loader,
        noise_scheduler=noise_scheduler,
        optimizer=optimizer,
        device=device
    )
    print(
        f"Epoch {epoch + 1}/{num_epochs} | "
        f"Loss: {loss:.6f}"
    )