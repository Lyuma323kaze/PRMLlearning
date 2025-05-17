import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F

# Set random seed for reproducibility
torch.manual_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# VAE Model
class VAE(nn.Module):
    def __init__(self, latent_dim=2):
        super(VAE, self).__init__()
        self.latent_dim = latent_dim

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=4, stride=2, padding=1),  # [32, 14, 14]
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),  # [64, 7, 7]
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 256),
            nn.ReLU(),
        )
        self.fc_mu = nn.Linear(256, latent_dim)
        self.fc_logvar = nn.Linear(256, latent_dim)

        # Decoder
        self.decoder_input = nn.Linear(latent_dim, 256)
        self.decoder = nn.Sequential(
            nn.Linear(256, 64 * 7 * 7),
            nn.ReLU(),
            nn.Unflatten(1, (64, 7, 7)),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),  # [32, 14, 14]
            nn.ReLU(),
            nn.ConvTranspose2d(32, 1, kernel_size=4, stride=2, padding=1),  # [1, 28, 28]
            nn.Sigmoid()
        )

    def encode(self, x):
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h = self.decoder_input(z)
        h = F.relu(h)
        return self.decoder(h)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decode(z)
        return recon_x, mu, logvar

# Loss function
def vae_loss(recon_x, x, mu, logvar):
    BCE = F.binary_cross_entropy(recon_x, x, reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD

# Data loading
transform = transforms.Compose([transforms.ToTensor()])
train_dataset = torchvision.datasets.MNIST(root='./data', train=True, transform=transform, download=True)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle=True)

# Initialize model, optimizer
latent_dim = 2
model = VAE(latent_dim=latent_dim).to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# Training
num_epochs = 15
train_losses = []
model.train()
for epoch in range(num_epochs):
    epoch_loss = 0
    for batch_idx, (data, _) in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(data)
        loss = vae_loss(recon_batch, data, mu, logvar)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    avg_loss = epoch_loss / len(train_loader.dataset)
    train_losses.append(avg_loss)
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}')

# Plot loss curve
plt.figure(figsize=(8, 6))
plt.plot(train_losses, label='Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('VAE Training Loss Curve')
plt.legend()
plt.savefig('vae_loss_curve.png')
plt.close()

# Generate and visualize reconstructed images
model.eval()

# Interpolation between digits
def interpolate_digits(model, digit1, digit2, num_samples=10, num_steps=10):
    model.eval()
    with torch.no_grad():
        # Collect multiple samples for each digit
        digit1_samples = []
        digit2_samples = []
        for data, labels in train_loader:
            data = data.to(device)
            idx1 = (labels == digit1).nonzero(as_tuple=True)[0]
            idx2 = (labels == digit2).nonzero(as_tuple=True)[0]
            if len(idx1) > 0:
                digit1_samples.extend(data[idx1[:min(len(idx1), num_samples)]])
            if len(idx2) > 0:
                digit2_samples.extend(data[idx2[:min(len(idx2), num_samples)]])
            if len(digit1_samples) >= num_samples and len(digit2_samples) >= num_samples:
                break
        if len(digit1_samples) < num_samples or len(digit2_samples) < num_samples:
            raise ValueError(f"Insufficient samples found for digits {digit1} or {digit2}")

        # Encode multiple samples and compute mean latent representation
        mu1_list, logvar1_list = [], []
        mu2_list, logvar2_list = [], []
        for img in digit1_samples[:num_samples]:
            mu, logvar = model.encode(img.unsqueeze(0))
            mu1_list.append(mu)
            logvar1_list.append(logvar)
        for img in digit2_samples[:num_samples]:
            mu, logvar = model.encode(img.unsqueeze(0))
            mu2_list.append(mu)
            logvar2_list.append(logvar)

        mu1 = torch.mean(torch.stack(mu1_list), dim=0)
        logvar1 = torch.mean(torch.stack(logvar1_list), dim=0)
        mu2 = torch.mean(torch.stack(mu2_list), dim=0)
        logvar2 = torch.mean(torch.stack(logvar2_list), dim=0)

        # Reparameterize to get stable latent points
        z1 = model.reparameterize(mu1, logvar1)
        z2 = model.reparameterize(mu2, logvar2)

        # Interpolate
        interpolated_images = []
        for alpha in np.linspace(0, 1, num_steps):
            z = (1 - alpha) * z1 + alpha * z2
            recon = model.decode(z)
            # Ensure image is in valid range and shape
            recon = torch.clamp(recon, 0, 1).cpu().squeeze()
            interpolated_images.append(recon)

        # Plot with improved visualization
        plt.figure(figsize=(num_steps * 1.5, 2))
        for i, img in enumerate(interpolated_images):
            plt.subplot(1, num_steps, i+1)
            plt.imshow(img, cmap='gray', vmin=0, vmax=1)
            plt.axis('off')
            if i == 0:
                plt.title(f'{digit1} to {digit2}')
        plt.tight_layout()
        plt.savefig(f'interpolation_{digit1}_to_{digit2}.png')
        plt.close()

        return z1.cpu().numpy(), z2.cpu().numpy()

# Perform interpolation
z1_7, z7 = interpolate_digits(model, 1, 7)
z1_4, z4 = interpolate_digits(model, 1, 4)

# Latent space visualization
def plot_reconstruction_grid(model, grid_size=30, z_range=5.0):
    model.eval()
    with torch.no_grad():
        # Create a grid of latent space coordinates
        z1 = np.linspace(-z_range, z_range, grid_size)
        z2 = np.linspace(-z_range, z_range, grid_size)
        z1_grid, z2_grid = np.meshgrid(z1, z2)
        z_grid = np.stack([z1_grid.flatten(), z2_grid.flatten()], axis=1)
        z_tensor = torch.FloatTensor(z_grid).to(device)

        # Decode each point in the grid
        recon_images = model.decode(z_tensor).cpu().numpy()
        recon_images = recon_images.reshape(grid_size, grid_size, 28, 28)

        # Plot the grid
        fig, axes = plt.subplots(grid_size, grid_size, figsize=(10, 10))
        for i in range(grid_size):
            for j in range(grid_size):
                axes[i, j].imshow(recon_images[i, j], cmap='gray')
                axes[i, j].axis('off')
        # Set labels for axes
        for i in range(grid_size):
            axes[i, 0].set_ylabel(f'{z2[i]:.1f}', rotation=0, labelpad=15)
            axes[grid_size-1, i].set_xlabel(f'{z1[i]:.1f}')
        plt.suptitle('Reconstructed Images in Latent Space')
        plt.savefig('latent_space_reconstruction_grid.png')
        plt.close()

# Plot the reconstruction grid
plot_reconstruction_grid(model)
