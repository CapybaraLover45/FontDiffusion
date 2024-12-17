import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import torch.nn.functional as F
from tqdm import tqdm


# ------------------ Residual Block ------------------
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.skip = nn.Conv2d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else nn.Identity()
        self.relu = nn.ReLU()

    def forward(self, x):
        identity = self.skip(x)
        x = self.relu(self.conv1(x))
        x = self.conv2(x)
        return self.relu(x + identity)

# ------------------ Self-Attention ------------------
class SelfAttention(nn.Module):
    def __init__(self, channels):
        super(SelfAttention, self).__init__()
        self.q = nn.Conv2d(channels, channels // 8, kernel_size=1)
        self.k = nn.Conv2d(channels, channels // 8, kernel_size=1)
        self.v = nn.Conv2d(channels, channels, kernel_size=1)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        b, c, h, w = x.size()
        q = self.q(x).view(b, -1, h * w).permute(0, 2, 1)
        k = self.k(x).view(b, -1, h * w)
        v = self.v(x).view(b, c, h * w)

        attn = self.softmax(torch.bmm(q, k))  # Attention map
        out = torch.bmm(v, attn.permute(0, 2, 1))
        return out.view(b, c, h, w) + x

# ------------------ Larger U-Net ------------------
class LargeUNet(nn.Module):
    def __init__(self, time_embed_dim=128):
        super(LargeUNet, self).__init__()

        # ---------------- Encoder ----------------
        # Change the first ResidualBlock to accept 257 channels
        self.enc1 = nn.Sequential(ResidualBlock(257, 64), ResidualBlock(64, 64))
        self.enc2 = nn.Sequential(ResidualBlock(64, 128), ResidualBlock(128, 128))
        self.enc3 = nn.Sequential(ResidualBlock(128, 256), ResidualBlock(256, 256))
        self.enc4 = nn.Sequential(ResidualBlock(256, 512), ResidualBlock(512, 512))

        # ---------------- Bottleneck ----------------
        self.bottleneck = nn.Sequential(
            ResidualBlock(512, 1024),
            SelfAttention(1024),
            ResidualBlock(1024, 1024)
        )

        # ---------------- Decoder ----------------
        self.dec4 = nn.Sequential(ResidualBlock(1024 + 512, 512), ResidualBlock(512, 512))
        self.dec3 = nn.Sequential(ResidualBlock(512 + 256, 256), ResidualBlock(256, 256))
        self.dec2 = nn.Sequential(ResidualBlock(256 + 128, 128), ResidualBlock(128, 128))
        self.dec1 = nn.Sequential(ResidualBlock(128 + 64, 64), ResidualBlock(64, 64))

        # ---------------- Output ----------------
        self.output = nn.Conv2d(64, 1, kernel_size=3, padding=1)

    def forward(self, x, t_embed):
        # ---------------- Encoder ----------------
        e1 = self.enc1(x)
        e2 = self.enc2(F.avg_pool2d(e1, 2))
        e3 = self.enc3(F.avg_pool2d(e2, 2))
        e4 = self.enc4(F.avg_pool2d(e3, 2))

        # ---------------- Bottleneck ----------------
        b = self.bottleneck(F.avg_pool2d(e4, 2))

        # ---------------- Decoder ----------------
        d4 = F.interpolate(b, size=e4.shape[2:], mode='bilinear', align_corners=False)
        d4 = self.dec4(torch.cat([d4, e4], dim=1))

        d3 = F.interpolate(d4, size=e3.shape[2:], mode='bilinear', align_corners=False)
        d3 = self.dec3(torch.cat([d3, e3], dim=1))

        d2 = F.interpolate(d3, size=e2.shape[2:], mode='bilinear', align_corners=False)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))

        d1 = F.interpolate(d2, size=e1.shape[2:], mode='bilinear', align_corners=False)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))

        # ---------------- Output ----------------
        return self.output(d1)



# ------------------ ScoreNet ------------------
class ScoreNet(nn.Module):
    def __init__(self):
        super(ScoreNet, self).__init__()

        # Time embedding with adaptive scaling
        self.time_embed = nn.Sequential(
            nn.Linear(1, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, 128)
        )
        
        self.time_scale = nn.Sequential(
            nn.Linear(128, 64), nn.Sigmoid(),
            nn.Linear(64, 1), nn.Sigmoid()
        )

        # Label embedding for MNIST (10 classes for digits 0-9)
        self.label_embed = nn.Embedding(10, 128)  # Adjust 10 to the number of unique labels

        # U-Net for score estimation
        self.unet = LargeUNet(time_embed_dim=128)

    def forward(self, x, t, labels):
        # Time embedding and scaling
        t_embed = self.time_embed(t.view(-1, 1))
        scale = self.time_scale(t_embed)
        t_embed = t_embed.view(t.size(0), 128, 1, 1)  # Reshape to (B, 128, 1, 1)
        t_embed = F.interpolate(t_embed, size=(x.size(2), x.size(3)), mode='bilinear', align_corners=False)

        # Scale the time embedding
        scaled_t_embed = scale.view(-1, 1, 1, 1) * t_embed

        # Process labels through embedding (MNIST labels)
        label_embed = self.label_embed(labels)  # Embed categorical labels
        label_embed = label_embed.view(label_embed.size(0), 128, 1, 1)  
        label_embed = F.interpolate(label_embed, size=(x.size(2), x.size(3)), mode='bilinear', align_corners=False)

        # Input image x has channels, label_embed has 128 channels, and scaled_t_embed has 128 channels
        x = torch.cat([x, label_embed, scaled_t_embed], dim=1)

        return self.unet(x, t_embed)




# Gaussian noise schedule. map t in [0,1] -> [beta_start, beta_end]
def noise_schedule(t, beta_min=0.1, beta_max=20.0):
    return 0.5*(t**2)*(beta_max-beta_min) + t*beta_min

def add_noise(x, t, device):
    noise = torch.randn_like(x).to(device)
    B_t = noise_schedule(t).to(device)
    x_0_coeff = torch.exp(-0.5*B_t).view(-1, 1, 1, 1)
    sigma = torch.sqrt(1 - torch.exp(-B_t))
    sigma = sigma.view(-1, 1, 1, 1)
    noisy_x = x_0_coeff * x + sigma * noise
    return noisy_x, noise, sigma

def dsm_loss(score_net, x, t, labels, device):
    # Add noise to the input images based on the time step t
    noisy_x, noise, sigma = add_noise(x, t, device)

    # Calculate the target score based on the noise
    target_score = -noise / sigma  # Conditional score

    # Pass both noisy images and labels through the model
    predicted_score = score_net(noisy_x, t, labels)  # Model is conditioned on labels

    # Compute the loss between the predicted score and the target score
    loss = torch.mean(0.5 * (predicted_score - target_score)**2)
    return loss



# Training loop
def train(score_net, dataloader, optimizer, device, num_epochs=200, patience=10):
    score_net.train() 
    best_loss = float('inf')  # Initialize the best loss to infinity
    epochs_without_improvement = 0  
    for epoch in range(num_epochs):
        total_loss = 0.0
        loop = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")

        for x, labels in loop:
            x, labels = x.to(device), labels.to(device)  # Move inputs and labels to device
            epsilon = 3e-2
            t = torch.rand(x.size(0), device=device) * (1 - epsilon) + epsilon  # time domain
            
            # Pass images and labels through the network
            optimizer.zero_grad()
            loss = dsm_loss(score_net, x, t, labels, device)  # Use the labels for conditioning
            loss.backward()
            optimizer.step()

            total_loss += loss.item()  # Accumulate loss

            # Update progress bar with the current loss
            loop.set_postfix(loss=loss.item())

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {avg_loss:.4f}")

        # Check if the loss has improved
        if avg_loss < best_loss:
            best_loss = avg_loss
            epochs_without_improvement = 0 
        else:
            epochs_without_improvement += 1

        # If the loss hasn't improved for 'patience' epochs, stop training
        if epochs_without_improvement >= patience:
            print(f"Early stopping triggered at epoch {epoch + 1}")
            break


# Main script
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    dataset = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
    dataloader = DataLoader(dataset, batch_size=128, shuffle=True)

    # Initialize model, optimizer
    score_net = ScoreNet().to(device)
    optimizer = optim.Adam(score_net.parameters(), lr=1e-3, weight_decay=1e-5)

    # Train the model
    train(score_net, dataloader, optimizer, device)

    # Save the model after training
    torch.save(score_net.state_dict(), "scorenet.pth")
    print("Model saved to scorenet.pth")




if __name__ == "__main__":
    main()
