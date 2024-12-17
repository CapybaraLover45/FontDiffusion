
import torch
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import Train_Unconditional_Model as train


def b_t(t, beta_min=0.1, beta_max=20.0):
    return  (beta_max-beta_min)


def sample(score_net, x_T, timesteps, device):
    x = x_T.to(device)  # Start with pure noise at t = T
    dt= 1/timesteps
    time_steps = torch.linspace(1.0, 1e-3, steps=timesteps).to(device)  # Time discretization
    for t in time_steps:
        t_tensor = torch.full((x.size(0),), t, device=device)  # Time tensor for batch
        beta_t = b_t(t)  # Noise schedule at time t
        
        # Score prediction
        score = score_net(x, t_tensor)
        
        # Generate Gaussian noise
        noise = torch.randn_like(x).to(device)

        # Euler-Maruyama update rule
        dt_tensor = torch.tensor(dt, dtype=x.dtype, device=x.device)
        x = x + (0.5 * beta_t * x + beta_t * score) * dt_tensor  # Subtract noise guided by score
        x = x + torch.sqrt(beta_t * dt_tensor) * noise

    return x


def evaluate():
    # Load device and model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    score_net = train.ScoreNet().to(device)
    score_net.load_state_dict(torch.load("scorenet.pth", map_location=device, weights_only=True))
    score_net.eval()
    print("Model loaded from scorenet.pth")
    x_T = torch.randn(16, 1, 28, 28)  # Batch of 1 noise samples (e.g., MNIST size 28x28)
    

    timesteps = 2000
    # Generate samples
    with torch.no_grad(): 
        sampled_images = sample(score_net, x_T, timesteps, device=device)


    # Plot the single sample
    for i in range(16):
        plt.subplot(4, 4, i + 1)
        plt.imshow(sampled_images[i].cpu().squeeze(), cmap="gray")
        plt.axis("off")
    plt.show()


if __name__ == "__main__":
    evaluate()

