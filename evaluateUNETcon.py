import torch
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import trainUNETcon as train


def b_t(t, beta_min=0.1, beta_max=20.0):
    return t * (beta_max - beta_min) + beta_min


def sample(score_net, x_T, timesteps, labels, device, snr=0.1, corrector_steps=1):
    x = x_T.to(device)  # Start with pure noise at t = T
    dt = 1 / timesteps
    time_steps = torch.linspace(1.0,3e-2, steps=timesteps).to(device)  # Time discretization
    
    for t in time_steps:
        t_tensor = torch.full((x.size(0),), t, device=device)  # Time tensor for batch
        beta_t = b_t(t)  # Noise schedule at time t
        
        # ---------------- Predictor Step ----------------
        # Score prediction (with conditioning on labels)
        score = score_net(x, t_tensor, labels)  # Pass labels along with x and t_tensor
        
        # Generate Gaussian noise
        noise = torch.randn_like(x).to(device)
        
        # Euler-Maruyama update rule (reverse SDE)
        dt_tensor = torch.tensor(dt, dtype=x.dtype, device=x.device)
        x = x + (0.5 * beta_t * x + beta_t * score) * dt_tensor
        x = x + torch.sqrt(beta_t * dt_tensor) * noise
        
        # ---------------- Corrector Step ----------------
        for _ in range(corrector_steps):
            # Score prediction for Langevin dynamics (with conditioning on labels)
            score = score_net(x, t_tensor, labels)
            
            # Noise magnitude using SNR
            noise = torch.randn_like(x).to(device)
            grad_norm = torch.norm(score.view(score.size(0), -1), dim=1).mean()
            noise_norm = torch.norm(noise.view(noise.size(0), -1), dim=1).mean()
            step_size = (snr * noise_norm / grad_norm) ** 2 * 2 * beta_t
            
            # Langevin dynamics update
            x = x + step_size * score + torch.sqrt(2 * step_size) * noise
    
    return x


def evaluate():
    # Load device and model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    score_net = train.ScoreNet().to(device)
    score_net.load_state_dict(torch.load("scorenet.pth", map_location=device, weights_only=True))
    score_net.eval()
    print("Model loaded from scorenet.pth")
    
    # Generate a batch of labels (e.g., 0 to 9 for MNIST)
    batch_size = 16
    specific_digit = 9  # The specific digit you want to generate
    labels = torch.full((batch_size,), specific_digit).to(device)  # All labels set to '5'

    # Start with pure noise at t = T
    x_T = torch.randn(batch_size, 1, 28, 28).to(device)
    
    timesteps = 2000
    # Generate samples
    with torch.no_grad():  # No gradients needed for sampling
        sampled_images = sample(score_net, x_T, timesteps, labels, device=device, snr=0.1, corrector_steps=1)

    # Plot the samples
    for i in range(batch_size):
        plt.subplot(4, 4, i + 1)
        plt.imshow(sampled_images[i].cpu().squeeze(), cmap="gray")
        plt.axis("off")
    plt.show()




if __name__ == "__main__":
    evaluate()


