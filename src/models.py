import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function

# --- HIGHER CAPACITY ENCODER WITH RESIDUAL CONNECTIONS ---
class SiameseEncoder(nn.Module):
    def __init__(self, input_dim=128, latent_dim=64):
        super(SiameseEncoder, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=9, stride=8, padding=4),  # Doubled channels
            nn.BatchNorm1d(64), nn.ReLU(), nn.Dropout(0.1)
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=3, stride=1, padding=4, dilation=4),  # Doubled
            nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.1)
        )
        self.conv3 = nn.Sequential(
            nn.Conv1d(128, 128, kernel_size=3, stride=1, padding=1),  # Doubled
            nn.BatchNorm1d(128), nn.ReLU()
        )
        # Residual projection for conv2->conv3
        self.res_proj = nn.Conv1d(64, 128, kernel_size=1)
        
        self.flatten_dim = 128 * 16
        self.fc = nn.Sequential(
            nn.Linear(self.flatten_dim, 512),  # Doubled
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 128),  # Doubled latent
            nn.ReLU(),
            nn.Linear(128, latent_dim),
            nn.BatchNorm1d(latent_dim)
        )

    def forward(self, x):
        x = x.unsqueeze(1)
        f1 = self.conv1(x)
        f2 = self.conv2(f1)
        f3 = self.conv3(f2) + self.res_proj(f1)  # Residual connection
        flat = f3.view(f3.size(0), -1) 
        z = self.fc(flat)
        return z

class TaskClassifier(nn.Module):
    def __init__(self, latent_dim=64, num_classes=6):
        super(TaskClassifier, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(32, num_classes)
        )
    def forward(self, z):
        return self.net(z)

class PhysicsHead(nn.Module):
    def __init__(self, latent_dim=64):
        super(PhysicsHead, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 32), nn.ReLU(),
            nn.Linear(32, 2) 
        )
        self.w = nn.Parameter(torch.tensor([0.5])) 
        self.b = nn.Parameter(torch.tensor([0.0]))

    def forward(self, z):
        sensitivity_mag = torch.norm(z, dim=1, keepdim=True)
        out = self.net(z)
        baseline_est = out[:, 1].unsqueeze(1)
        return sensitivity_mag, baseline_est

# --- NEW: DYNAMIC INCREMENTAL DISCRIMINATOR ---
class GradientReversalFn(Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.alpha, None

class DynamicDomainDiscriminator(nn.Module):
    def __init__(self, latent_dim=64, initial_domains=2):
        super(DynamicDomainDiscriminator, self).__init__()
        self.latent_dim = latent_dim
        self.num_domains = initial_domains
        
        # Shared Layers
        self.shared = nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # Output Layer (Will grow)
        self.output_layer = nn.Linear(32, initial_domains)

    def forward(self, z, alpha=1.0):
        z_rev = GradientReversalFn.apply(z, alpha)
        feat = self.shared(z_rev)
        return self.output_layer(feat)

    def add_new_domain(self):
        """
        Dynamically adds a neuron to the output layer for the new batch.
        Preserves existing weights.
        """
        old_weights = self.output_layer.weight.data
        old_bias = self.output_layer.bias.data
        
        self.num_domains += 1
        
        # Create new layer
        new_layer = nn.Linear(32, self.num_domains)
        
        # Copy old weights
        new_layer.weight.data[:self.num_domains-1] = old_weights
        new_layer.bias.data[:self.num_domains-1] = old_bias
        
        # Initialize new weights (Random small noise)
        nn.init.normal_(new_layer.weight.data[self.num_domains-1], std=0.01)
        new_layer.bias.data[self.num_domains-1] = 0.0
        
        self.output_layer = new_layer.to(self.output_layer.weight.device)
        print(f"  [Model] Domain Discriminator expanded to {self.num_domains} classes.")