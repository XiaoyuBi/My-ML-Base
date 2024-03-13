import torch
from torch import nn 
import torch.nn.functional as F

from models.base import BaseGenerativeModel
from models.types_ import Dict, Tensor


class Autoencoder(BaseGenerativeModel):
    """
    Simple autoencoder implementation with convolutional layers.
    Assume usage on MNIST dataset where inputs are 28*28 binary images.
    """
    
    def __init__(self, 
            latent_dim: int = 64
        ):
        super(Autoencoder, self).__init__()
        self.latent_dim = latent_dim
        
        # Input: [B, 1, 28, 28]
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=4, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(4), nn.LeakyReLU(inplace=True),  # [B, 4, 14, 14]
            nn.Conv2d(in_channels=4, out_channels=16, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(16), nn.LeakyReLU(inplace=True), # [B, 16, 7, 7]
            nn.Flatten(), # [B, 16*7*7]
            nn.Linear(784, 256), nn.LeakyReLU(inplace=True),
            nn.Linear(256, latent_dim), nn.LeakyReLU(inplace=True), # [B, latent_dim]
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 256), nn.LeakyReLU(inplace=True),
            nn.Linear(256, 784), nn.LeakyReLU(inplace=True),
            nn.Unflatten(1, (16, 7, 7)),
            nn.ConvTranspose2d(
                in_channels=16, out_channels=4, 
                kernel_size=3, stride=2, 
                padding=1, output_padding=1),
            nn.BatchNorm2d(4), nn.LeakyReLU(inplace=True),
            nn.ConvTranspose2d(
                in_channels=4, out_channels=1, 
                kernel_size=3, stride=2, 
                padding=1, output_padding=1),
            nn.BatchNorm2d(1), nn.Sigmoid(),
        )
    
    def forward(self, x: Tensor) -> Tensor:
        # [B, H, W] -> [B, C, H, W]
        x = x.unsqueeze(1)

        z = self.encoder(x) # [B, latent_dim]
        x_ = self.decoder(z) # [B, C, H, W]
        x_ = x_.squeeze(1) # [B, H, W]

        return x_
    
    def loss_function(self, 
            input: Tensor, target: Tensor, loss_type: str = 'mse'
        ) -> Dict[str, Tensor]:
        
        if loss_type == 'mse':
            return {'loss': F.mse_loss(input, target)}
        elif loss_type == 'binary':
            return {'loss': F.binary_cross_entropy(input, target)}
        else:
            raise ValueError(f"Loss Type {loss_type} not supported!")
    
    def sample(self, 
            num_samples: int, device
        ) -> Tensor:
        
        z = torch.randn(size = (num_samples, self.latent_dim))
        z = z.to(device)
        samples = self.decoder(z)
        
        return samples
    