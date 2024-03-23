import torch
from torch import nn 
import numpy as np
import torch.nn.functional as F

from models.base import BaseGenerativeModel
from models.types_ import List, Dict, Tensor


class betaVAE(BaseGenerativeModel):
    """
    Simple beta-VAE implementation with convolutional layers.
    Assume usage on MNIST dataset where inputs are 1*28*28 binary images.
    """

    def __init__(self, 
            latent_dim: int = 32,
            beta: float = 1.0,
        ):
        super(betaVAE, self).__init__()
        self.latent_dim = latent_dim
        self.beta = beta
        
        # Input: [B, 1, 28, 28]
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=4, 
                      kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(4), nn.LeakyReLU(inplace=True),  # [B, 4, 14, 14]
            nn.Conv2d(in_channels=4, out_channels=16, 
                      kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(16), nn.LeakyReLU(inplace=True), # [B, 16, 7, 7]
            nn.Flatten(), # [B, 16*7*7]
        )

        self.mu_layer = nn.Sequential(
            nn.Linear(784, 256), nn.LeakyReLU(inplace=True),
            nn.Linear(256, latent_dim), nn.LeakyReLU(inplace=True), # [B, latent_dim]
        )
        self.logvar_layer = nn.Sequential(
            nn.Linear(784, 256), nn.LeakyReLU(inplace=True),
            nn.Linear(256, latent_dim), nn.LeakyReLU(inplace=True), # [B, latent_dim]
        )

        self.decoder_input = nn.Sequential(
            nn.Linear(latent_dim, 256), nn.LeakyReLU(inplace=True),
            nn.Linear(256, 784), nn.LeakyReLU(inplace=True),
            nn.Unflatten(1, (16, 7, 7)),
        )
        self.decoder = nn.Sequential(
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
    
    def encode(self, x: Tensor) -> List[Tensor]:
        """
        Return latent mu and logvar from the input image,
        each with size of [B, latent_dim]
        """
        x = self.encoder(x)

        mu = self.mu_layer(x)
        logvar = self.logvar_layer(x)

        return [mu, logvar]
    
    def reparameterization(self, mu: Tensor, logvar: Tensor) -> Tensor:
        """
        Sample latent variable from mu and logvar with reparameterization trick
        """
        std = torch.exp(0.5 * logvar)

        eps = torch.randn_like(std)
        z = mu + eps * std 

        return z
    
    def decode(self, z: Tensor) -> Tensor:
        """
        Decode to reconstruct the original image from latent variable z
        """
        x_ = self.decoder_input(z)
        x_ = self.decoder(x_)

        return x_
    
    def forward(self, x: Tensor) -> List[Tensor]:
        #[B, C, H, W]
        mu, logvar = self.encode(x)
        z = self.reparameterization(mu, logvar)
        x_ = self.decode(z)

        return [x_, mu, logvar]
    
    def loss_function(self, 
            input: Tensor, target: Tensor, 
            mu: Tensor, logvar: Tensor
        ) -> Dict[str, Tensor]:
        ### "mean" reduction does not work at all, need further investigation ???
        
        ### binary loss gives better results than mse loss
        reconstruction_loss = F.binary_cross_entropy(input, target, reduction="sum")
        
        # KLD Loss Derivation, reduction with "mean" to align with mse_loss
        # Reference: https://zhuanlan.zhihu.com/p/34998569
        kld_loss = -0.5 * torch.sum(1 + logvar - mu ** 2 - logvar.exp())
        
        # kld_weight are introduced as beta_norm from the original paper
        # kld_weight = (dimension of z) / (dimension of input x)
        # Reference: https://openreview.net/pdf?id=Sy2fzU9gl
        
        # dim_z = self.latent_dim
        # dim_x = np.prod(input.size()[1:])
        loss = reconstruction_loss + self.beta * kld_loss

        return {
            "loss": loss,
            "reconstruction": reconstruction_loss,
            "kld_loss": kld_loss
        }
    
    def sample(self,
            num_samples: int, current_device
        ) -> Tensor:
        """
        Samples from the latent space and return the corresponding image space map.
        """
        z = torch.randn(num_samples, self.latent_dim)
        z = z.to(current_device)

        samples = self.decode(z)
        return samples
