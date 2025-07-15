import torch
import numpy as np
import torch.nn.functional as F

from typing import Tuple
from torch import nn, Tensor
from omegaconf import DictConfig

from einops import rearrange, repeat

from future_motion.models.modules.e2e_diffusion import ConditionalUNet, Diffusion, LearnableGaussianPrior2D

class UNetTraj(nn.Module):
    def __init__(
        self,
        num_points: int,
        emb_dim: int, 
        **kwargs,
    ) -> None:
        super().__init__()

        self.gaussian_prior = LearnableGaussianPrior2D(num_points)

        self.route_emb_layer = nn.Embedding(
                3,
                emb_dim
            )
        
        self.unet_conditional = ConditionalUNet(
            in_dim = 2,
            out_dim = 2,
            time_dim = emb_dim
        )
        

    def forward(
        self,
        prior: Tensor,
        routing: Tensor,
        t: Tensor,
        **kwargs,
    ):
        """
        Args:

        Returns:
        """
        routing_emb = self.route_emb_layer(routing)

        out = self.unet_conditional(
            noisy_traj = prior,
            t = t,
            prompt = routing_emb
        )

        return out


if __name__ == '__main__':


    batch_size = 1
    device = "cuda:0"

    diffusion = Diffusion(
        noise_steps=10,
        beta_start=1e-4,
        beta_end=0.02,
        device = device
        )
    
    model = UNetTraj(
        num_points=20,
        emb_dim=256,
    )

    model.to(device)

    traj = torch.rand(batch_size, 20, 1, 2, dtype=torch.float32).to(device)

    noise_seed = model.gaussian_prior.sample(batch_size, device)

    t = diffusion.sample_timesteps(batch_size).to(device)

    routing = torch.randint(
            low=1, 
            high=3, 
            size=(batch_size,)
            ).to(device)
    
    noisy_traj, noise = diffusion.noise_trajectory(noise_seed, traj, t)

    predicted_noise = model(
        noisy_traj.permute(0,3,1,2),
        routing,
        t
    ).permute(0,2,3,1)

    print(f"Trajectory:         {traj.shape}")
    print(f"Noisy Trajectory:   {noisy_traj.shape}")
    print(f"Noise:              {noise.shape}")
    print(f"Routing:            {routing.shape}")
    print(f"Denoising Timestep: {routing.shape}")
    print(f"Predicted Noise:    {predicted_noise.shape}")

    # Trajectory:         torch.Size([1, 20, 1, 2])
    # Noisy Trajectory:   torch.Size([1, 20, 1, 2])
    # Noise:              torch.Size([1, 20, 1, 2])
    # Routing:            torch.Size([1])
    # Denoising Timestep: torch.Size([1])
    # Predicted Noise:    torch.Size([1, 20, 1, 2])