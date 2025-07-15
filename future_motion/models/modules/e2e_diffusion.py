import torch
import torch.nn as nn
import math
import numpy as np
import torch.nn.functional as F

class LearnableGaussianPrior2D(nn.Module):
    def __init__(self, num_points: int):
        super().__init__()
        self.num_points = num_points
        
        self.mu_x = nn.Parameter(torch.zeros(num_points))
        self.logvar_x = nn.Parameter(torch.zeros(num_points))
        
        self.mu_y = nn.Parameter(torch.zeros(num_points))
        self.logvar_y = nn.Parameter(torch.zeros(num_points))

    def sample(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """
        Used in generation mode

        Args:
            batch_size: int
            device: torch.device

        Returns:
            sample: [batch_size, 20, 1, 2]
        """
        
        eps_x = torch.randn(batch_size, self.num_points, device=device)
        eps_y = torch.randn(batch_size, self.num_points, device=device)
        
        x = self.mu_x.unsqueeze(0) + torch.exp(0.5 * self.logvar_x).unsqueeze(0) * eps_x
        y = self.mu_y.unsqueeze(0) + torch.exp(0.5 * self.logvar_y).unsqueeze(0) * eps_y
        
        return torch.stack([x, y], dim=-1).unsqueeze(2)

    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        """
        Calculates log likelihood of x, and sums it for all points

        Args:
            x: [batch_size, 20, 1, 2]

        Returns:
            log_prob: [batch_size, ]
        """
        B, N, _, D = x.shape
        assert N == self.num_points and D == 2

        x_coords = x[..., 0]
        y_coords = x[..., 1]

        
        var_x = torch.exp(self.logvar_x).unsqueeze(0).unsqueeze(2)
        var_y = torch.exp(self.logvar_y).unsqueeze(0).unsqueeze(2)

        
        logp_x = -0.5 * (
            ((x_coords - self.mu_x.unsqueeze(0).unsqueeze(2)) ** 2) / var_x
            + self.logvar_x.unsqueeze(0).unsqueeze(2)
            + math.log(2 * math.pi)
        )  

        logp_y = -0.5 * (
            ((y_coords - self.mu_y.unsqueeze(0).unsqueeze(2)) ** 2) / var_y
            + self.logvar_y.unsqueeze(0).unsqueeze(2)
            + math.log(2 * math.pi)
        ) 

        logp = (logp_x + logp_y).sum(dim=1)

        return logp

class SelfAttention(nn.Module):
    def __init__(
            self, 
            channels: int,
            num_heads: int = 4
            ):
        super(SelfAttention, self).__init__()
        self.channels = channels
        self.mha = nn.MultiheadAttention(channels, num_heads, batch_first=True)
        self.ln = nn.LayerNorm([channels])
        self.ff_self = nn.Sequential(
            nn.LayerNorm([channels]),
            nn.Linear(channels, channels),
            nn.GELU(),
            nn.Linear(channels, channels),
        )

    def forward(self, x):
        # x: [batch_size, channels, 1, w]
        w = x.shape[-2]
        x = x.view(-1, self.channels, w).swapaxes(1, 2)
        x_ln = self.ln(x)
        attention_value, _ = self.mha(x_ln, x_ln, x_ln)
        attention_value = attention_value + x
        attention_value = self.ff_self(attention_value) + attention_value
        return attention_value.swapaxes(2, 1).view(-1, self.channels, w, 1)

class DoubleConv(nn.Module):
    def __init__(
            self, 
            in_dim: int, 
            out_dim: int,
            kernel_size: int = 3,
            mid_dim: int = None, 
            residual: bool = False
            ):
        super(DoubleConv, self).__init__()
        self.residual = residual
        self.in_dim = in_dim
        self.out_dim = out_dim

        if not mid_dim:
            mid_dim = out_dim

        self.double_conv = nn.Sequential(
            nn.Conv2d(
                in_dim, 
                mid_dim, 
                kernel_size = (kernel_size, 1), 
                padding = (kernel_size//2, 0), 
                bias = False
                ),
            nn.GroupNorm(1, mid_dim),
            nn.GELU(),
            nn.Conv2d(
                mid_dim, 
                out_dim, 
                kernel_size = (kernel_size, 1), 
                padding = (kernel_size//2, 0),
                bias = False
                ),
            nn.GroupNorm(1, out_dim)
        )

    def forward(self, x):
        """
        Args:
            x: [batch_size, 2, 20, 1]
        Returns:
            out: [batch_size, out_dim, 20, 1]
        """

        if self.residual:
            assert self.in_dim == self.out_dim, "Residual needs in_dim == out_dim"
            return F.gelu(x + self.double_conv(x))
        else:
            return self.double_conv(x)
        
class Down(nn.Module):
    def __init__ (
            self, 
            in_dim: int,
            out_dim: int,
            emb_dim: int = 256,
            down_kernel = (2, 1),
            down_stride = (2, 1)
            ):
        super(Down, self).__init__()
        
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(kernel_size = down_kernel, stride = down_stride),
            DoubleConv(in_dim, in_dim, residual = True),
            DoubleConv(in_dim, out_dim, residual = False)
        )

        self.emb_layer = nn.Sequential(
            nn.SiLU(),
            nn.Linear(
                emb_dim,
                out_dim
            ),
        )

    def forward(self, x, t):

        x = self.maxpool_conv(x)
        emb = self.emb_layer(t)[:, :, None, None].repeat(1, 1, x.shape[-2], x.shape[-1])

        return x + emb
    
class Up(nn.Module):
    def __init__ (
            self, 
            in_dim: int,
            out_dim: int,
            emb_dim: int = 256,
            kernel_size: int = None
            ):
        
        super(Up, self).__init__()

        if not kernel_size:
            kernel_size = in_dim//2

        self.conv = nn.Sequential(
            DoubleConv(in_dim, in_dim, residual=True),
            DoubleConv(in_dim, out_dim, kernel_size=kernel_size, residual=False)
        )

        self.emb_layer = nn.Sequential(
            nn.SiLU(),
            nn.Linear(
                emb_dim,
                out_dim
            ),
        )

    def forward(self, x, skip_x, t):
        x = F.interpolate(x,
                          size=skip_x.shape[2:],
                          mode="bilinear",
                          align_corners=False
                          )
        x = torch.cat([skip_x, x], dim=1)
        x = self.conv(x)

        emb = self.emb_layer(t)[:, :, None, None].repeat(1, 1, x.shape[-2], x.shape[-1])

        return x + emb

class ConditionalUNet(nn.Module):
    def __init__(
            self, 
            in_dim: int = 2, 
            out_dim: int = 2,
            time_dim: int = 256,
            device: torch.device = "cuda"
            ):
        
        super(ConditionalUNet, self).__init__()
        
        self.device = device
        self.time_dim = time_dim

        # Down Block
        self.increase = DoubleConv(in_dim, 64, residual = False)
        self.down_1 = Down(64, 128)
        self.self_attn_1 = SelfAttention(128)
        self.down_2 = Down(128, 256)
        self.self_attn_2 = SelfAttention(256)
        self.down_3 = Down(256, 256, down_kernel=(5,1), down_stride= (5,1))
        self.self_attn_3 = SelfAttention(256)

        # Bottle Neck Block
        self.bottle_neck_1 = DoubleConv(256, 512)
        self.bottle_neck_2 = DoubleConv(512, 512)
        self.bottle_neck_3 = DoubleConv(512, 256)

        # Up Block
        self.up_1 = Up(512, 128, kernel_size=3)
        self.self_attn_4 = SelfAttention(128)
        self.up_2 = Up(256, 64, kernel_size = 3)
        self.self_attn_5 = SelfAttention(64)
        self.up_3 = Up(128, 64, kernel_size = 3)
        self.self_attn_6 = SelfAttention(64)
        
        self.out_layer = nn.Conv2d(64, out_dim, kernel_size = (1,1))

    def pos_encoding(self, t, dim):

        inv_freq = 1.0 / (
            10e4
            ** (torch.arange(0, dim, 2, device=t.device).float() / dim)
        )

        pos_enc_a = torch.sin(t.repeat(1, dim // 2) * inv_freq)
        pos_enc_b = torch.cos(t.repeat(1, dim // 2) * inv_freq)
        
        return torch.cat([pos_enc_a, pos_enc_b], dim =-1)

    def forward(self, noisy_traj, t, prompt):

        # Denoising timestep
        t = t.unsqueeze(-1).type(torch.float) 
        t = self.pos_encoding(t, self.time_dim)

        # TODO Increase dims for t and prompt embeddings
        t += prompt

        # Down block
        x1 = self.increase(noisy_traj)
        x2 = self.down_1(x1, t)
        x2 = self.self_attn_1(x2)
        x3 = self.down_2(x2, t)
        x3 = self.self_attn_2(x3)
        x4 = self.down_3(x3, t)
        x4 = self.self_attn_3(x4)

        # Bottle Neck
        x4 = self.bottle_neck_1(x4)
        x4 = self.bottle_neck_2(x4)
        x4 = self.bottle_neck_3(x4)

        # Up Block
        x5 = self.up_1(x4, x3, t)
        x5 = self.self_attn_4(x5)
        x5 = self.up_2(x5, x2, t)
        x5 = self.self_attn_5(x5)
        x5 = self.up_3(x5, x1, t)
        x5 = self.self_attn_6(x5)

        traj = self.out_layer(x5)

        return traj
    

class Diffusion:
    def __init__(
            self,
            noise_steps: int,
            beta_start: float,
            beta_end: float,
            device
            ):
        
        self.noise_steps = noise_steps
        self.beta_start = beta_start
        self.beta_end = beta_end

        self.beta = self.prepare_noise_schedule().to(device)
        self.alpha = 1.0 - self.beta
        self.alpha_hat = torch.cumprod(self.alpha, dim = 0)

    def prepare_noise_schedule(self):
        return torch.linspace(
            self.beta_start, 
            self.beta_end, 
            self.noise_steps
            )
    
    def sample_timesteps(self, batch_size):
        return torch.randint(
            low=1, 
            high=self.noise_steps, 
            size=(batch_size,)
            )

    def noise_trajectory(self, noise, traj, t):

        sqrt_alpha_hat = torch.sqrt(self.alpha_hat[t])[:, None, None, None]
        sqrt_one_minus_alpha_hat = torch.sqrt(1 - self.alpha_hat[t])[:, None, None, None]

        # Noise must be sampled from the model's gaussian prior

        return sqrt_alpha_hat * traj + sqrt_one_minus_alpha_hat * noise, noise