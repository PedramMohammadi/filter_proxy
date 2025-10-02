#!/usr/bin/env python3
import torch
import torch.nn as nn
import torch.nn.functional as F

# ----- decoder with matched concats -----
def _cat_skip(x, skip):
    # Ensure spatial sizes match the skip (robust to any rounding/odd sizes)
    if x.shape[-2:] != skip.shape[-2:]:
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
    return torch.cat([x, skip], dim=1)

class PreProcessorNet(nn.Module):
    def __init__(self, n_frames=1, crop_size=128):
        """
        Neural network for pre-processing video frames to enhance VMAF scores.
        
        Args:
            n_frames (int): Number of input frames (default=1 for intra-coding).
            crop_size (int): Spatial size of input frames (default=128).
        """
        super(PreProcessorNet, self).__init__()
        self.n_frames = n_frames
        self.crop_size = crop_size

        # Initial convolution: 3D for n_frames>1, 2D for n_frames=1
        if n_frames > 1:
            self.initial_conv = nn.Conv3d(1, 32, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        else:
            self.initial_conv = nn.Conv2d(1, 32, kernel_size=3, padding=1)

        # Encoder: 3 levels of Conv2D + GroupNorm + ReLU + MaxPool
        self.encoder = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.GroupNorm(8, 64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.GroupNorm(8, 128),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.GroupNorm(8, 256),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        # Bottleneck: 2 residual blocks
        self.bottleneck = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.GroupNorm(8, 256),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.GroupNorm(8, 256),
            nn.ReLU()
        )

        # Decoder: 3 levels of UpConv + Concat + Conv2D
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.Conv2d(384, 128, kernel_size=3, padding=1),
            nn.GroupNorm(8, 128),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.Conv2d(192, 64, kernel_size=3, padding=1),
            nn.GroupNorm(8, 64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.Conv2d(96, 32, kernel_size=3, padding=1),
            nn.GroupNorm(8, 32),
            nn.ReLU()
        )

        # DCM: Blur and edge enhancement branches
        self.dcm = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(32, 32, kernel_size=5, padding=2),
                nn.GroupNorm(8, 32)
            ),
            nn.Sequential(
                nn.Conv2d(32, 32, kernel_size=3, padding=1),
                nn.GroupNorm(8, 32)
            )
        ])
        self.dcm_fuse = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=1),
            nn.GroupNorm(8, 32)
        )

        # Channel attention for content-adaptive DCM
        self.dcm_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(32, 16, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(16, 64, kernel_size=1),
            nn.Sigmoid()
        )

        # Output convolution
        self.out_conv = nn.Conv2d(32, 1, kernel_size=3, padding=1)

    def forward(self, x):
        """
        Forward pass for pre-processing frames.
        
        Args:
            x (torch.Tensor): Input tensor of shape [batch, n_frames, crop_size, crop_size]
        
        Returns:
            torch.Tensor: Pre-processed frame of shape [batch, 1, crop_size, crop_size]
        """
        # Validate input dimensions
        expected_shape = (x.size(0), self.n_frames, self.crop_size, self.crop_size)
        if x.shape != expected_shape:
            raise ValueError(f"Expected input shape {expected_shape}, got {x.shape}")

        input_frame = x[:, 0 if self.n_frames == 1 else self.n_frames // 2].unsqueeze(1)

        if self.n_frames > 1:
            x = x.unsqueeze(1)                         # [B,1,T,H,W]
            x = F.relu(self.initial_conv(x))           # [B,32,T,H,W]
            x = x[:, :, x.shape[2] // 2]               # [B,32,H,W] center time slice
        else:
            x = F.relu(self.initial_conv(x))           # [B,32,H,W]

        # ----- collecting skips -----
        s0 = x                                         # 32-ch, full res

        # block 1
        x = self.encoder[0](x); x = self.encoder[1](x); x = self.encoder[2](x)   # Conv(32->64), GN, ReLU
        s1 = x                                         # 64-ch, full res
        x = self.encoder[3](x)                         # MaxPool -> 1/2

        # block 2
        x = self.encoder[4](x); x = self.encoder[5](x); x = self.encoder[6](x)   # Conv(64->128), GN, ReLU
        s2 = x                                         # 128-ch, 1/2
        x = self.encoder[7](x)                         # MaxPool -> 1/4

        # block 3
        x = self.encoder[8](x); x = self.encoder[9](x); x = self.encoder[10](x)  # Conv(128->256), GN, ReLU
        s3 = x                                         # 256-ch, 1/4
        x = self.encoder[11](x)                        # MaxPool -> 1/8

        # bottleneck
        x = self.bottleneck(x)                         # 256-ch, 1/8

        # up to 1/4, concat with s3 -> channels: 128 + 256 = 384
        x = self.decoder[0](x); x = F.relu(x)             # ConvT(256->128)
        x = _cat_skip(x, s3)
        x = self.decoder[2](x); x = self.decoder[3](x); x = F.relu(x)   # Conv2d(384->128), GN, ReLU

        # up to 1/2, concat with s2 -> 64 + 128 = 192
        x = self.decoder[5](x); x = F.relu(x)
        x = _cat_skip(x, s2)
        x = self.decoder[7](x); x = self.decoder[8](x); x = F.relu(x)

        # up to 1x, concat with s1 -> 32 + 64 = 96
        x = self.decoder[10](x); x = F.relu(x) 
        x = _cat_skip(x, s1)
        x = self.decoder[12](x); x = self.decoder[13](x); x = F.relu(x) 

        dcm_outs = [F.relu(conv(x)) for conv in self.dcm]
        dcm_cat = torch.cat(dcm_outs, dim=1)
        attn = self.dcm_attention(x)
        dcm_cat = dcm_cat * attn
        x = F.relu(self.dcm_fuse(dcm_cat))

        x = torch.tanh(self.out_conv(x))
        x = torch.clamp(x + input_frame, 0.0, 1.0)
        return x