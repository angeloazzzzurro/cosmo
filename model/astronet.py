"""
COSMO — AstroNet (PyTorch)
Dual-stream 1D CNN per classificazione esopianeti da curve di luce Kepler/TESS.

Architettura (Shallue & Vanderburg 2018, adattata):
  Stream 1 — Global view  (2001 timestep): 5 blocchi Conv1D → Flatten
  Stream 2 — Local view   ( 201 timestep): 2 blocchi Conv1D → Flatten
  Fusion: concat → 4×Dense(512) → output probabilità
"""

import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel=5, pool=5, pool_stride=2):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel, padding=kernel // 2),
            nn.ReLU(),
            nn.Conv1d(out_ch, out_ch, kernel, padding=kernel // 2),
            nn.ReLU(),
            nn.MaxPool1d(pool, stride=pool_stride),
        )

    def forward(self, x):
        return self.block(x)


class GlobalStream(nn.Module):
    """5 blocchi su global view (2001 punti) → filtri 16→32→64→128→256"""
    def __init__(self):
        super().__init__()
        channels = [1, 16, 32, 64, 128, 256]
        self.blocks = nn.Sequential(*[
            ConvBlock(channels[i], channels[i + 1]) for i in range(5)
        ])

    def forward(self, x):                        # (B, 1, 2001)
        return self.blocks(x).flatten(1)


class LocalStream(nn.Module):
    """2 blocchi su local view (201 punti) → filtri 16→32"""
    def __init__(self):
        super().__init__()
        self.blocks = nn.Sequential(
            ConvBlock(1, 16, pool=7, pool_stride=2),
            ConvBlock(16, 32, pool=7, pool_stride=2),
        )

    def forward(self, x):                        # (B, 1, 201)
        return self.blocks(x).flatten(1)


class AstroNet(nn.Module):
    def __init__(self, dropout: float = 0.5):
        super().__init__()
        self.global_stream = GlobalStream()
        self.local_stream = LocalStream()

        # Calcola dimensioni flatten dinamicamente
        dummy_g = torch.zeros(1, 1, 2001)
        dummy_l = torch.zeros(1, 1, 201)
        g_size = self.global_stream(dummy_g).shape[1]
        l_size = self.local_stream(dummy_l).shape[1]
        fusion_size = g_size + l_size

        self.fc = nn.Sequential(
            nn.Linear(fusion_size, 512), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(512, 512),         nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(512, 512),         nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(512, 512),         nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(512, 1),
        )

    def forward(self, global_view, local_view):
        g = self.global_stream(global_view)
        l = self.local_stream(local_view)
        x = torch.cat([g, l], dim=1)
        return torch.sigmoid(self.fc(x)).squeeze(1)


if __name__ == "__main__":
    model = AstroNet()
    total = sum(p.numel() for p in model.parameters())
    print(f"AstroNet — parametri totali: {total:,}")
    g = torch.randn(4, 1, 2001)
    l = torch.randn(4, 1, 201)
    out = model(g, l)
    print(f"Output shape: {out.shape}  →  {out.detach().numpy()}")
