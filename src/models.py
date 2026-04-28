"""
models.py — MLP architecture for the snowmelt emulator.
"""

import torch
import torch.nn as nn


class SnowmeltMLP(nn.Module):
    """
    Small MLP that predicts daily snowmelt (mm/day).
    
    Architecture: 4 inputs → [64 → 64 → 64] → 1 output
    Activation: ReLU
    Output: raw scalar (no activation) — loss functions handle clipping
    
    Inputs (in order, normalised):
        t2m_c   : daily mean temperature (°C)
        rad_mj  : net solar radiation (MJ/m²/day)
        swe_mm  : snow water equivalent (mm) — previous day
        elev_m  : station elevation (m)
    """
    
    def __init__(self, n_features=4, hidden=64, n_layers=3):
        super().__init__()
        
        layers = []
        in_dim = n_features
        for _ in range(n_layers):
            layers += [nn.Linear(in_dim, hidden), nn.ReLU()]
            in_dim = hidden
        layers += [nn.Linear(hidden, 1)]
        
        self.net = nn.Sequential(*layers)
    
    def forward(self, x):
        """
        Parameters
        ----------
        x : torch.Tensor  shape (batch, n_features)
        
        Returns
        -------
        out : torch.Tensor  shape (batch,)  predicted melt
        """
        return self.net(x).squeeze(-1)