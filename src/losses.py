"""
losses.py — vanilla MSE loss and physics-informed loss.
"""

import torch


def mse_loss(y_pred, y_true):
    """Standard mean squared error."""
    return torch.mean((y_pred - y_true) ** 2)


def physics_informed_loss(y_pred, y_true, swe_mm, rad_mj,
                           lambda1=1.0, lambda2=1.0, lambda3=1.0):
    """
    Physics-informed loss = MSE + three penalty terms.
    
    Penalty 1 — non-negativity:
        Snow cannot un-melt. Penalise negative predictions.
        L1 = mean(max(-y_pred, 0)²)
    
    Penalty 2 — snow availability:
        Cannot melt more snow than exists.
        L2 = mean(max(y_pred - swe_mm, 0)²)
    
    Penalty 3 — energy budget:
        Cannot melt more than available solar energy allows.
        1 MJ/m²/day ≈ 2.99 mm SWE (from L_f and ρ_w)
        L3 = mean(max(y_pred - max_melt_energy, 0)²)
    
    Parameters
    ----------
    y_pred   : (N,) predicted melt (mm/day)
    y_true   : (N,) observed melt  (mm/day)
    swe_mm   : (N,) snow water equivalent (mm) — upper bound on melt
    rad_mj   : (N,) net solar radiation (MJ/m²/day)
    lambda1/2/3 : penalty weights, start at 1.0
    
    Returns
    -------
    total_loss : scalar tensor
    components : dict with individual loss values (for logging)
    """
    # Core MSE
    L_mse = mse_loss(y_pred, y_true)
    
    # Penalty 1: no negative melt
    L_nonneg = torch.mean(torch.clamp(-y_pred, min=0.0) ** 2)
    
    # Penalty 2: can't melt more than SWE available
    L_snow = torch.mean(torch.clamp(y_pred - swe_mm, min=0.0) ** 2)
    
    # Penalty 3: energy budget — 1 MJ/m²/day melts ~2.99 mm SWE
    max_melt_energy = rad_mj * 2.99   # mm/day
    L_energy = torch.mean(torch.clamp(y_pred - max_melt_energy, min=0.0) ** 2)
    
    total = L_mse + lambda1 * L_nonneg + lambda2 * L_snow + lambda3 * L_energy
    
    components = {
        "mse":    L_mse.item(),
        "nonneg": L_nonneg.item(),
        "snow":   L_snow.item(),
        "energy": L_energy.item(),
        "total":  total.item(),
    }
    return total, components