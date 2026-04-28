"""
physics.py — degree-day model and energy balance constraint.
"""

import numpy as np


# Physical constants
L_F     = 334_000.0   # latent heat of fusion (J/kg)
RHO_W   = 1_000.0     # water density (kg/m³)
MJ_TO_J = 1_000_000.0 # 1 MJ = 1e6 J


def degree_day_model(T_mean, DDF=5.79, T_base=0.0):
    """
    Predict daily melt using degree-day equation.
    
    M = DDF × max(T_mean − T_base, 0)
    
    Parameters
    ----------
    T_mean : array-like  daily mean temperature (°C)
    DDF    : float       degree-day factor (mm/°C/day), fitted = 5.79
    T_base : float       base temperature (°C), default 0
    
    Returns
    -------
    melt : np.ndarray  predicted melt (mm/day), always >= 0
    """
    T_mean = np.asarray(T_mean, dtype=np.float32)
    return DDF * np.maximum(T_mean - T_base, 0.0)


def max_melt_from_energy(Q_net_mj):
    """
    Maximum physically possible melt given net solar radiation.
    
    From energy balance: M × L_f × ρ_w ≤ Q_net
    → M_max = Q_net / (L_f × ρ_w)   [in metres]
    → convert to mm
    
    Parameters
    ----------
    Q_net_mj : array-like  net solar radiation (MJ/m²/day)
    
    Returns
    -------
    max_melt_mm : np.ndarray  maximum possible melt (mm/day)
    """
    Q_net_mj = np.asarray(Q_net_mj, dtype=np.float32)
    Q_net_j  = Q_net_mj * MJ_TO_J                     # J/m²
    M_max_m  = Q_net_j / (L_F * RHO_W)               # metres
    return M_max_m * 1000.0                             # mm


def rmse(y_true, y_pred):
    """Root mean squared error."""
    return float(np.sqrt(np.mean((np.asarray(y_true) - np.asarray(y_pred))**2)))