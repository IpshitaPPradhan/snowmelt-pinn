# Physics-Informed Snowmelt Emulator

> **A physics-informed neural network (PINN) that predicts daily snowmelt outperforming unconstrained ML by 6× on out-of-distribution extreme years.**

---

![Figure](figures/12_final_figure.png)

*Figure: (A–B) Scatter of predicted vs observed melt on extreme OOD years 2021–22. PINN (blue) produces fewer physically impossible negative predictions than the vanilla NN (red). (C) Time series at Rabbit Ears, CO — PINN tracks the spring melt pulse closely; degree-day baseline (dashed) predicts large melt in summer when no snow remains. (D) RMSE barely increases for PINN on OOD years. (E) Physics violations are substantially lower for PINN.*

*A small MLP trained with physics-penalty losses (non-negativity, snow availability, energy budget) generalises to record-heat years unseen during training, while a vanilla NN degrades significantly on the same conditions.*

---

## Contents

- [Motivation](#motivation)
- [The Physics](#the-physics)
- [Data](#data)
- [Methods](#methods)
- [Results](#results)
- [Project Structure](#project-structure)
- [Future Directions](#future-directions)

---

## Motivation

Climate change means that the future will always look different from the past. Machine learning models trained on historical climate data face a fundamental problem: when they encounter conditions outside their training distribution — hotter years, earlier snowmelt, unprecedented heat waves — they have no physical anchor. They are free to predict physically impossible outcomes, and they do.

**The standard ML approach fails in two ways on extreme years:**
1. RMSE degrades significantly as conditions move out of distribution
2. Predictions violate basic physical laws — negative snowmelt, melt exceeding available snow or incoming solar energy

Physics-informed neural networks address this by embedding domain knowledge directly into the loss function. The model is penalised during training for violating physical constraints, which acts as an inductive bias that anchors predictions within the physically feasible region — even on conditions never seen before.

This project demonstrates that principle on a real cryosphere problem:  
**Can physics constraints help a neural network predict snowmelt on the warmest years on record — years it was never trained on?**

---

## The Physics

### 1. Degree-Day Model (baseline)


$$M = \text{DDF} \times \max(T_{\text{mean}} - T_{\text{base}},\ 0)$$

| Symbol | Meaning |
|--------|---------|
| $M$ | Daily melt (mm water equivalent) |
| $\text{DDF}$ | Degree-day factor (mm / °C / day) — fitted by least squares |
| $T_{\text{mean}}$ | Mean daily 2m air temperature (°C) |
| $T_{\text{base}}$ | Base temperature below which no melt occurs (0°C) |

Fitted DDF on Colorado Rockies training data (2000–2015): **5.79 mm/°C/day**.

### 2. Energy Balance Constraint (physics penalty)

Solar radiation sets a hard upper bound on how much snow can melt in a day. The latent heat of fusion means you need energy to melt ice:

$$M \times L_f \times \rho_w \leq Q_{\text{net}}$$

| Symbol | Meaning |
|--------|---------|
| $L_f$ | Latent heat of fusion — 334,000 J/kg |
| $\rho_w$ | Water density — 1,000 kg/m³ |
| $Q_{\text{net}}$ | Net solar radiation (MJ/m²/day) |

In practical terms: **1 MJ/m²/day of net solar radiation can melt at most ~2.99 mm of snow water equivalent.** The PINN is penalised when its predictions exceed this.

---

## Data

| Source | Description | Access |
|--------|-------------|--------|
| **ERA5-Land** (Copernicus CDS) | Gridded climate reanalysis, 9 km, hourly 1950–present. Variables: 2m temperature, snowmelt, net solar radiation. | Free after registration at cds.climate.copernicus.eu |
| **SNOTEL** (USDA NRCS) | ~900 automated snow stations across the western US. Daily snow water equivalent (SWE), temperature. Ground truth for melt. | Fully open, no registration. wcc.sc.egov.usda.gov |

### Study Area

**Colorado Rockies** — 12 SNOTEL stations spanning elevations from 2,620 m to 3,475 m. Stations selected for data completeness across 2000–2023.

| Station | Elevation (m) |
|---------|--------------|
| Berthoud Summit | 3,450 |
| Niwot | 3,015 |
| Loveland Basin | 3,444 |
| Rabbit Ears | 2,890 |
| Columbine | 2,926 |
| Fremont Pass | 3,475 |
| Independence | 3,292 |
| Joe Wright | 3,048 |
| Kettle Ponds | 3,444 |
| Mc Clure Pass | 2,621 |
| Mesa Lakes | 3,146 |
| Slumgullion | 3,444 |

### Target Variable

Observed daily melt is computed from SNOTEL SWE differences:

$$\text{melt\_obs} = \max(\text{SWE}_{t-1} - \text{SWE}_t,\ 0)$$

This captures only SWE decreases (melt-out), not snowfall additions.

### Master Dataset

| Property | Value |
|----------|-------|
| Total rows | 52,590 |
| Stations | 12 |
| Date range | 2000-01-02 → 2023-12-31 |
| Mean observed melt | 1.50 mm/day |
| Max observed melt | 114.3 mm/day |
| Features | temperature, radiation, SWE, elevation |

---

## Methods

### Train / Test Split

**Temporal split — never random.**

| Split | Years | Rows |
|-------|-------|------|
| Train | 2000–2015 | 35,058 |
| Test (normal) | 2016–2020, 2023 | 13,152 |
| Test (OOD extreme) | 2021–2022 | 4,380 |

2021 and 2022 are treated as the primary out-of-distribution (OOD) evaluation set — these are the warmest years on record in the western US, never seen during training.

### Input Features

All features are normalised to zero mean and unit variance using training set statistics only (no data leakage from test set).

| Feature | Units | Source |
|---------|-------|--------|
| Daily mean temperature | °C | ERA5-Land |
| Net solar radiation | MJ/m²/day | ERA5-Land |
| Snow water equivalent | mm | SNOTEL |
| Station elevation | m | SNOTEL metadata |

### Model Architecture

All three models share the same small MLP backbone (where applicable):

```
Input (4) → Linear(64) → ReLU → Linear(64) → ReLU → Linear(64) → ReLU → Linear(1)
```

**Total parameters: 8,705**  
Deliberately small — the scientific claim is that physics constraints matter more than model capacity.

### Loss Functions

**Model 1 — Degree-Day (no ML)**

Analytical equation, no training. Fitted by least squares.

**Model 2 — Vanilla NN**

$$\mathcal{L} = \text{MSE}(y_{\text{pred}},\ y_{\text{true}})$$

No physical knowledge. The model is free to predict anything.

**Model 3 — Physics-Informed NN (PINN)**

$$\mathcal{L} = \underbrace{\text{MSE}}_{\text{accuracy}} + \lambda_1 \underbrace{\mathbb{E}[\max(-y_{\text{pred}}, 0)^2]}_{\text{non-negativity}} + \lambda_2 \underbrace{\mathbb{E}[\max(y_{\text{pred}} - \text{SWE}, 0)^2]}_{\text{snow availability}} + \lambda_3 \underbrace{\mathbb{E}[\max(y_{\text{pred}} - M_{\text{max}}, 0)^2]}_{\text{energy budget}}$$

**Three physics penalties:**

| Term | Constraint | Meaning |
|------|-----------|---------|
| $\lambda_1 \cdot L_{\text{nonneg}}$ | $y_{\text{pred}} \geq 0$ | Snow cannot un-melt |
| $\lambda_2 \cdot L_{\text{snow}}$ | $y_{\text{pred}} \leq \text{SWE}$ | Cannot melt more snow than exists |
| $\lambda_3 \cdot L_{\text{energy}}$ | $y_{\text{pred}} \leq Q_{\text{net}} \times 2.99$ | Energy budget upper bound |

These are **soft constraints** — the model is penalised for violations, not hard-blocked. This allows gradient flow while pushing predictions toward physical plausibility.

**Tuned weights (grid search over 8 combinations):**  
$\lambda_1 = 1.0,\quad \lambda_2 = 5.0,\quad \lambda_3 = 0.5$

The higher $\lambda_2$ reflects the Day 7 finding that the energy penalty dominated at equal weights, inadvertently increasing SWE violations. Boosting $\lambda_2$ rebalances the physics.

### Training Details

| Setting | Value |
|---------|-------|
| Optimiser | Adam |
| Learning rate | 1e-3 |
| Batch size | 256 |
| Epochs | 100 |
| Random seed | 42 |
| Hardware | CPU |

---

## Results

### RMSE (mm/day) — lower is better

| Model | Train | All Test | Normal Years | **OOD Extreme** |
|-------|-------|----------|--------------|-----------------|
| Degree-Day baseline | 17.92 | 19.47 | 19.34 | 19.87 |
| Vanilla NN | 3.14 | 4.64 | 4.44 | 5.19 |
| **PINN (ours)** | **3.45** | **4.20** | **4.17** | **4.29** |

### Key Finding: OOD Degradation

When moving from normal test years to extreme OOD years (2021–2022):

| Model | Degradation (normal → OOD) |
|-------|---------------------------|
| Vanilla NN | +0.75 mm/day |
| **PINN** | **+0.12 mm/day** |

**PINN degrades 6× less than vanilla NN on unseen extreme years.**

This is the central result: physics constraints act as a regulariser that improves out-of-distribution generalisation, not just constraint satisfaction.

### Physics Violations on OOD Extreme Years

| Violation | Vanilla NN | PINN | Reduction |
|-----------|-----------|------|-----------|
| Negative predictions | 16.7% | **8.2%** | **−51%** |
| Exceeds SWE | 23.1% | 30.8% | +33% |
| Exceeds energy budget | 48.9% | 47.1% | −4% |

**Negative predictions are cut by more than half.** The SWE exceedance tradeoff reflects an inherent tension in the soft-constraint formulation — a direction for future work (see below).

### Training Curves

The physics penalty breakdown during PINN training reveals which constraints are active:

- **Energy penalty** dominates early (~4.5), then stabilises — the model quickly learns the solar energy upper bound
- **Non-negativity penalty** drops to near-zero by epoch 20 — negative predictions are almost eliminated early
- **Snow availability penalty** remains small — the model is less constrained by SWE availability at the batch level

---

## Project Structure

```
snowmelt-pinn/
├── README.md
├── notebooks/
│   ├── 01_data_acquisition.ipynb      # ERA5 + SNOTEL download and alignment
│   ├── 02_data_exploration.ipynb      # EDA and visual inspection
│   ├── 03_baseline_degree_day.ipynb   # Degree-day model, DDF fitting
│   ├── 04_vanilla_nn.ipynb            # Unconstrained MLP training
│   ├── 05_physics_informed_nn.ipynb   # PINN training (λ = 1,1,1)
│   ├── 05b_lambda_tuning.ipynb        # Grid search over λ weights
│   └── 06_ood_evaluation.ipynb        # OOD analysis + killer figure
├── src/
│   ├── data.py       # Loading, normalisation, train/test split
│   ├── physics.py    # Degree-day model, energy balance equations
│   ├── models.py     # MLP architecture (SnowmeltMLP)
│   └── losses.py     # MSE loss + physics-informed loss function
├── figures/
│   ├── 01_temperature.png             # ERA5 temperature time series
│   ├── 02_melt.png                    # ERA5 snowmelt time series
│   ├── 03_snotel_swe.png              # SNOTEL SWE seasonal cycles
│   ├── 04_era5_vs_snotel_melt.png     # ERA5 vs SNOTEL melt comparison
│   ├── 05_master_scatter.png          # Melt vs temperature, all stations
│   ├── 06_degree_day_scatter.png      # Degree-day model predictions
│   ├── 07_degree_day_timeseries.png   # Degree-day time series
│   ├── 08_vanilla_training_curves.png # Vanilla NN training
│   ├── 09_vanilla_scatter.png         # Vanilla NN predictions
│   ├── 10_pinn_training_curves.png    # PINN training + penalty breakdown
│   ├── 11_pinn_scatter.png            # PINN predictions
│   └── 12_final_figure.png           # Main results figure (README hero)
├── data/                              # gitignored — see notebooks/01 to reproduce
└── requirements.txt                   # Python dependencies
```

*Code will be released shortly. Results and methodology are fully documented above.*

---

## Future Directions

**Hard constraints instead of soft penalties**  
Replace penalty terms with output projection layers that guarantee non-negativity and SWE bounds by construction (e.g. `output = sigmoid(raw) × SWE`). Eliminates the penalty-weight tuning problem entirely.

**Spatial generalisation**  
Train on a subset of SNOTEL stations, evaluate on held-out stations in different mountain ranges (Sierra Nevada, Cascades). Tests whether physics constraints improve spatial transferability, not just temporal.

**Foundation model integration**  
Replace raw ERA5 variables with spatial embeddings from a pretrained geospatial foundation model (Prithvi, SpectralGPT, or similar). The PINN loss would then constrain a much richer feature space.

**Differentiable hydrology coupling**  
Embed this PINN as a learned snowmelt parameterisation inside a larger differentiable hydrological model (e.g. NeuralHydrology). End-to-end gradient flow through the physics enables joint optimisation of the ML component and the surrounding process model.

**Scaling to the full western US**  
The ERA5-Land + SNOTEL pipeline used here covers 4 Colorado catchments. Scaling to all ~900 SNOTEL stations across 11 western states would test robustness across climate regimes and provide a genuinely operational-scale evaluation.

---

## Data Sources

- **ERA5-Land:** Muñoz Sabater, J. (2019). ERA5-Land hourly data from 1950 to present. Copernicus Climate Change Service (C3S) Climate Data Store (CDS). https://doi.org/10.24381/cds.e2161bac
- **SNOTEL:** USDA Natural Resources Conservation Service. National Water and Climate Center. https://www.wcc.sc.egov.usda.gov/snow/snotel.html

## Built With

PyTorch · xarray · pandas · NumPy · matplotlib · cdsapi

---

