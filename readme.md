# PulmoTrace

## Reverse-Inference Pulmonary Dosimetry via Physics-Informed Variational Autoencoders

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## Overview

**PulmoTrace** is a computational framework for reconstructing inhalational exposure history from transcriptomic signatures. By treating the lung as a biological sensor array, this system inverts the causal chain of toxicology—from transcriptomic *effect* back to physical *cause*—using a deep generative model constrained by the deterministic laws of aerosol physics.

### The Problem

Traditional exposure assessment relies on:
- Self-reported questionnaires (unreliable, subject to recall bias)
- Environmental monitoring (doesn't capture individual dose)
- Clinical symptoms (appear late in disease progression)

**The Disconnect:** Two individuals exposed to identical concentrations of particulate matter may receive vastly different internal doses due to differences in breathing patterns, airway geometry, and particle aerodynamics.

### The Solution

PulmoTrace uses a **Physics-Informed Variational Autoencoder (PI-VAE)** that:

1. **Encodes** high-dimensional transcriptomic data into a compact latent space
2. **Grounds** the latent space in physical units (particle size, concentration, duration)
3. **Validates** inferences against the laws of aerosol physics via a differentiable MPPD surrogate
4. **Outputs** a forensic reconstruction of the exposure scenario

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         PulmoTrace Pipeline                              │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│   Gene Expression       ─────►   PI-VAE Encoder   ─────►   Latent z    │
│   (20,000 genes)                                           [z_phys,    │
│                                                             z_bio]     │
│                                                               │        │
│                                           ┌───────────────────┴───┐    │
│                                           │                       │    │
│                                           ▼                       ▼    │
│                                    ┌─────────────┐        ┌───────────┐│
│                                    │   Physics   │        │ Biological││
│                                    │   Decoder   │        │  Decoder  ││
│                                    │   (MPPD)    │        │           ││
│                                    └──────┬──────┘        └─────┬─────┘│
│                                           │                     │      │
│                                           ▼                     ▼      │
│                                    ┌─────────────┐        ┌───────────┐│
│                                    │ Deposition  │        │Reconstructed
│                                    │  Fractions  │        │   Genes   ││
│                                    │ (F_TB,F_ALV)│        │           ││
│                                    └─────────────┘        └───────────┘│
│                                           │                     │      │
│                                           ▼                     ▼      │
│                                    ┌─────────────────────────────────┐ │
│                                    │      Physics Loss + Recon Loss  │ │
│                                    │      (Constrained Optimization) │ │
│                                    └─────────────────────────────────┘ │
│                                                                         │
│   OUTPUT: Forensic Exposure Report                                     │
│   ├── Particle Size (MMAD): 0.08 μm [95% CI: 0.05-0.12]               │
│   ├── Concentration: 312 μg/m³ [95% CI: 180-450]                      │
│   └── Duration: 2.3 hours [95% CI: 1.5-3.1]                           │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Key Features

- **Physics-Constrained Inference**: Latent variables must satisfy the laws of aerosol transport (MPPD)
- **Uncertainty Quantification**: Bayesian inference provides credible intervals, not just point estimates
- **Interpretable Outputs**: Results are in physical units (microns, μg/m³, hours)
- **Multi-fidelity Ready**: Architecture supports CFD surrogate integration for higher fidelity
- **Cross-Species Transfer**: Biological encoder learns universal stress responses; physics decoder is species-specific

---

## Project Structure

```
pulmotrace/
├── __init__.py              # Package initialization
├── config.yaml              # Configuration file
├── requirements.txt         # Python dependencies
├── train.py                 # Training pipeline
├── infer.py                 # Inference engine
│
├── models/
│   ├── __init__.py
│   ├── surrogate.py         # MPPD Neural Surrogate
│   └── pivae.py             # Physics-Informed VAE
│
├── data/
│   ├── __init__.py
│   └── processing.py        # Data loading and preprocessing
│
├── outputs/
│   ├── logs/                # Training logs
│   ├── figures/             # Visualization outputs
│   └── results/             # Inference results
│
└── models/
    └── checkpoints/         # Saved model weights
```

---

## Installation

```bash
# Clone repository
git clone https://github.com/yourusername/pulmotrace.git
cd pulmotrace

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

### Requirements

- Python 3.10+
- PyTorch 2.0+
- CUDA 11.8+ (optional, for GPU acceleration)

---

## Quick Start

### 1. Train the Model

```bash
# Full training pipeline
python train.py --config config.yaml

# Skip surrogate training (if checkpoint exists)
python train.py --config config.yaml --skip-surrogate
```

### 2. Run Inference

```bash
# Infer exposure from expression data
python infer.py \
    --input data/patient_expression.csv \
    --output results.json \
    --report report.txt
```

### 3. Programmatic Usage

```python
from pulmotrace import PIVAE, MPPDSurrogate
from pulmotrace.data import prepare_training_data

# Load trained model
surrogate = MPPDSurrogate()
surrogate.load_state_dict(torch.load('models/checkpoints/best_surrogate.pth')['model_state_dict'])

model = PIVAE(surrogate=surrogate)
model.load_state_dict(torch.load('models/checkpoints/best_pivae.pth')['model_state_dict'])
model.eval()

# Infer exposure
results = model.infer_exposure(expression_tensor, n_samples=100)

print(f"Particle Size: {results['MMAD']['mean']:.2f} μm")
print(f"Concentration: {results['Concentration']['mean']:.1f} μg/m³")
```

---

## The Science

### Aerosol Deposition Physics

The PI-VAE leverages the deterministic relationship between particle properties and regional lung deposition:

| Particle Size | Dominant Mechanism | Primary Deposition Region |
|--------------|-------------------|--------------------------|
| > 5 μm | Inertial Impaction | Upper airways (bronchi) |
| 0.5 - 5 μm | Gravitational Sedimentation | Mid-to-lower airways |
| < 0.5 μm | Brownian Diffusion | Deep alveoli |

This creates a unique *spatial fingerprint* in the transcriptome that allows reverse inference.

### Biological Markers

The model uses established "dosimeter genes" that record regional exposure:

| Gene | Region | Role |
|------|--------|------|
| AKR1B10 | Bronchial | Oxidative stress response |
| CYP1A1 | Bronchial | PAH metabolism |
| MMP12 | Alveolar | Macrophage activation |
| SPP1 | Alveolar | Fibrotic remodeling |
| AHRR (methylation) | Systemic | Cumulative duration clock |

### Loss Function

The PI-VAE optimizes a composite objective:

```
L_total = L_recon + β·L_KL + γ·L_physics + δ·L_supervised

where:
  L_recon   = ||x - x̂||²                    (biological fidelity)
  L_KL      = KL[q(z|x) || p(z)]            (latent regularization)
  L_physics = ||MPPD(z_phys) - BioSignal||² (physical consistency)
  L_supervised = ||z_phys - z_true||²       (optional ground truth)
```

---

## Training Data

The model is trained on public GEO datasets:

| Dataset | Organism | Exposure | N | Role |
|---------|----------|----------|---|------|
| GSE25531 | Human | Diesel Exhaust (Controlled) | 14 | Supervised |
| GSE47460 | Human | Smoking (Chronic) | 582 | Semi-supervised |
| GSE237251 | Rat | Wood Smoke | 12 | Cross-species |

---

## Model Architecture

### Neural Surrogate (MPPD)

```
Input: [MMAD, GSD, Conc, Duration, BreathRate] (5 dims)
  │
  ├── Linear(5, 256) + LeakyReLU + BatchNorm + Dropout
  ├── Linear(256, 256) + LeakyReLU + BatchNorm + Dropout
  ├── Linear(256, 256) + LeakyReLU + BatchNorm + Dropout
  ├── Linear(256, 256) + LeakyReLU + BatchNorm + Dropout
  │
  ├── Head: Fractions (Sigmoid) → [F_TB, F_ALV]
  └── Head: Mass (Softplus) → [M_Retained]
```

### PI-VAE Encoder

```
Input: Gene Expression (2000 dims)
  │
  ├── Linear(2000, 512) + BatchNorm + LeakyReLU + Dropout
  ├── Linear(512, 256) + BatchNorm + LeakyReLU + Dropout
  ├── Linear(256, 128) + BatchNorm + LeakyReLU + Dropout
  │
  ├── μ Head: Linear(128, 8)
  └── σ Head: Linear(128, 8)
  
Latent Space (8 dims):
  ├── z_phys (dims 0-4): [MMAD, GSD, Conc, Duration, BreathRate]
  └── z_bio (dims 5-7): [Unexplained biological variance]
```

---

## Future Enhancements

- [ ] **CFD Integration**: Multi-fidelity surrogate with 3D CFD for higher accuracy
- [ ] **Patient-Specific Geometry**: CT-derived lung meshes for personalized physics
- [ ] **Desktop Application**: PySide6 GUI with real-time 3D visualization
- [ ] **Multi-mixture Modeling**: Support for complex exposure mixtures

---

## Citation

If you use PulmoTrace in your research, please cite:

```bibtex
@software{pulmotrace2025,
  title = {PulmoTrace: Reverse-Inference Pulmonary Dosimetry via Physics-Informed VAE},
  author = {Your Name},
  year = {2025},
  url = {https://github.com/yourusername/pulmotrace}
}
```

---

## License

MIT License - see [LICENSE](LICENSE) for details.

---

## Acknowledgments

- MPPD Model: Applied Research Associates (ARA)
- Aerosol physics foundations: W.C. Hinds, "Aerosol Technology"
- Physics-Informed Neural Networks: Raissi et al., 2019