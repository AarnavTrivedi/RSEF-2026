"""
Neural Surrogate for MPPD (Multiple-Path Particle Dosimetry)

This module provides a differentiable approximation of the MPPD model,
enabling gradient-based optimization in the PI-VAE framework.

The surrogate learns to map:
    (MMAD, GSD, Concentration, Duration, BreathingRate) → (F_TB, F_ALV, M_Retained)

Where:
    - MMAD: Mass Median Aerodynamic Diameter (μm)
    - GSD: Geometric Standard Deviation
    - F_TB: Tracheobronchial Deposition Fraction
    - F_ALV: Alveolar Deposition Fraction
    - M_Retained: Retained Mass at 24h
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from pathlib import Path
from typing import Tuple, Optional, Dict, List
from loguru import logger
from tqdm import tqdm


class MPPDSurrogate(nn.Module):
    """
    Differentiable Neural Surrogate for the MPPD Model.
    
    This network learns the physics of aerosol deposition from MPPD simulations,
    providing the differentiable "Physics Engine" for the PI-VAE.
    
    Architecture:
        - Input: [MMAD, GSD, Conc, Duration, BreathRate] (5 dims)
        - Hidden: 4 layers × 256 neurons with LeakyReLU
        - Output: [F_TB, F_ALV, M_Retained] (3 dims)
    
    The output layer uses:
        - Sigmoid for fractions (F_TB, F_ALV) to ensure 0-1 range
        - Softplus for mass (M_Retained) to ensure positivity
    """
    
    def __init__(
        self,
        input_dim: int = 5,
        hidden_dims: List[int] = [256, 256, 256, 256],
        output_dim: int = 3,
        dropout: float = 0.1
    ):
        super(MPPDSurrogate, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim
        
        # Build the network
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.LeakyReLU(0.1),
                nn.BatchNorm1d(hidden_dim),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
        
        self.hidden_layers = nn.Sequential(*layers)
        
        # Separate output heads for different physical quantities
        self.head_fractions = nn.Linear(prev_dim, 2)  # F_TB, F_ALV
        self.head_mass = nn.Linear(prev_dim, 1)       # M_Retained
        
        # Initialize weights
        self._init_weights()
        
        logger.info(f"Initialized MPPDSurrogate: {input_dim} → {hidden_dims} → {output_dim}")
    
    def _init_weights(self):
        """Initialize weights using Kaiming initialization."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, nonlinearity='leaky_relu')
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the surrogate.
        
        Args:
            x: Input tensor of shape (batch_size, 5)
               [MMAD, GSD, Conc, Duration, BreathRate]
        
        Returns:
            Output tensor of shape (batch_size, 3)
            [F_TB, F_ALV, M_Retained]
        """
        # Pass through hidden layers
        h = self.hidden_layers(x)
        
        # Deposition fractions (must be 0-1)
        fractions = torch.sigmoid(self.head_fractions(h))
        
        # Retained mass (must be positive)
        mass = F.softplus(self.head_mass(h))
        
        # Concatenate outputs
        output = torch.cat([fractions, mass], dim=-1)
        
        return output
    
    def predict_deposition(
        self,
        mmad: torch.Tensor,
        gsd: torch.Tensor,
        concentration: torch.Tensor,
        duration: torch.Tensor,
        breath_rate: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Convenience method for prediction with named parameters.
        
        Returns:
            Dictionary with keys: 'F_TB', 'F_ALV', 'M_Retained'
        """
        # Stack inputs
        x = torch.stack([mmad, gsd, concentration, duration, breath_rate], dim=-1)
        
        # Forward pass
        output = self.forward(x)
        
        return {
            'F_TB': output[:, 0],
            'F_ALV': output[:, 1],
            'M_Retained': output[:, 2]
        }


class MPPDSurrogateTrainer:
    """
    Training utilities for the MPPD Surrogate.
    
    Handles:
        - Data loading and preprocessing
        - Training loop with early stopping
        - Validation and metrics computation
        - Model checkpointing
    """
    
    def __init__(
        self,
        model: MPPDSurrogate,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        self.model = model.to(device)
        self.device = device
        self.optimizer = torch.optim.Adam(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=10
        )
        self.criterion = nn.MSELoss()
        
        # Training history
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'val_r2': []
        }
        
        logger.info(f"Trainer initialized on device: {device}")
    
    def prepare_data(
        self,
        inputs: np.ndarray,
        targets: np.ndarray,
        val_split: float = 0.2,
        batch_size: int = 512
    ) -> Tuple[DataLoader, DataLoader]:
        """
        Prepare data loaders for training.
        
        Args:
            inputs: MPPD input parameters (N, 5)
            targets: MPPD output values (N, 3)
            val_split: Fraction for validation
            batch_size: Batch size
        
        Returns:
            train_loader, val_loader
        """
        # Convert to tensors
        X = torch.FloatTensor(inputs)
        y = torch.FloatTensor(targets)
        
        # Normalize inputs (log-transform for MMAD, Conc)
        X_normalized = self._normalize_inputs(X)
        
        # Train/val split
        n_val = int(len(X) * val_split)
        indices = torch.randperm(len(X))
        
        train_idx = indices[n_val:]
        val_idx = indices[:n_val]
        
        train_dataset = TensorDataset(X_normalized[train_idx], y[train_idx])
        val_dataset = TensorDataset(X_normalized[val_idx], y[val_idx])
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        logger.info(f"Data prepared: {len(train_dataset)} train, {len(val_dataset)} val")
        
        return train_loader, val_loader
    
    def _normalize_inputs(self, X: torch.Tensor) -> torch.Tensor:
        """
        Normalize inputs for better training.
        
        Log-transform MMAD (col 0) and Concentration (col 2).
        """
        X_norm = X.clone()
        X_norm[:, 0] = torch.log10(X[:, 0] + 1e-6)  # MMAD
        X_norm[:, 2] = torch.log10(X[:, 2] + 1e-6)  # Concentration
        return X_norm
    
    def train_epoch(self, train_loader: DataLoader) -> float:
        """Run one training epoch."""
        self.model.train()
        total_loss = 0.0
        
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(self.device)
            batch_y = batch_y.to(self.device)
            
            self.optimizer.zero_grad()
            output = self.model(batch_x)
            loss = self.criterion(output, batch_y)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            total_loss += loss.item()
        
        return total_loss / len(train_loader)
    
    @torch.no_grad()
    def validate(self, val_loader: DataLoader) -> Tuple[float, float]:
        """Run validation and compute metrics."""
        self.model.eval()
        total_loss = 0.0
        all_preds = []
        all_targets = []
        
        for batch_x, batch_y in val_loader:
            batch_x = batch_x.to(self.device)
            batch_y = batch_y.to(self.device)
            
            output = self.model(batch_x)
            loss = self.criterion(output, batch_y)
            total_loss += loss.item()
            
            all_preds.append(output.cpu())
            all_targets.append(batch_y.cpu())
        
        # Compute R²
        preds = torch.cat(all_preds, dim=0).numpy()
        targets = torch.cat(all_targets, dim=0).numpy()
        
        ss_res = np.sum((targets - preds) ** 2)
        ss_tot = np.sum((targets - targets.mean(axis=0)) ** 2)
        r2 = 1 - (ss_res / (ss_tot + 1e-8))
        
        return total_loss / len(val_loader), r2
    
    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int = 100,
        early_stopping_patience: int = 20,
        target_r2: float = 0.99,
        checkpoint_dir: Optional[Path] = None
    ) -> Dict:
        """
        Full training loop with early stopping.
        
        Args:
            train_loader: Training data
            val_loader: Validation data
            epochs: Maximum epochs
            early_stopping_patience: Epochs without improvement before stopping
            target_r2: Stop training when R² exceeds this
            checkpoint_dir: Directory for saving checkpoints
        
        Returns:
            Training history dictionary
        """
        best_r2 = -np.inf
        patience_counter = 0
        
        if checkpoint_dir:
            checkpoint_dir = Path(checkpoint_dir)
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Starting training for {epochs} epochs (target R² = {target_r2})")
        
        pbar = tqdm(range(epochs), desc="Training Surrogate")
        
        for epoch in pbar:
            # Train
            train_loss = self.train_epoch(train_loader)
            
            # Validate
            val_loss, val_r2 = self.validate(val_loader)
            
            # Update scheduler
            self.scheduler.step(val_loss)
            
            # Record history
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['val_r2'].append(val_r2)
            
            # Update progress bar
            pbar.set_postfix({
                'train_loss': f'{train_loss:.4f}',
                'val_loss': f'{val_loss:.4f}',
                'R²': f'{val_r2:.4f}'
            })
            
            # Check for improvement
            if val_r2 > best_r2:
                best_r2 = val_r2
                patience_counter = 0
                
                # Save best model
                if checkpoint_dir:
                    self.save_checkpoint(checkpoint_dir / 'best_surrogate.pth')
            else:
                patience_counter += 1
            
            # Early stopping checks
            if val_r2 >= target_r2:
                logger.success(f"Target R² ({target_r2}) achieved at epoch {epoch+1}!")
                break
            
            if patience_counter >= early_stopping_patience:
                logger.warning(f"Early stopping at epoch {epoch+1} (no improvement for {early_stopping_patience} epochs)")
                break
        
        logger.info(f"Training complete. Best R² = {best_r2:.4f}")
        
        return self.history
    
    def save_checkpoint(self, path: Path):
        """Save model checkpoint."""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'history': self.history
        }, path)
        logger.debug(f"Checkpoint saved to {path}")
    
    def load_checkpoint(self, path: Path):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.history = checkpoint.get('history', self.history)
        logger.info(f"Checkpoint loaded from {path}")


class MPPDDataGenerator:
    """
    Generate synthetic MPPD simulation data for surrogate training.
    
    Since MPPD v3.04 is a Windows executable without a Python API,
    this class provides utilities to:
        1. Generate parameter sweeps
        2. Create MPPD input files
        3. Parse MPPD output files
        4. (Alternative) Use analytical approximations for rapid prototyping
    """
    
    def __init__(
        self,
        mmad_range: Tuple[float, float] = (0.01, 20.0),
        gsd_range: Tuple[float, float] = (1.1, 3.0),
        conc_range: Tuple[float, float] = (1.0, 5000.0),
        duration_range: Tuple[float, float] = (0.1, 480.0),  # minutes
        breath_rate_range: Tuple[float, float] = (10.0, 25.0)
    ):
        self.mmad_range = mmad_range
        self.gsd_range = gsd_range
        self.conc_range = conc_range
        self.duration_range = duration_range
        self.breath_rate_range = breath_rate_range
    
    def generate_parameter_sweep(self, n_samples: int = 100000) -> np.ndarray:
        """
        Generate random parameter combinations for MPPD simulation.
        
        Uses log-uniform for MMAD and Concentration, uniform for others.
        
        Returns:
            Array of shape (n_samples, 5)
        """
        # Log-uniform sampling for MMAD
        log_mmad = np.random.uniform(
            np.log10(self.mmad_range[0]),
            np.log10(self.mmad_range[1]),
            n_samples
        )
        mmad = 10 ** log_mmad
        
        # Uniform sampling for GSD
        gsd = np.random.uniform(
            self.gsd_range[0],
            self.gsd_range[1],
            n_samples
        )
        
        # Log-uniform sampling for Concentration
        log_conc = np.random.uniform(
            np.log10(self.conc_range[0]),
            np.log10(self.conc_range[1]),
            n_samples
        )
        concentration = 10 ** log_conc
        
        # Uniform sampling for Duration and Breathing Rate
        duration = np.random.uniform(
            self.duration_range[0],
            self.duration_range[1],
            n_samples
        )
        
        breath_rate = np.random.uniform(
            self.breath_rate_range[0],
            self.breath_rate_range[1],
            n_samples
        )
        
        params = np.column_stack([mmad, gsd, concentration, duration, breath_rate])
        
        logger.info(f"Generated {n_samples} parameter combinations")
        
        return params
    
    def analytical_deposition_approximation(
        self,
        params: np.ndarray,
        species: str = 'human'
    ) -> np.ndarray:
        """
        Analytical approximation of MPPD deposition for rapid prototyping.
        
        This uses simplified deposition equations based on:
        - Impaction (Stokes number)
        - Sedimentation (settling velocity)
        - Diffusion (diffusion coefficient)
        
        NOTE: This is for development/testing. Real training should use
        actual MPPD simulations.
        
        Args:
            params: Array of shape (N, 5) [MMAD, GSD, Conc, Duration, BreathRate]
            species: 'human' or 'rat'
        
        Returns:
            Array of shape (N, 3) [F_TB, F_ALV, M_Retained]
        """
        mmad = params[:, 0]  # μm
        gsd = params[:, 1]
        conc = params[:, 2]  # μg/m³
        duration = params[:, 3]  # minutes
        breath_rate = params[:, 4]  # breaths/min
        
        # Physical constants
        mu = 1.81e-5  # Air viscosity (Pa·s)
        rho_p = 1000  # Particle density (kg/m³)
        g = 9.81  # Gravity (m/s²)
        k_B = 1.38e-23  # Boltzmann constant
        T = 310  # Temperature (K)
        
        # Convert MMAD to meters
        d_p = mmad * 1e-6
        
        # Cunningham slip correction factor (simplified)
        lambda_air = 65e-9  # Mean free path (m)
        Cc = 1 + (2 * lambda_air / d_p) * (1.257 + 0.4 * np.exp(-0.55 * d_p / lambda_air))
        
        # Relaxation time
        tau = (rho_p * d_p**2 * Cc) / (18 * mu)
        
        # Settling velocity
        v_s = tau * g
        
        # Diffusion coefficient
        D = (k_B * T * Cc) / (3 * np.pi * mu * d_p)
        
        # Simplified deposition model
        # TB deposition (impaction-dominated for large particles)
        # Characteristic velocity in TB region ~1 m/s
        U_tb = 1.0  # m/s
        L_tb = 0.01  # Characteristic length (m)
        Stk = tau * U_tb / L_tb
        eta_imp = np.minimum(1.0, 0.8 * Stk**0.5)  # Impaction efficiency
        
        # Alveolar deposition (diffusion-dominated for small particles)
        # Characteristic time in alveolar region
        t_alv = 2.0  # seconds
        R_alv = 0.1e-3  # Alveolar radius (m)
        Delta = D * t_alv / R_alv**2
        eta_diff = np.minimum(1.0, 5.78 * Delta**(2/3))  # Diffusion efficiency
        
        # Sedimentation (affects both regions)
        eta_sed = np.minimum(1.0, v_s * 1.0 / (0.001))  # Simplified
        
        # Combined deposition fractions
        # Large particles (>2.5 μm): mostly TB
        # Small particles (<0.5 μm): mostly ALV
        # Medium particles: mixed
        
        F_TB = np.clip(eta_imp * 0.8 + eta_sed * 0.2, 0, 0.6)
        F_ALV = np.clip(eta_diff * 0.6 + eta_sed * 0.3, 0, 0.8)
        
        # Particle size effects
        # Large particles deposit more in TB
        large_particle_mask = mmad > 2.5
        F_TB[large_particle_mask] *= 1.5
        F_ALV[large_particle_mask] *= 0.3
        
        # Small particles deposit more in ALV
        small_particle_mask = mmad < 0.5
        F_TB[small_particle_mask] *= 0.3
        F_ALV[small_particle_mask] *= 1.5
        
        # Clip to valid range
        F_TB = np.clip(F_TB, 0, 1)
        F_ALV = np.clip(F_ALV, 0, 1)
        
        # Ensure total deposition <= 1
        total_dep = F_TB + F_ALV
        over_unity = total_dep > 1
        F_TB[over_unity] /= total_dep[over_unity]
        F_ALV[over_unity] /= total_dep[over_unity]
        
        # Retained mass (simplified)
        # Volume of air breathed
        tidal_volume = 0.5e-3  # m³ (500 mL)
        total_volume = breath_rate * duration * tidal_volume  # m³
        inhaled_mass = conc * total_volume * 1e-9  # Convert μg/m³ to kg
        
        # Retained mass with 24h clearance (simplified)
        clearance_factor = 0.7  # 30% cleared in 24h
        M_retained = inhaled_mass * (F_TB + F_ALV) * clearance_factor * 1e6  # Convert back to μg
        
        output = np.column_stack([F_TB, F_ALV, M_retained])
        
        logger.info(f"Generated analytical approximations for {len(params)} samples")
        
        return output


def create_and_train_surrogate(
    n_samples: int = 100000,
    epochs: int = 100,
    target_r2: float = 0.99,
    checkpoint_dir: Path = Path("models/surrogate"),
    use_analytical: bool = True
) -> Tuple[MPPDSurrogate, Dict]:
    """
    Complete pipeline to create and train the MPPD surrogate.
    
    Args:
        n_samples: Number of training samples
        epochs: Training epochs
        target_r2: Target R² score
        checkpoint_dir: Where to save checkpoints
        use_analytical: If True, use analytical approximation (for development)
    
    Returns:
        Trained model and training history
    """
    logger.info("=" * 60)
    logger.info("MPPD Surrogate Training Pipeline")
    logger.info("=" * 60)
    
    # Generate data
    generator = MPPDDataGenerator()
    params = generator.generate_parameter_sweep(n_samples)
    
    if use_analytical:
        logger.warning("Using analytical approximation (for development only)")
        targets = generator.analytical_deposition_approximation(params)
    else:
        raise NotImplementedError(
            "Real MPPD integration requires Windows environment and MPPD v3.04. "
            "Use use_analytical=True for development."
        )
    
    # Create model
    model = MPPDSurrogate()
    
    # Create trainer
    trainer = MPPDSurrogateTrainer(model)
    
    # Prepare data
    train_loader, val_loader = trainer.prepare_data(
        params, targets,
        val_split=0.2,
        batch_size=512
    )
    
    # Train
    history = trainer.train(
        train_loader, val_loader,
        epochs=epochs,
        target_r2=target_r2,
        checkpoint_dir=checkpoint_dir
    )
    
    return model, history


# =============================================================================
# Testing
# =============================================================================
if __name__ == "__main__":
    # Quick test
    logger.info("Testing MPPD Surrogate module...")
    
    # Create model
    model = MPPDSurrogate()
    
    # Test forward pass
    batch_size = 32
    test_input = torch.randn(batch_size, 5)
    test_input[:, 0] = torch.abs(test_input[:, 0]) + 0.01  # MMAD > 0
    test_input[:, 1] = torch.abs(test_input[:, 1]) + 1.1   # GSD > 1.1
    test_input[:, 2] = torch.abs(test_input[:, 2]) * 100   # Concentration
    test_input[:, 3] = torch.abs(test_input[:, 3]) * 60    # Duration
    test_input[:, 4] = torch.abs(test_input[:, 4]) * 10 + 10  # Breath rate
    
    output = model(test_input)
    
    logger.info(f"Input shape: {test_input.shape}")
    logger.info(f"Output shape: {output.shape}")
    logger.info(f"F_TB range: [{output[:, 0].min():.3f}, {output[:, 0].max():.3f}]")
    logger.info(f"F_ALV range: [{output[:, 1].min():.3f}, {output[:, 1].max():.3f}]")
    logger.info(f"M_Retained range: [{output[:, 2].min():.3f}, {output[:, 2].max():.3f}]")
    
    logger.success("MPPD Surrogate module test passed!")
