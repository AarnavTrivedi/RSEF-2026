"""
Physics-Informed Variational Autoencoder (PI-VAE) for Pulmonary Dosimetry

This module implements the core reverse-inference engine that:
1. Encodes transcriptomic signatures into a latent space
2. Grounds the latent space in physical units (MMAD, GSD, Conc, Duration)
3. Uses a frozen MPPD surrogate to enforce physical consistency
4. Reconstructs the biological signal to validate the inference

The key insight: The latent space is PARTITIONED into:
    - z_phys: Physical parameters (MMAD, GSD, Concentration, Duration, BreathRate)
    - z_bio: Unexplained biological variance (genetic background, health state, noise)

The Physics Loss compares:
    - Neural Surrogate prediction (based on z_phys) 
    - Actual biological damage pattern (from deconvolved transcriptomics)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, kl_divergence
from typing import Dict, Tuple, Optional, List
import numpy as np
from pathlib import Path
from loguru import logger

from .surrogate import MPPDSurrogate

# Import ICRP-based surrogate for physics loss
try:
    from .mppd_surrogate import MPPDSurrogate as ICRPSurrogate, create_trained_surrogate
    ICRP_AVAILABLE = True
except ImportError:
    ICRP_AVAILABLE = False
    logger.warning("ICRP surrogate not available - physics loss will use default surrogate")


class Encoder(nn.Module):
    """
    Encoder network: Maps gene expression to latent space.
    
    Input: Deconvolved gene expression vector (top 2000 variable genes)
    Output: Mean (μ) and log-variance (log σ²) of latent distribution
    
    The latent space is partitioned:
        - z_phys (dims 0-4): Physical parameters
        - z_bio (dims 5+): Biological context
    """
    
    def __init__(
        self,
        input_dim: int = 2000,
        hidden_dims: List[int] = [512, 256, 128],
        latent_dim: int = 8,
        dropout: float = 0.2
    ):
        super(Encoder, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.latent_dim = latent_dim
        
        # Build encoder network
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.LeakyReLU(0.1),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
        
        self.encoder = nn.Sequential(*layers)
        
        # Latent space heads
        self.fc_mu = nn.Linear(prev_dim, latent_dim)
        self.fc_logvar = nn.Linear(prev_dim, latent_dim)
        
        self._init_weights()
        
        logger.info(f"Encoder: {input_dim} → {hidden_dims} → {latent_dim}")
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            x: Gene expression tensor (batch_size, input_dim)
        
        Returns:
            mu: Mean of latent distribution (batch_size, latent_dim)
            logvar: Log-variance of latent distribution (batch_size, latent_dim)
        """
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        
        return mu, logvar


class BiologicalDecoder(nn.Module):
    """
    Decoder network: Reconstructs gene expression from latent space.
    
    Uses BOTH z_phys and z_bio to reconstruct the full transcriptome.
    """
    
    def __init__(
        self,
        latent_dim: int = 8,
        hidden_dims: List[int] = [128, 256, 512],
        output_dim: int = 2000,
        dropout: float = 0.2
    ):
        super(BiologicalDecoder, self).__init__()
        
        self.latent_dim = latent_dim
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim
        
        # Build decoder network
        layers = []
        prev_dim = latent_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.LeakyReLU(0.1),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, output_dim))
        
        self.decoder = nn.Sequential(*layers)
        
        self._init_weights()
        
        logger.info(f"BiologicalDecoder: {latent_dim} → {hidden_dims} → {output_dim}")
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            z: Latent vector (batch_size, latent_dim)
        
        Returns:
            x_recon: Reconstructed gene expression (batch_size, output_dim)
        """
        return self.decoder(z)


class PhysicsDecoder(nn.Module):
    """
    Physics Decoder: Wraps the frozen MPPD surrogate.
    
    This module:
    1. Extracts z_phys from the full latent vector
    2. Transforms z_phys to physical units using scaling functions
    3. Passes through the frozen MPPD surrogate
    4. Returns predicted regional deposition
    
    The scaling ensures physical validity:
        - MMAD: 0.01 - 20.0 μm (log-scale)
        - GSD: 1.1 - 3.0
        - Concentration: 1 - 5000 μg/m³ (log-scale)
        - Duration: 0.1 - 480 minutes (log-scale)
        - Breath Rate: 10 - 25 breaths/min
    """
    
    def __init__(
        self,
        surrogate: MPPDSurrogate,
        z_phys_dim: int = 5,
        freeze_surrogate: bool = True
    ):
        super(PhysicsDecoder, self).__init__()
        
        self.surrogate = surrogate
        self.z_phys_dim = z_phys_dim
        
        # Freeze surrogate weights
        if freeze_surrogate:
            for param in self.surrogate.parameters():
                param.requires_grad = False
            logger.info("MPPD Surrogate frozen (no gradients)")
        
        # Physical bounds for scaling
        self.register_buffer('mmad_min', torch.tensor(0.01))
        self.register_buffer('mmad_max', torch.tensor(20.0))
        self.register_buffer('gsd_min', torch.tensor(1.1))
        self.register_buffer('gsd_max', torch.tensor(3.0))
        self.register_buffer('conc_min', torch.tensor(1.0))
        self.register_buffer('conc_max', torch.tensor(5000.0))
        self.register_buffer('duration_min', torch.tensor(0.1))
        self.register_buffer('duration_max', torch.tensor(480.0))
        self.register_buffer('breath_min', torch.tensor(10.0))
        self.register_buffer('breath_max', torch.tensor(25.0))
        
        # Learnable scaling factors (α, β) for aligning bio units to physics units
        self.alpha = nn.Parameter(torch.tensor(1.0))  # TB scaling
        self.beta = nn.Parameter(torch.tensor(1.0))   # ALV scaling
        
        logger.info(f"PhysicsDecoder initialized with z_phys_dim={z_phys_dim}")
    
    def _scale_latent_to_physics(self, z_phys: torch.Tensor) -> torch.Tensor:
        """
        Transform latent variables to physical units.
        
        Uses sigmoid to bound values, then scales to physical ranges.
        Log-scale for MMAD, Concentration, Duration (spans orders of magnitude).
        """
        # Apply sigmoid to get 0-1 range
        z_sigmoid = torch.sigmoid(z_phys)
        
        # MMAD (log-scale)
        log_mmad_min = torch.log10(self.mmad_min)
        log_mmad_max = torch.log10(self.mmad_max)
        mmad = 10 ** (log_mmad_min + z_sigmoid[:, 0] * (log_mmad_max - log_mmad_min))
        
        # GSD (linear scale)
        gsd = self.gsd_min + z_sigmoid[:, 1] * (self.gsd_max - self.gsd_min)
        
        # Concentration (log-scale)
        log_conc_min = torch.log10(self.conc_min)
        log_conc_max = torch.log10(self.conc_max)
        conc = 10 ** (log_conc_min + z_sigmoid[:, 2] * (log_conc_max - log_conc_min))
        
        # Duration (log-scale)
        log_dur_min = torch.log10(self.duration_min)
        log_dur_max = torch.log10(self.duration_max)
        duration = 10 ** (log_dur_min + z_sigmoid[:, 3] * (log_dur_max - log_dur_min))
        
        # Breathing rate (linear scale)
        breath_rate = self.breath_min + z_sigmoid[:, 4] * (self.breath_max - self.breath_min)
        
        # Stack into physics input
        physics_input = torch.stack([mmad, gsd, conc, duration, breath_rate], dim=-1)
        
        return physics_input
    
    def forward(
        self,
        z: torch.Tensor,
        return_physics_params: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through physics decoder.
        
        Args:
            z: Full latent vector (batch_size, latent_dim)
            return_physics_params: If True, also return scaled physical parameters
        
        Returns:
            Dictionary with:
                - 'F_TB': Predicted TB deposition fraction
                - 'F_ALV': Predicted Alveolar deposition fraction
                - 'M_Retained': Predicted retained mass
                - 'physics_params': (optional) Scaled physical parameters
        """
        # Extract z_phys (first z_phys_dim dimensions)
        z_phys = z[:, :self.z_phys_dim]
        
        # Scale to physical units
        physics_input = self._scale_latent_to_physics(z_phys)
        
        # Normalize for surrogate (log-transform MMAD and Conc)
        physics_normalized = physics_input.clone()
        physics_normalized[:, 0] = torch.log10(physics_input[:, 0] + 1e-6)
        physics_normalized[:, 2] = torch.log10(physics_input[:, 2] + 1e-6)
        
        # Detect if using Modern Surrogate (has 'icrp_rest' buffer)
        is_modern_surrogate = hasattr(self.surrogate, 'icrp_rest')
        
        if is_modern_surrogate:
            # Modern Surrogate: Expects raw values as kwargs (handles logging internally)
            # Physics Params: 0=MMAD, 1=GSD, 2=Conc, 3=Duration, 4=BreathRate
            surrogate_output = self.surrogate(
                mmad=physics_input[:, 0],
                gsd=physics_input[:, 1],
                # density defaults to 1.0 in surrogate
                breathing_freq=physics_input[:, 4],
                # tidal_volume/flow_rate default in surrogate
            )
        else:
            # Legacy Surrogate: Expects single tensor with log-transformed inputs
            surrogate_output = self.surrogate(physics_normalized)
        
        # Handle dict output from MPPD surrogate
        # Mapping:
        # F_TB = Tracheobronchial (BB + bb)
        # F_ALV = Pulmonary (AI)
        # M_Retained = Pulmonary (approximation, as ALV clears slowly)
        
        if isinstance(surrogate_output, dict):
            f_tb = surrogate_output['tracheobronchial']
            f_alv = surrogate_output['pulmonary']
        else:
            # Fallback if surrogate returns tensor
            if surrogate_output.shape[1] == 5:
                # Assuming tensor shape [batch, 5] -> [ET1, ET2, BB, bb, AI]
                f_tb = surrogate_output[:, 2] + surrogate_output[:, 3]
                f_alv = surrogate_output[:, 4]
            elif surrogate_output.shape[1] == 3:
                # Legacy surrogate: [TB, ALV, Retained]
                f_tb = surrogate_output[:, 0]
                f_alv = surrogate_output[:, 1]
            else:
                # Unknown shape, default to 0
                f_tb = surrogate_output[:, 0]
                f_alv = surrogate_output[:, 1]
            
        result = {
            'F_TB': f_tb,
            'F_ALV': f_alv,
            'M_Retained': f_alv  # Approximation
        }
        
        if isinstance(surrogate_output, dict) and 'regional_df' in surrogate_output:
             result['regional_df'] = surrogate_output['regional_df']
        
        if return_physics_params:
            result['physics_params'] = {
                'MMAD': physics_input[:, 0],
                'GSD': physics_input[:, 1],
                'Concentration': physics_input[:, 2],
                'Duration': physics_input[:, 3],
                'BreathRate': physics_input[:, 4]
            }
        
        return result


class PIVAE(nn.Module):
    """
    Physics-Informed Variational Autoencoder for Pulmonary Dosimetry.
    
    This is the main model that:
    1. Encodes transcriptomic data to latent space
    2. Partitions latent space into z_phys and z_bio
    3. Reconstructs biology using the full latent vector
    4. Validates physics using z_phys through the surrogate
    
    Loss function:
        L_total = L_recon + β*L_KL + γ*L_physics + δ*L_supervised
    
    Where:
        - L_recon: Reconstruction of gene expression
        - L_KL: KL divergence for latent regularization
        - L_physics: Physics consistency via MPPD surrogate
        - L_supervised: (optional) Direct supervision on z_phys
    """
    
    def __init__(
        self,
        input_dim: int = 2000,
        latent_dim: int = 8,
        z_phys_dim: int = 5,
        encoder_hidden: List[int] = [512, 256, 128],
        decoder_hidden: List[int] = [128, 256, 512],
        surrogate: Optional[MPPDSurrogate] = None,
        dropout: float = 0.2
    ):
        super(PIVAE, self).__init__()
        
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.z_phys_dim = z_phys_dim
        self.z_bio_dim = latent_dim - z_phys_dim
        
        # ICRP-based deposition surrogate (pre-trained on benchmark data)
        self.icrp_surrogate = None
        if ICRP_AVAILABLE:
            try:
                self.icrp_surrogate = create_trained_surrogate(n_epochs=300)
                logger.info("ICRP deposition surrogate initialized (trained on ICRP 66 benchmarks)")
            except Exception as e:
                logger.warning(f"Could not create ICRP surrogate: {e}")
        
        # Encoder
        self.encoder = Encoder(
            input_dim=input_dim,
            hidden_dims=encoder_hidden,
            latent_dim=latent_dim,
            dropout=dropout
        )
        
        # Biological Decoder
        self.bio_decoder = BiologicalDecoder(
            latent_dim=latent_dim,
            hidden_dims=decoder_hidden,
            output_dim=input_dim,
            dropout=dropout
        )
        
        # Physics Decoder (requires pre-trained surrogate)
        if surrogate is None:
            logger.warning("No surrogate provided - creating untrained surrogate")
            surrogate = MPPDSurrogate()
        
        self.physics_decoder = PhysicsDecoder(
            surrogate=surrogate,
            z_phys_dim=z_phys_dim,
            freeze_surrogate=True
        )
        
        # Biomarker indices (set these based on your gene list)
        self.biomarker_indices = {
            'AKR1B10': None,  # Bronchial marker
            'CYP1A1': None,   # Bronchial marker
            'MMP12': None,    # Alveolar marker
            'SPP1': None      # Alveolar marker
        }
        
        logger.info(f"PI-VAE initialized: latent_dim={latent_dim}, z_phys={z_phys_dim}, z_bio={self.z_bio_dim}")
    
    def set_biomarker_indices(self, gene_list: List[str]):
        """
        Set the indices of biomarker genes in the gene expression vector.
        
        Args:
            gene_list: List of gene names in order
        """
        for gene in self.biomarker_indices.keys():
            if gene in gene_list:
                self.biomarker_indices[gene] = gene_list.index(gene)
                logger.info(f"Biomarker {gene} at index {self.biomarker_indices[gene]}")
            else:
                logger.warning(f"Biomarker {gene} not found in gene list")
    
    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """
        Reparameterization trick for VAE training.
        
        z = μ + σ * ε, where ε ~ N(0, 1)
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode input to latent distribution parameters."""
        return self.encoder(x)
    
    def decode_biology(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent to reconstructed gene expression."""
        return self.bio_decoder(z)
    
    def decode_physics(
        self,
        z: torch.Tensor,
        return_params: bool = False
    ) -> Dict[str, torch.Tensor]:
        """Decode latent to physical deposition prediction."""
        return self.physics_decoder(z, return_physics_params=return_params)
    
    def forward(
        self,
        x: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Full forward pass.
        
        Args:
            x: Gene expression (batch_size, input_dim)
        
        Returns:
            Dictionary containing all outputs needed for loss computation
        """
        # Encode
        mu, logvar = self.encode(x)
        
        # Sample latent
        z = self.reparameterize(mu, logvar)
        
        # Decode biology
        x_recon = self.decode_biology(z)
        
        # Decode physics
        physics_output = self.decode_physics(z, return_params=True)
        
        return {
            'x_recon': x_recon,
            'mu': mu,
            'logvar': logvar,
            'z': z,
            'z_phys': z[:, :self.z_phys_dim],
            'z_bio': z[:, self.z_phys_dim:],
            'F_TB': physics_output['F_TB'],
            'F_ALV': physics_output['F_ALV'],
            'M_Retained': physics_output['M_Retained'],
            'physics_params': physics_output['physics_params']
        }
    
    def compute_loss(
        self,
        x: torch.Tensor,
        outputs: Dict[str, torch.Tensor],
        bio_tb: Optional[torch.Tensor] = None,
        bio_alv: Optional[torch.Tensor] = None,
        z_true: Optional[torch.Tensor] = None,
        label_mask: Optional[torch.Tensor] = None,
        beta: float = 0.01,
        gamma: float = 1.0,
        delta: float = 10.0
    ) -> Dict[str, torch.Tensor]:
        """
        Compute the composite loss function.
        
        L_total = L_recon + β*L_KL + γ*L_physics + δ*L_supervised
        
        Args:
            x: Original gene expression
            outputs: Forward pass outputs
            bio_tb: Biological TB signal (from deconvolution)
            bio_alv: Biological ALV signal (from deconvolution)
            z_true: Ground truth physical parameters (if available)
            beta: KL divergence weight
            gamma: Physics loss weight
            delta: Supervised loss weight
        
        Returns:
            Dictionary with individual and total losses
        """
        losses = {}
        
        # 1. Reconstruction Loss
        losses['recon'] = F.mse_loss(outputs['x_recon'], x)
        
        # 2. KL Divergence Loss
        # KL(q(z|x) || p(z)) where p(z) = N(0, 1)
        kl_loss = -0.5 * torch.sum(
            1 + outputs['logvar'] - outputs['mu'].pow(2) - outputs['logvar'].exp(),
            dim=1
        )
        losses['kl'] = kl_loss.mean()
        
        # 3. Physics Loss
        if bio_tb is not None and bio_alv is not None:
            alpha = self.physics_decoder.alpha
            beta_scale = self.physics_decoder.beta
            
            # Compare predicted deposition to biological signal
            physics_loss_tb = F.mse_loss(
                outputs['F_TB'],
                alpha * bio_tb
            )
            physics_loss_alv = F.mse_loss(
                outputs['F_ALV'],
                beta_scale * bio_alv
            )
            losses['physics'] = physics_loss_tb + physics_loss_alv
        else:
            losses['physics'] = torch.tensor(0.0, device=x.device)
        
        # 4. Supervised Loss (optional)
        # 4. Supervised Loss (optional)
        if z_true is not None:
            # If mask is provided, use it
            if label_mask is not None:
                # Ensure booleanmask 
                mask = label_mask.bool()
                if mask.any():
                    # Compute MSE only on labeled samples
                    # TARGET: outputs['physics_params'] (Decoded Parameters) vs z_true
                    # Need to extract relevant params matching z_true order [MMAD, GSD, Conc, Dur, Rate]
                    # physics_decoder.forward returns 'physics_params' dict.
                    # We need to stack them back to tensor or compare element-wise?
                    # outputs['physics_params'] might be a dict. Check forward.
                    # Yes, it is a dict. We need to stack or compare key-by-key.
                    # z_true is tensor [B, 5]. 
                    # Col 0: MMAD, 1: GSD, 2: Conc, 3: Duration, 4: BreathRate.
                    
                    phys_dict = outputs.get('physics_params')
                    if phys_dict is not None:
                         # Stack predictions: [B, 5]
                         pred_phys = torch.stack([
                             phys_dict['MMAD'],
                             phys_dict['GSD'],
                             phys_dict['Concentration'],
                             phys_dict['Duration'],
                             phys_dict['BreathRate']
                         ], dim=1)
                         
                         losses['supervised'] = F.mse_loss(
                            pred_phys[mask],
                            z_true[mask]
                         )
                    else:
                         # Fallback if params not returned (should not happen if return_params=True)
                         losses['supervised'] = torch.tensor(0.0, device=x.device)
                else:
                    losses['supervised'] = torch.tensor(0.0, device=x.device)
            else:
                # Standard full supervision
                phys_dict = outputs.get('physics_params')
                if phys_dict is not None:
                     pred_phys = torch.stack([
                             phys_dict['MMAD'],
                             phys_dict['GSD'],
                             phys_dict['Concentration'],
                             phys_dict['Duration'],
                             phys_dict['BreathRate']
                     ], dim=1)
                     losses['supervised'] = F.mse_loss(
                        pred_phys,
                        z_true
                     )
                else:
                     losses['supervised'] = torch.tensor(0.0, device=x.device)
        else:
            losses['supervised'] = torch.tensor(0.0, device=x.device)
        
        # 5. ICRP Deposition Consistency Loss (NEW)
        # Ensures predicted F_TB + F_ALV are physically plausible
        if self.icrp_surrogate is not None:
            # Extract MMAD from physics params (if available)
            physics_params = outputs.get('physics_params', None)
            if physics_params is not None:
                mmad = physics_params['MMAD']
                
                # Get ICRP predictions for these MMADs
                with torch.no_grad():
                    icrp_pred = self.icrp_surrogate(
                        mmad=mmad,
                        activity_level=torch.zeros_like(mmad)  # Assume rest
                    )
                
                # Physics consistency: predicted F_TB/F_ALV should be similar to ICRP
                # Scale factor accounts for dose-response relationship
                icrp_tb = icrp_pred['tracheobronchial'].detach()
                icrp_alv = icrp_pred['pulmonary'].detach()
                
                # Soft constraint: predictions should be proportional to ICRP fractions
                # Allow learned scaling since gene expression isn't linear with deposition
                losses['icrp_consistency'] = (
                    F.mse_loss(outputs['F_TB'] / (outputs['F_TB'].mean() + 1e-6),
                               icrp_tb / (icrp_tb.mean() + 1e-6)) +
                    F.mse_loss(outputs['F_ALV'] / (outputs['F_ALV'].mean() + 1e-6),
                               icrp_alv / (icrp_alv.mean() + 1e-6))
                )
            else:
                losses['icrp_consistency'] = torch.tensor(0.0, device=x.device)
        else:
            losses['icrp_consistency'] = torch.tensor(0.0, device=x.device)
        
        # Total Loss (with ICRP consistency weighted by epsilon)
        epsilon = 0.5  # Weight for ICRP consistency
        losses['total'] = (
            losses['recon'] +
            beta * losses['kl'] +
            gamma * losses['physics'] +
            delta * losses['supervised'] +
            epsilon * losses['icrp_consistency']
        )
        
        return losses
    
    def infer_exposure(
        self,
        x: torch.Tensor,
        n_samples: int = 100
    ) -> Dict[str, Dict[str, float]]:
        """
        Infer exposure parameters with uncertainty quantification.
        
        Samples from the latent distribution multiple times to get
        credible intervals for the physical parameters.
        
        Args:
            x: Gene expression (1, input_dim) - single sample
            n_samples: Number of samples for uncertainty estimation
        
        Returns:
            Dictionary with mean and CI for each physical parameter
        """
        self.eval()
        
        with torch.no_grad():
            # Encode
            mu, logvar = self.encode(x)
            
            # Sample multiple times
            samples = []
            for _ in range(n_samples):
                z = self.reparameterize(mu, logvar)
                physics_output = self.decode_physics(z, return_params=True)
                samples.append(physics_output['physics_params'])
            
            # Aggregate results
            results = {}
            param_names = ['MMAD', 'GSD', 'Concentration', 'Duration', 'BreathRate']
            
            for name in param_names:
                values = torch.stack([s[name] for s in samples]).cpu().numpy()
                results[name] = {
                    'mean': float(np.mean(values)),
                    'std': float(np.std(values)),
                    'ci_low': float(np.percentile(values, 2.5)),
                    'ci_high': float(np.percentile(values, 97.5))
                }
        
        return results
    
    def get_deposition_prediction(
        self,
        x: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Get deposition prediction for a sample.
        
        Args:
            x: Gene expression
        
        Returns:
            Deposition fractions and physical parameters
        """
        self.eval()
        
        with torch.no_grad():
            outputs = self.forward(x)
            
            return {
                'F_TB': outputs['F_TB'],
                'F_ALV': outputs['F_ALV'],
                'M_Retained': outputs['M_Retained'],
                'MMAD': outputs['physics_params']['MMAD'],
                'GSD': outputs['physics_params']['GSD'],
                'Concentration': outputs['physics_params']['Concentration'],
                'Duration': outputs['physics_params']['Duration'],
                'BreathRate': outputs['physics_params']['BreathRate']
            }


def load_pivae(
    checkpoint_path: Path,
    surrogate_path: Path,
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
) -> PIVAE:
    """
    Load a trained PI-VAE model from checkpoint.
    
    Args:
        checkpoint_path: Path to PI-VAE checkpoint
        surrogate_path: Path to MPPD surrogate checkpoint
        device: Device to load model on
    
    Returns:
        Loaded PI-VAE model
    """
    # Load surrogate
    surrogate = MPPDSurrogate()
    surrogate_ckpt = torch.load(surrogate_path, map_location=device)
    surrogate.load_state_dict(surrogate_ckpt['model_state_dict'])
    
    # Load PI-VAE
    ckpt = torch.load(checkpoint_path, map_location=device)
    
    model = PIVAE(
        surrogate=surrogate,
        **ckpt.get('model_config', {})
    )
    model.load_state_dict(ckpt['model_state_dict'])
    model.to(device)
    model.eval()
    
    logger.info(f"Loaded PI-VAE from {checkpoint_path}")
    
    return model


# =============================================================================
# Testing
# =============================================================================
if __name__ == "__main__":
    logger.info("Testing PI-VAE module...")
    
    # Create surrogate (untrained for testing)
    surrogate = MPPDSurrogate()
    
    # Create PI-VAE
    model = PIVAE(
        input_dim=2000,
        latent_dim=8,
        z_phys_dim=5,
        surrogate=surrogate
    )
    
    # Test forward pass
    batch_size = 16
    x = torch.randn(batch_size, 2000)
    
    outputs = model(x)
    
    logger.info(f"Input shape: {x.shape}")
    logger.info(f"Reconstruction shape: {outputs['x_recon'].shape}")
    logger.info(f"Latent shape: {outputs['z'].shape}")
    logger.info(f"z_phys shape: {outputs['z_phys'].shape}")
    logger.info(f"z_bio shape: {outputs['z_bio'].shape}")
    logger.info(f"F_TB shape: {outputs['F_TB'].shape}")
    logger.info(f"F_ALV shape: {outputs['F_ALV'].shape}")
    
    # Test loss computation
    bio_tb = torch.rand(batch_size)
    bio_alv = torch.rand(batch_size)
    
    losses = model.compute_loss(x, outputs, bio_tb, bio_alv)
    
    logger.info(f"Loss components:")
    for name, value in losses.items():
        logger.info(f"  {name}: {value.item():.4f}")
    
    # Test inference
    single_sample = torch.randn(1, 2000)
    exposure = model.infer_exposure(single_sample, n_samples=50)
    
    logger.info(f"Inferred exposure:")
    for param, stats in exposure.items():
        logger.info(f"  {param}: {stats['mean']:.3f} [{stats['ci_low']:.3f}, {stats['ci_high']:.3f}]")
    
    logger.success("PI-VAE module test passed!")
