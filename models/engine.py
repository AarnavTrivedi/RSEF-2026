
"""
PulmoTrace Engine
-----------------
The core logic engine for the PulmoTrace Dashboard.
Encapsulates model loading, inference, and biological alignment.
"""

import sys
import os
import torch
import numpy as np
import pandas as pd
from typing import Dict, Tuple, List, Optional, Any

# Ensure we can import from parent directory if needed
sys.path.insert(0, os.path.abspath('..'))

from models.pivae import PIVAE
from models.mppd_surrogate import create_trained_surrogate
from data.processing import SpatialDeconvolution
from data import prepare_geo_training_data

class PulmoTraceEngine:
    """
    Main engine for PulmoTrace inference.
    Handles loading the PI-VAE model, running predictions, 
    and interpreting physics/biology alignment.
    """
    
    # Validated list of 45 genes used in training
    VALIDATED_GENES = [
        'ADH7', 'AGER', 'AHR', 'AHRR', 'ALDH1A1', 'ALDH3A1', 'ARNT', 'BAX', 'BCL2', 
        'CAT', 'CCL2', 'CDKN1A', 'COL1A1', 'CXCL2', 'CYP1A1', 'CYP1B1', 'EPHX1', 
        'GCLC', 'GCLM', 'GPX2', 'GSR', 'HMOX1', 'IL1B', 'IL6', 'KEAP1', 'MMP12', 
        'MMP2', 'MMP9', 'MUC5B', 'NFE2L2', 'NLRP3', 'NQO1', 'PTGS2', 'S100A8', 
        'S100A9', 'SCGB1A1', 'SFTPC', 'SOD1', 'SOD2', 'SPP1', 'TGFB1', 'TIMP1', 
        'TIPARP', 'TNF', 'TP53'
    ]
    
    def __init__(self, checkpoints_dir: str = '../data/checkpoints', use_modern_surrogate: bool = True):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.checkpoints_dir = checkpoints_dir
        self.model = None
        self.deconv = None
        self.gene_names = None
        self.use_modern_surrogate = use_modern_surrogate
        
        # Test Data Cache
        self.test_data = None # {'X': np.array, 'meta': DataFrame}
        
        # Load everything on init
        self._load_resources()
        
    def _load_resources(self):
        """Load PI-VAE model, Surrogate, Deconv module, and Gene names."""
        print("Loading PulmoTrace Engine resources...")
        
        # 1. Load Deconvolution Module (for marker genes)
        self.deconv = SpatialDeconvolution()
        
        # Use the 45 physics genes hardcoded (Validation Step 1094)
        self.gene_names = self.VALIDATED_GENES
        print(f"Loaded {len(self.gene_names)} physics-informed genes.")

        # 2. Initialize Surrogate
        print("Initializing ICRP Surrogate...")
        icrp_surrogate = create_trained_surrogate(n_epochs=100)
        
        # 3. Load PI-VAE
        print("Loading PI-VAE Model...")
        model_path = os.path.join(self.checkpoints_dir, 'best_model.pt')
        if not os.path.exists(model_path):
             model_path = 'checkpoints/best_model.pt'
             
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model checkpoint not found at {model_path}")
            
        checkpoint = torch.load(model_path, map_location=self.device)
        
        self.model = PIVAE(
            input_dim=len(self.gene_names),
            latent_dim=8,
            z_phys_dim=5,
            encoder_hidden=[256, 128, 64],
            decoder_hidden=[64, 128, 256],
            surrogate=icrp_surrogate
        )
        
        self.model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        self.model.eval()
        self.model.to(self.device)
        
        print("Engine Loaded Successfully.")

    def load_test_data(self) -> pd.DataFrame:
        """
        Loads the official test set from data processing pipeline.
        Returns:
            DataFrame containing metadata for the test set (index=SampleID).
        """
        if self.test_data is None:
            print("Loading Official Test Set...")
            # Returns: dataloader, gene_names, metadata
            try:
                loader, gene_names, meta = prepare_geo_training_data()
                
                # Extract expression data from the dataset
                # loader.dataset is a PulmoTraceDataset instance
                if hasattr(loader.dataset, 'expression'):
                    X = loader.dataset.expression
                    if isinstance(X, torch.Tensor):
                        X = X.cpu().numpy()
                else:
                    # Fallback if structure is different
                    raise AttributeError("Could not extract expression data from dataset")

                self.test_data = {
                    'X': X,
                    'meta': meta
                }
                print(f"Test Set Loaded: {len(X)} samples.")
            except Exception as e:
                print(f"Error loading test set: {str(e)}")
                # Return empty if failed
                return pd.DataFrame()
            
        return self.test_data['meta']

    def get_test_sample(self, sample_id: str) -> Optional[np.ndarray]:
        """Retrieve expression data for a specific test sample."""
        if self.test_data is None:
            self.load_test_data()
            
        meta = self.test_data['meta']
        
        if sample_id not in meta.index:
            # Handle potential type mismatch (String vs Int)
            if pd.api.types.is_numeric_dtype(meta.index):
                try:
                    sample_id_int = int(sample_id)
                    if sample_id_int in meta.index:
                        # Found it as int
                         idx = meta.index.get_loc(sample_id_int)
                         if isinstance(idx, slice) or isinstance(idx, np.ndarray):
                            idx = idx.start if isinstance(idx, slice) else idx[0]
                         return self.test_data['X'][idx]
                except ValueError:
                    pass
            
            # If still not found
            print(f"Warning: Sample ID '{sample_id}' not found in metadata.")
            return None
            
        # Find integer index
        idx = meta.index.get_loc(sample_id)
        # Handle duplicate indices if any (though typically SampleID is unique)
        if isinstance(idx, slice) or isinstance(idx, np.ndarray):
            idx = idx.start if isinstance(idx, slice) else idx[0]
            
        return self.test_data['X'][idx]

    def generate_synthetic_cohort(self, n_samples: int = 50, scenario: str = "Mixed") -> pd.DataFrame:
        """
        Generate a synthetic cohort of patients using the VAE decoder.
        
        Args:
            n_samples: Number of synthetic patients to generate.
            scenario: 'Smoker' (High MMAD), 'Diesel' (Low MMAD), or 'Mixed'.
            
        Returns:
            DataFrame with metadata (z_phys parameters) and gene expression.
        """
        if self.model is None:
            raise ValueError("Model not loaded.")
            
        # 1. Sample Physics Latents (z_phys) based on Scenario
        # Dimensions: [MMAD, GSD, Conc, Time, Rate]
        # We focus mainly on MMAD (dim 0) for the visual effect
        
        z_phys = torch.zeros((n_samples, 5)).to(self.device)
        
        if scenario == "Smoker":
            # Coarser particles (Tobacco smoke ~0.3 - 0.5 um, but let's exaggerate for viz)
            # Log-Norm space? The model z_phys is mostly normalized. 
            # Assuming latent space is roughly N(0,1), but physics params are specific.
            # Let's target the interpretable range derived from training.
            # For demo, we sweep MMAD in [0.3, 0.8] range
            z_phys[:, 0] = torch.normal(mean=0.5, std=0.2, size=(n_samples,))
            
        elif scenario == "Diesel Exhaust":
            # Ultrafine (<0.1 um)
            z_phys[:, 0] = torch.normal(mean=-1.0, std=0.2, size=(n_samples,))
            
        else: # Mixed
            # Multimodal distribution
            n1 = n_samples // 2
            n2 = n_samples - n1
            z_phys[:n1, 0] = torch.normal(mean=0.5, std=0.3, size=(n1,)) # Coarse
            z_phys[n1:, 0] = torch.normal(mean=-1.0, std=0.3, size=(n2,)) # Fine
            
        # Randomize other params (GSD, Conc, Time, Rate)
        z_phys[:, 1:] = torch.randn((n_samples, 4)) 
        
        # 2. Sample Biological Latents (z_bio) ~ N(0, 1)
        z_bio = torch.randn((n_samples, 3)).to(self.device)
        
        # 3. Decode -> Synthetic Expression
        z = torch.cat([z_phys, z_bio], dim=1)
        
        with torch.no_grad():
            x_recon = self.model.decoder(z)
            # Re-scale from log-norm if needed, but we usually stay in log-space for heatmap
            x_recon_np = x_recon.cpu().numpy()
            
        # 4. Construct DataFrame
        # Genes columns
        df = pd.DataFrame(x_recon_np, columns=self.gene_names)
        
        # Add Metadata columns
        df['Scenario'] = scenario
        if scenario == "Mixed":
            # Infer label based on MMAD latent
            df['Scenario'] = ['Smoker-Like' if z > -0.2 else 'Diesel-Like' for z in z_phys[:, 0].cpu().numpy()]
            
        df['z_mmad'] = z_phys[:, 0].cpu().numpy()
        df['SampleID'] = [f"Syn_{i:03d}" for i in range(n_samples)]
        
        return df

    def run_validation_study(self, n_samples: int = 100) -> Dict[str, Any]:
        """
        Run a full validation study:
        1. Generate synthetic ground truth (z_true).
        2. Decode to get synthetic expression (x_syn).
        3. Re-encode to get inferred parameters (z_pred).
        4. Calculate accuracy metrics.
        """
        if self.model is None:
            raise ValueError("Model not loaded.")
            
        # 1. Generate Ground Truth (Covering full range)
        z_phys_true = torch.zeros((n_samples, 5)).to(self.device)
        # Scan MMAD from -2 to +2 (broad range)
        z_phys_true[:, 0] = torch.linspace(-2, 2, n_samples).to(self.device)
        # Randomize others
        z_phys_true[:, 1:] = torch.randn((n_samples, 4)).to(self.device)
        
        z_bio_true = torch.randn((n_samples, 3)).to(self.device)
        z_true = torch.cat([z_phys_true, z_bio_true], dim=1)
        
        with torch.no_grad():
            # 2. Decode -> Synthetic Expression
            x_syn = self.model.decode_biology(z_true)
            
            # 3. Re-Encode -> Inferred Latents
            mu, _ = self.model.encode(x_syn)
            
            # 4. Physics Decoder (for explicit parameter recovery if used)
            # pure z_phys from encoder
            z_phys_pred = mu[:, :5]
            
        # Metrics
        z_true_np = z_phys_true[:, 0].cpu().numpy()
        z_pred_np = z_phys_pred[:, 0].cpu().numpy()
        
        from sklearn.metrics import r2_score, mean_squared_error
        
        r2 = r2_score(z_true_np, z_pred_np)
        rmse = np.sqrt(mean_squared_error(z_true_np, z_pred_np))
        
        df = pd.DataFrame({
            'True_MMAD_Latent': z_true_np,
            'Pred_MMAD_Latent': z_pred_np,
            'Error': np.abs(z_true_np - z_pred_np)
        })
        
        return {
            'metrics': {'R2': r2, 'RMSE': rmse},
            'data': df,
            'x_syn': x_syn.cpu().numpy() # For deconvolution validation
        }

    def process_sample(self, expression_data: np.ndarray, sample_name: str = "Sample") -> Dict[str, Any]:
        """
        Run full analysis pipeline on a single sample.
        """
        if len(expression_data) != len(self.gene_names):
             # Try to align if names provided? For now assume pre-aligned
             pass
             
        inputs = torch.tensor(expression_data, dtype=torch.float32).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            mu, _ = self.model.encode(inputs)
            
            # 1. Physics Inference
            phys_out = self.model.decode_physics(mu, return_params=True)
            mmad = phys_out['physics_params']['MMAD'].item()
            
            # Deposition Fractions
            raw_tb = phys_out['F_TB'].item()
            raw_alv = phys_out['F_ALV'].item()
            
            # Head Deposition (Sum of ET1+ET2 if available)
            raw_head = 0.0
            if 'regional_df' in phys_out:
                 raw_head = (phys_out['regional_df'][:, 0] + phys_out['regional_df'][:, 1]).item()
            
            # Normalize to 1.0 sum (Conservation of Mass)
            total_sum = raw_tb + raw_alv + raw_head
            scale = 1.0 if total_sum <= 1.0 else (1.0 / total_sum)
            
            f_tb = raw_tb * scale
            f_alv = raw_alv * scale
            f_head = raw_head * scale
            f_exhaled = max(0.0, 1.0 - (f_tb + f_alv + f_head))
            
            # 2. Biological Deconvolution
            cell_props = self.deconv.deconvolve(
                expression_data.reshape(1, -1),
                self.gene_names,
                method='spatialddls'
            )
            
            # Unwrap results
            probs = {}
            if 'proportions' in cell_props and 'cell_types' in cell_props:
                prop_mat = cell_props['proportions']
                c_types = cell_props['cell_types']
                if len(prop_mat) > 0:
                    row_probs = prop_mat[0]
                    for c_name, c_val in zip(c_types, row_probs):
                        probs[c_name] = float(c_val)
            
            for k in ['bronchial_signal', 'alveolar_signal']:
                if k in cell_props:
                    val = cell_props[k]
                    if isinstance(val, (np.ndarray, list)) and len(val) > 0:
                        probs[k] = float(val[0])
                    else:
                        probs[k] = float(val)
            
            # 3. Interpretation
            physics_interp = self._interpret_mmad(mmad)
            
            # 4. Alignment
            alignment = self._check_alignment(f_tb, f_alv, probs)
            
            return {
                "sample_name": sample_name,
                "physics": {
                    "mmad": mmad,
                    "interpretation": physics_interp,
                    "deposition": {
                        "Head": f_head,
                        "Tracheobronchial": f_tb,
                        "Alveolar": f_alv,
                        "Exhaled": f_exhaled
                    }
                },
                "biology": {
                    "proportions": probs,
                    "dominant_tissue": alignment['dominant_tissue']
                },
                "alignment": alignment,
                "input_expression": expression_data.tolist() if hasattr(expression_data, 'tolist') else list(expression_data)
            }
            
    def _interpret_mmad(self, mmad: float) -> Dict[str, str]:
        if mmad < 0.1:
            size_class = "Ultrafine (Nanoparticles)"
            source = "Engine Exhaust / Virus"
        elif mmad < 1.0:
            size_class = "Fine (Accumulation Mode)"
            source = "Tobacco Smoke / Smog"
        elif mmad < 2.5:
            size_class = "Fine (Dust Mode)"
            source = "Bacteria / Fine Dust"
        else:
            size_class = "Coarse"
            source = "Pollen / Coarse Dust"
            
        return {"size_class": size_class, "source": source}
    
    def _check_alignment(self, f_tb, f_alv, cell_props) -> Dict[str, Any]:
        """Check consistency between Physics Dose and Biological Tissue."""
        bio_tb = cell_props.get('bronchial_signal', 0.0)
        bio_alv = cell_props.get('alveolar_signal', 0.0)
        
        lung_retained = f_tb + f_alv
        if lung_retained > 0:
            rel_tb = f_tb / lung_retained
            rel_alv = f_alv / lung_retained
        else:
            rel_tb = 0
            rel_alv = 0
            
        bio_dom = "Airway" if bio_tb > bio_alv else "Alveolar"
        phys_dom = "Airway" if rel_tb > rel_alv else "Alveolar"
        
        consistent = (bio_dom == phys_dom)
        
        total_bio = bio_tb + bio_alv
        if total_bio > 0:
            norm_bio_tb = bio_tb / total_bio
            norm_bio_alv = bio_alv / total_bio
        else:
            norm_bio_tb = 0.5
            norm_bio_alv = 0.5
            
        diff = (abs(rel_tb - norm_bio_tb) + abs(rel_alv - norm_bio_alv)) / 2.0
        score = max(0.0, 1.0 - diff)

        return {
            "consistent": consistent,
            "overall_score": score,
            "dominant_tissue": bio_dom,
            "bio_scores": {"Airway": bio_tb, "Alveolar": bio_alv},
            "phys_scores": {"Airway": rel_tb, "Alveolar": rel_alv},
            "interpretation": f"Biology sees {bio_dom} tissue, Physics sees {phys_dom} dose."
        }
