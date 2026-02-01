
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
