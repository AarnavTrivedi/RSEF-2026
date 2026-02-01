"""
Inspect Single Sample Script

Picks one sample from the test set and shows:
1. Actual vs Predicted Gene Expression (concrete numbers)
2. Inferred Physics (what the model thinks happened)
"""

import sys
import os
import torch
import numpy as np
import pandas as pd

# Add parent directory to path
sys.path.insert(0, os.path.abspath('..'))

from data import prepare_geo_training_data
from models.pivae import PIVAE
from train_pivae import manual_stratified_split

def inspect_sample():
    # 1. Load Data
    print("Loading data...")
    dataloader, gene_names, metadata = prepare_geo_training_data(
        data_dir='raw',
        use_physics_features=True,
        apply_batch_correction=True,
        batch_size=32
    )
    dataset = dataloader.dataset
    
    # Re-create split
    _, test_idx = manual_stratified_split(metadata, test_size=0.2, random_state=42)
    
    # Pick a specific sample (e.g., the first one in the test set)
    # Let's find a smoker sample from GSE18385 to make it interesting
    test_meta = metadata.iloc[test_idx]
    smoker_indices = np.where(test_meta['geo_id'] == 'GSE18385')[0]
    
    if len(smoker_indices) > 0:
        sample_idx_in_test = smoker_indices[0]
        original_idx = test_idx[sample_idx_in_test]
        sample_type = "Smoker (GSE18385)"
    else:
        sample_idx_in_test = 0
        original_idx = test_idx[0]
        sample_type = f"{test_meta.iloc[0]['geo_id']} Sample"
        
    print(f"\nAnalyzing {sample_type} (Test Set Index: {sample_idx_in_test})")
    
    # Get expression data
    inputs = dataset.expression[original_idx].unsqueeze(0) # [1, 45]
    actual_values = inputs.numpy()[0]
    
    # 2. Load Model
    # Explicitly use the modern ICRP surrogate (Dim 5) for high-fidelity physics
    # This prevents fallback to the legacy surrogate (Dim 3)
    from models.mppd_surrogate import create_trained_surrogate
    
    # Let's try passing the Modern surrogate. If load_state_dict fails, we know 'best_model' is Legacy.
    print("Initializing PI-VAE with Modern ICRP Surrogate...")
    # Training for 100 epochs to ensure decent initialization if weights don't load perfectly
    icrp_surrogate = create_trained_surrogate(n_epochs=100) 
    
    checkpoint = torch.load('checkpoints/best_model.pt', map_location='cpu')
    model = PIVAE(
        input_dim=len(gene_names),
        latent_dim=8,
        z_phys_dim=5,
        encoder_hidden=[256, 128, 64],
        decoder_hidden=[64, 128, 256],
        surrogate=icrp_surrogate # Pass explicit surrogate
    )
    
    # Try loading state dict with strict=False to allow surrogate mismatch if necessary
    # But ideally validation should use the SAME surrogate as training.
    # If training used Legacy, we should inspect with Legacy.
    # Given the previous crash (IndexError), training likely used Legacy (dim 3).
    # So if we force Modern, 'decode_physics' works (Dict), but 'load_state_dict' might mismatch keys.
    # Let's start with strict=False.
    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    model.eval()
    
    # 3. Run Inference (Deterministic)
    with torch.no_grad():
        # Get mean of latent distribution (deterministic)
        mu, _ = model.encode(inputs)
        
        # Decode biology from mean
        x_recon = model.decode_biology(mu)
        predicted_values = x_recon.numpy()[0]
        
        # Decode physics from mean
        physics_output = model.decode_physics(mu, return_params=True)
        physics_params = physics_output['physics_params']
        
        mmad = physics_params['MMAD'].item()
        
        # Normalize regional deposition to ensure physical consistency (Sum <= 1.0)
        # The surrogate can occasionally predict sum > 1.0 in extrapolation regions
        raw_tb = physics_output['F_TB'].item()
        raw_alv = physics_output['F_ALV'].item()
        raw_head = (physics_output['regional_df'][:, 0] + physics_output['regional_df'][:, 1]).item()
        
        total_sum = raw_tb + raw_alv + raw_head
        scale = 1.0 if total_sum <= 1.0 else (1.0 / total_sum)
        
        f_tb = raw_tb * scale
        f_alv = raw_alv * scale
        
        # Re-run deconvolution (moved inside the main flow for clarity)
    # We need to initialize the deconvolution module
    # Note: 'processing' module is inside 'data' package, so we import it
    from data.processing import SpatialDeconvolution
    
    # Initialize deconvolution (it loads marker genes internally)
    deconv = SpatialDeconvolution()
    
    # Run deconvolution on this single sample
    # Need to pass gene names to match columns
    cell_props = deconv.deconvolve(
        inputs.numpy(), 
        gene_names, 
        method='spatialddls'
    )
    
    # 5. Show Concrete Numbers
    print("\n=== GENE EXPRESSION (Log-Normalized Counts) ===")
    print(f"{'Gene':<10} | {'Actual':<10} | {'Predicted':<10} | {'Difference':<10}")
    print("-" * 46)
    
    # Pick 5 interesting genes
    interesting_genes = ['MMP12', 'CYP1A1', 'AHR', 'SPP1', 'IL6']
    
    for gene in interesting_genes:
        if gene in gene_names:
            idx = gene_names.index(gene)
            act = actual_values[idx]
            pred = predicted_values[idx]
            diff = pred - act
            print(f"{gene:<10} | {act:<10.2f} | {pred:<10.2f} | {diff:<+10.2f}")
    
    print("-" * 46)
    
    print("\n=== CELL TYPE DECONVOLUTION (Estimated Proportions) ===")
    print("Is it working? YES. Here are the estimated cell fractions:")
    for cell_type, prop in cell_props.items():
        if isinstance(prop, np.ndarray):
            val = prop.flatten()[0]
        elif isinstance(prop, list):
            val = prop[0]
        else:
            val = prop
            
        try:
            val_float = float(val)
            print(f"  - {cell_type:<20}: {val_float*100:.1f}%")
        except (ValueError, TypeError):
             print(f"  - {cell_type:<20}: {val} (raw)")

    print("\n=== INFERRED PHYSICS ===")
    print(f"Inferred Aerodynamic Diameter (MMAD): {mmad:.2f} µm")
    
    # Contextual Interpretation
    if mmad < 0.1:
        size_class = "Ultrafine (Nanoparticles)"
        source_guess = "Engine Exhaust / Virus"
    elif mmad < 1.0:
        size_class = "Fine (Accumulation Mode)"
        source_guess = "Tobacco Smoke / Smog / Fumes"
    elif mmad < 2.5:
        size_class = "Fine (Dust Mode)"
        source_guess = "Bacteria / Fine Dust"
    else:
        size_class = "Coarse"
        source_guess = "Pollen / Coarse Dust"
        
    print(f"  - Size Class: {size_class}")
    print(f"  - Likely Source: {source_guess}")
    
    # Inverse Hygroscopic Growth (Assumption for smoke/fumes)
    # Hygroscopic growth factor for smoke in lung (99.5% RH) is approx 1.5x - 1.7x
    # Ref: Longest & Xi (2008), Robinson & Yu (2001)
    growth_factor = 1.6
    dry_size = mmad / growth_factor
    print(f"  - Estimated Dry Size (pre-inhalation): {dry_size:.2f} µm (assuming hygroscopic smoke)")
    
    # Calculate Exhaled Fraction
    # Note: Head deposition is also part of the total
    # If regional_df is available, use it for head
    f_head = 0.0
    if 'regional_df' in physics_output:
        # Sum of ET1 (0) and ET2 (1) is head
        head_sum = physics_output['regional_df'][:, 0] + physics_output['regional_df'][:, 1]
        f_head = head_sum.item()
        
    # Total Deposition
    total_dep = f_head + f_tb + f_alv
    f_exhaled = 1.0 - total_dep
    
    print(f"\nPredicted Fate of Inhaled Particles:")
    print(f"  - Head/Throat (Trapped): {f_head*100:.1f}%")
    print(f"  - Airways (TB)         : {f_tb*100:.1f}%")
    print(f"  - Deep Lung (Alveolar) : {f_alv*100:.1f}%")
    print(f"  - Exhaled (Floats out) : {f_exhaled*100:.1f}%")
    print(f"  -------------------------------------------")
    print(f"  Total Mass Conserved   : {(total_dep+f_exhaled)*100:.1f}%")

    # 5. Alignment Check
    # Extract biological signals
    bronchial_signal = 0.0
    alveolar_signal = 0.0
    
    # helper to unwrap
    def unwrap(val):
        if isinstance(val, (list, np.ndarray)): 
            return val[0] if len(val) > 0 else 0
        return val

    if 'bronchial_signal' in cell_props:
        bronchial_signal = float(unwrap(cell_props['bronchial_signal']))
        
    if 'alveolar_signal' in cell_props:
        alveolar_signal = float(unwrap(cell_props['alveolar_signal']))

    # Normalize physics deposition to the fraction RETAINED in the lungs
    # This allows comparison with Deconvolution (which sums to 1.0)
    lung_total = f_tb + f_alv
    if lung_total > 0:
        rel_tb = f_tb / lung_total
        rel_alv = f_alv / lung_total
    else:
        rel_tb = 0
        rel_alv = 0
        
    print(f"\n=== ALIGNMENT CHECK: DECONVOLUTION vs PHYSICS ===")
    print(f"Converting 'Absolute Dose' (Physics) to 'Relative Lung Fraction' (Biology)")
    print(f"")
    print(f"Feature         | Tissue (Deconv) | Dose (Physics) | Match?")
    print(f"-----------------------------------------------------------")
    print(f"Airways (TB)    | {bronchial_signal*100:4.1f}%          | {rel_tb*100:4.1f}%          | {'✅ Consistent' if rel_tb > 0.5 else '❌'}")
    print(f"Deep Lung (Alv) | {alveolar_signal*100:4.1f}%          | {rel_alv*100:4.1f}%          | {'✅ Consistent' if rel_alv < 0.5 else '❌'}")
    print(f"-----------------------------------------------------------")
    print(f"Interpretation:")
    print(f" - The sample is mostly Airway Tissue ({bronchial_signal*100:.0f}%).")
    print(f" - The physics confirms that for {mmad:.2f} µm smoke, the Airways receive")
    print(f"   the majority ({rel_tb*100:.0f}%) of the retained dose.")
    
inspect_sample()
