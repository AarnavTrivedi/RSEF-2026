"""
Physics Validation Script
-------------------------
Compares PI-VAE inferred physical parameters against ground truth metadata
defined in data.features.VALIDATED_DATASETS.

Outputs:
1. Quantitative Error Metrics (MAE, RMSE, Bias)
2. Scatter Plots (True vs Predicted)
3. Summary Report (Markdown)
"""

import sys
import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Add parent directory to path
sys.path.insert(0, os.path.abspath('..'))

from data import prepare_geo_training_data
from models.pivae import PIVAE
from train_pivae import manual_stratified_split
from data.features import get_dataset_info, VALIDATED_DATASETS

# Style
plt.style.use('bmh')

def validate_physics(checkpoint_path='checkpoints/best_model.pt', output_dir='results'):
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # 1. Load Data
    print("Loading data...")
    dataloader, gene_names, metadata = prepare_geo_training_data(
        data_dir='raw',
        use_physics_features=True,
        apply_batch_correction=True,
        batch_size=32
    )
    dataset = dataloader.dataset
    
    # Re-create split (Validation on Test Set)
    _, test_idx = manual_stratified_split(metadata, test_size=0.2, random_state=42)
    
    test_expr = dataset.expression[test_idx]
    test_meta = metadata.iloc[test_idx].copy()
    
    # 2. Load Model
    print(f"Loading model from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    model = PIVAE(
        input_dim=len(gene_names),
        latent_dim=8,
        z_phys_dim=5,
        encoder_hidden=[256, 128, 64],
        decoder_hidden=[64, 128, 256]
    )
    # Allow strict=False in case of surrogate buffer mismatch (Legacy vs Modern)
    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    model.eval()
    
    # 3. Get Ground Truth for Test Samples
    print("Extracting ground truth...")
    truth_records = []
    
    for _, row in test_meta.iterrows():
        geo_id = row['geo_id']
        try:
            info = get_dataset_info(geo_id)
            truth_records.append({
                'geo_id': geo_id,
                'True_MMAD': info.mmad_um,
                'True_Concentration': info.concentration,
                'True_GSD': info.gsd
                # Duration is harder to validate as it's often qualitative in metadata ("Chronic")
            })
        except ValueError:
            # Skip if dataset not in validated registry
            truth_records.append({
                'geo_id': geo_id,
                'True_MMAD': np.nan, 
                'True_Concentration': np.nan,
                'True_GSD': np.nan
            })
            
    df_truth = pd.DataFrame(truth_records)
    
    # 4. Run Inference
    print("Running inference...")
    with torch.no_grad():
        outputs = model(test_expr)
        params = outputs['physics_params']
        
    # Extract predictions
    df_pred = pd.DataFrame({
        'Pred_MMAD': params['MMAD'].numpy(),
        'Pred_Concentration': params['Concentration'].numpy(),
        'Pred_GSD': params['GSD'].numpy()
    })
    
    # Combine
    # df_truth has 'geo_id' which is also in test_meta, drop it from df_truth to avoid duplicates
    df_truth_clean = df_truth.drop(columns=['geo_id'], errors='ignore')
    results = pd.concat([test_meta.reset_index(drop=True), df_truth_clean, df_pred], axis=1)
    
    # Drop rows with NaN truth (if any)
    results = results.dropna(subset=['True_MMAD'])
    
    # USER REQUEST: Filter specifically for GSE25531 (Gold Standard)
    results = results[results['geo_id'] == 'GSE25531']
    if len(results) == 0:
        print("Warning: No GSE25531 samples found in test set!")
        # Fallback to all for debugging if needed, but for now strict filter.
    
    # 5. Calculate Metrics
    print("\n=== Validation Results ===")
    
    metrics = {}
    
    for param in ['MMAD', 'Concentration', 'GSD']:
        y_true = results[f'True_{param}']
        y_pred = results[f'Pred_{param}']
        
        # Log-scale metrics for Conc/MMAD? 
        # For now, stick to raw units but print carefully
        
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        
        # Mean Bias
        bias = np.mean(y_pred - y_true)
        
        # Percentage Error (MAPE) - handle 0 concentration
        mask = y_true > 0
        if mask.sum() > 0:
            mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
        else:
            mape = np.nan
            
        metrics[param] = {
            'MAE': mae,
            'RMSE': rmse,
            'Bias': bias,
            'MAPE': mape
        }
        
        print(f"\n{param}:")
        print(f"  MAE:  {mae:.4f}")
        print(f"  RMSE: {rmse:.4f}")
        print(f"  Bias: {bias:.4f}")
        print(f"  MAPE: {mape:.1f}%")
        
    # 6. Group-level comparison
    print("\n--- Dataset-Level Averages ---")
    group_stats = results.groupby('geo_id')[['True_MMAD', 'Pred_MMAD', 'True_Concentration', 'Pred_Concentration']].mean()
    print(group_stats)
    
    # 7. Plots
    plot_validation(results, output_dir)
    
    
    # 8. Save Report
    with open(output_dir / 'physics_validation_report_GSE25531.md', 'w') as f:
        f.write("# Physics Validation Report: GSE25531 (Gold Standard)\n\n")
        f.write("> **Dataset**: GSE25531 (Human Diesel Exhaust)\n")
        f.write("> **Condition**: 300 μg/m³, 0.3 μm MMAD\n\n")
        
        f.write("## Overall Metrics\n")
        f.write("| Parameter | MAE | RMSE | Bias | MAPE |\n")
        f.write("|---|---|---|---|---|\n")
        for param, m in metrics.items():
            f.write(f"| {param} | {m['MAE']:.3f} | {m['RMSE']:.3f} | {m['Bias']:.3f} | {m['MAPE']:.1f}% |\n")
            
        f.write("\n## Dataset-Level Comparison\n")
        f.write(group_stats.to_markdown())
        
        f.write("\n\n## Per-Sample Predictions\n")
        f.write(results[['geo_id', 'True_MMAD', 'Pred_MMAD', 'True_Concentration', 'Pred_Concentration']].to_markdown())
    
    print(f"\nReport saved to {output_dir / 'physics_validation_report_GSE25531.md'}")

def plot_validation(results, output_dir):
    """Generate True vs Predicted plots."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Map datasets to colors
    unique_ids = results['geo_id'].unique()
    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_ids)))
    color_map = dict(zip(unique_ids, colors))
    c = results['geo_id'].map(color_map)
    
    # MMAD
    axes[0].scatter(results['True_MMAD'], results['Pred_MMAD'], c=c, alpha=0.7, s=50)
    # Identity line
    min_val = min(results['True_MMAD'].min(), results['Pred_MMAD'].min())
    max_val = max(results['True_MMAD'].max(), results['Pred_MMAD'].max())
    axes[0].plot([min_val, max_val], [min_val, max_val], 'k--', lw=1)
    axes[0].set_xlabel('True MMAD (μm)')
    axes[0].set_ylabel('Predicted MMAD (μm)')
    axes[0].set_title('Particle Size (MMAD)')
    
    # Concentration (Log Scale Plot)
    axes[1].scatter(results['True_Concentration'], results['Pred_Concentration'], c=c, alpha=0.7, s=50)
    min_val = 0 # min(results['True_Concentration'].min(), results['Pred_Concentration'].min())
    max_val = max(results['True_Concentration'].max(), results['Pred_Concentration'].max())
    axes[1].plot([min_val, max_val], [min_val, max_val], 'k--', lw=1)
    axes[1].set_xlabel('True Concentration (μg/m³)')
    axes[1].set_ylabel('Predicted Concentration (μg/m³)')
    axes[1].set_title('Concentration')
    # Use log scale if range is huge
    if max_val > 1000:
        axes[1].set_xscale('log')
        axes[1].set_yscale('log')
    
    # GSD
    axes[2].scatter(results['True_GSD'], results['Pred_GSD'], c=c, alpha=0.7, s=50)
    min_val = min(results['True_GSD'].min(), results['Pred_GSD'].min())
    max_val = max(results['True_GSD'].max(), results['Pred_GSD'].max())
    axes[2].plot([min_val, max_val], [min_val, max_val], 'k--', lw=1)
    axes[2].set_xlabel('True GSD')
    axes[2].set_ylabel('Predicted GSD')
    axes[2].set_title('Polydispersity (GSD)')
    
    # Legend
    handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=col, label=uid, markersize=10) 
               for uid, col in color_map.items()]
    fig.legend(handles=handles, title='Dataset', loc='center right')
    
    plt.tight_layout()
    plt.subplots_adjust(right=0.9) # Make room for legend
    plt.savefig(output_dir / 'physics_validation_plots.png', dpi=300)
    plt.close()

if __name__ == "__main__":
    validate_physics()
