
"""
PulmoTrace Visualization Script
-------------------------------
Generates publication-quality plots to demonstrate Phase 1 results.

Plots:
1. PCA: Dataset Integration (Batch Effect Removal)
2. Volcano Plot: Differential Expression (Smoker vs Non-Smoker)
3. Heatmap: Physics-Informed Gene Signature
4. Physics Manifold: Inferred MMAD Analysis
"""

import sys
import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# Add parent directory to path
sys.path.insert(0, os.path.abspath('..'))

from data import prepare_geo_training_data
from models.pivae import PIVAE
from models.mppd_surrogate import create_trained_surrogate

# Configure Plot Style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_context("paper", font_scale=1.5)
PALETTE = sns.color_palette("viridis")

def load_data_and_model():
    print("Loading data...")
    dataloader, gene_names, metadata = prepare_geo_training_data(
        data_dir='raw',
        use_physics_features=True,
        apply_batch_correction=True,
        batch_size=32
    )
    dataset = dataloader.dataset
    
    # Load Model
    print("Loading PI-VAE model...")
    icrp_surrogate = create_trained_surrogate(n_epochs=100)
    checkpoint = torch.load('checkpoints/best_model.pt', map_location='cpu')
    model = PIVAE(
        input_dim=len(gene_names),
        latent_dim=8,
        z_phys_dim=5,
        encoder_hidden=[256, 128, 64],
        decoder_hidden=[64, 128, 256],
        surrogate=icrp_surrogate
    )
    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    model.eval()
    
    # Normalize Condition Column
    # Coalesce 'Smoking Status' and 'Smoking status'
    cond_col = 'Smoking Status' if 'Smoking Status' in metadata.columns else 'Smoking status'
    if cond_col in metadata.columns:
        metadata['condition'] = metadata[cond_col].fillna('Unknown')
    else:
        metadata['condition'] = 'Unknown'
        
    # Simplify labels for plotting
    metadata['condition'] = metadata['condition'].astype(str).str.lower()
    
    # Priority-based assignment (Order matters!)
    # 1. Healthy / Non-Smokers
    mask_healthy = metadata['condition'].str.contains('non-smoker') | \
                   metadata['condition'].str.contains('never smoker') | \
                   metadata['condition'].str.contains('healthy') | \
                   metadata['condition'].str.contains('control')
    
    # 2. Former Smokers
    mask_former = metadata['condition'].str.contains('former')
    
    # 3. COPD
    mask_copd = metadata['condition'].str.contains('copd')
    
    # 4. Active Smokers (Must contain 'smoker' but NOT 'non' or 'former')
    mask_smoker = metadata['condition'].str.contains('smoker') & \
                  ~mask_healthy & \
                  ~mask_former
                  
    # Apply simplified labels
    metadata.loc[mask_healthy, 'Group'] = 'Healthy'
    metadata.loc[mask_former, 'Group'] = 'Former Smoker'
    metadata.loc[mask_copd, 'Group'] = 'COPD'
    metadata.loc[mask_smoker, 'Group'] = 'Smoker'
    metadata['Group'] = metadata['Group'].fillna('Other')
    
    # Print distribution
    print("\nGroup Distribution:")
    print(metadata['Group'].value_counts())
    
    return dataset, metadata, model, gene_names

def plot_pca_integration(dataset, metadata):
    print("Generating PCA Plot (Dataset Integration)...")
    expr = dataset.expression.numpy()
    pca = PCA(n_components=2)
    embedding = pca.fit_transform(expr)
    
    df_pca = pd.DataFrame(embedding, columns=['PC1', 'PC2'])
    df_pca['Dataset'] = metadata['geo_id'].values
    df_pca['Group'] = metadata['Group'].values 
    
    plt.figure(figsize=(14, 6))
    
    # Subplot 1: Color by Dataset
    plt.subplot(1, 2, 1)
    sns.scatterplot(data=df_pca, x='PC1', y='PC2', hue='Dataset', alpha=0.7, palette='tab10', s=60)
    plt.title("Integration Check (Batch Effect)")
    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)
    
    # Subplot 2: Color by Group
    plt.subplot(1, 2, 2)
    # Filter out 'Other' for cleaner plot if desired, or keep to show unknowns
    sns.scatterplot(data=df_pca, x='PC1', y='PC2', hue='Group', alpha=0.8, palette='viridis', s=60)
    plt.title("Biological Signal (Condition)")
    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)
    
    plt.tight_layout()
    plt.savefig('pca_integration.png', dpi=300, bbox_inches='tight')
    print("Saved pca_integration.png")

def plot_heatmap(dataset, gene_names, metadata):
    print("Generating Gene Expression Heatmap...")
    expr = dataset.expression.numpy()
    
    # Filter for relevant groups
    mask = metadata['Group'] != 'Other'
    
    if mask.sum() == 0:
        mask = pd.Series([True] * len(metadata))
    
    subset_expr = expr[mask]
    subset_meta = metadata[mask]
    
    # Create DataFrame for Heatmap
    df_heatmap = pd.DataFrame(subset_expr, columns=gene_names)
    df_heatmap['Group'] = subset_meta['Group'].values
    
    # Aggregate by Group
    df_avg = df_heatmap.groupby('Group').mean()
    
    plt.figure(figsize=(10, 12))
    sns.heatmap(df_avg.T, cmap='RdBu_r', center=0, annot=False, cbar_kws={'label': 'Log-Normalized Expression'})
    plt.title("Physics-Informed Gene Signature")
    plt.xlabel("Biological Group")
    plt.ylabel("Gene Name")
    plt.tight_layout()
    plt.savefig('gene_heatmap.png', dpi=300, bbox_inches='tight')
    print("Saved gene_heatmap.png")

def plot_physics_manifold(dataset, model, metadata):
    print("Generating Physics Manifold Plot...")
    inputs = dataset.expression
    
    with torch.no_grad():
        mu, _ = model.encode(inputs)
        physics_output = model.decode_physics(mu, return_params=True)
        mmad = physics_output['physics_params']['MMAD'].numpy()
        
    df_phys = pd.DataFrame({'MMAD': mmad})
    df_phys['Group'] = metadata['Group'].values
    
    # Filter out 'Other' for clarity
    df_phys = df_phys[df_phys['Group'] != 'Other']
    
    plt.figure(figsize=(10, 6))
    
    # Use Kernel Density Estimate
    sns.kdeplot(data=df_phys, x='MMAD', hue='Group', fill=True, common_norm=False, palette='turbo', linewidth=2)
    
    plt.title("Inferred Particle Size (MMAD) Distribution")
    plt.xlabel("Aerodynamic Diameter (µm)")
    plt.ylabel("Density")
    plt.xlim(0, 3.0) # Focus on fine/accumulation mode
    
    # Add annotations
    plt.axvline(x=0.4, color='gray', linestyle='--', alpha=0.5)
    plt.text(0.45, plt.ylim()[1]*0.8, 'Tobacco Smoke\n(~0.4-0.6 µm)', color='gray')
    
    plt.tight_layout()
    plt.savefig('physics_manifold.png', dpi=300, bbox_inches='tight')
    print("Saved physics_manifold.png")

def main():
    dataset, metadata, model, gene_names = load_data_and_model()
    
    plot_pca_integration(dataset, metadata)
    plot_heatmap(dataset, gene_names, metadata)
    plot_physics_manifold(dataset, model, metadata)
    
    print("\nVisualization Complete! Check the .png files.")

if __name__ == "__main__":
    main()
