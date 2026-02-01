"""
PI-VAE Evaluation Script

Generates performance metrics and visualizations for the trained PI-VAE model.
1. Loss Curves (Train vs Val)
2. Gene Reconstruction Accuracy (Parity Plot)
3. Latent Space Visualization (PCA of z_bio and z_phys)
4. Inferred Physical Parameters Distributions
5. Physics-Biology Correlation Analysis

Outputs saved to 'results/' directory.
"""

import sys
import os
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
from pathlib import Path
import json

# Add parent directory to path
sys.path.insert(0, os.path.abspath('..'))

from data import prepare_geo_training_data
from models.pivae import PIVAE
from train_pivae import manual_stratified_split

# Plotting style
plt.style.use('bmh')

def manual_r2_score(y_true, y_pred):
    """Manual R2 score calculation."""
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - (ss_res / ss_tot)

class ManualPCA:
    """Manual PCA implementation using SVD."""
    def __init__(self, n_components=2):
        self.n_components = n_components
        self.components_ = None
        self.mean_ = None
        
    def fit_transform(self, X):
        X = np.array(X)
        self.mean_ = np.mean(X, axis=0)
        X_centered = X - self.mean_
        
        # SVD
        U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)
        
        # Project
        X_reduced = U[:, :self.n_components] * S[:self.n_components]
        return X_reduced

def load_model_and_data(checkpoint_path='checkpoints/best_model.pt'):
    """Load trained model and test data."""
    
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
    
    # Get test data
    test_expr = dataset.expression[test_idx]
    test_meta = metadata.iloc[test_idx]
    
    # 2. Load Model
    print(f"Loading model from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model_state = checkpoint['model_state_dict']
    history = checkpoint['history']
    
    # Initialize model
    model = PIVAE(
        input_dim=len(gene_names),
        latent_dim=8,
        z_phys_dim=5,
        encoder_hidden=[256, 128, 64],
        decoder_hidden=[64, 128, 256]
    )
    model.load_state_dict(model_state)
    model.eval()
    
    return model, test_expr, test_meta, history, gene_names

def plot_loss_curves(history, save_dir):
    """Plot training and validation loss curves."""
    epochs = range(1, len(history['train_loss']) + 1)
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Total Loss
    axes[0].plot(epochs, history['train_loss'], label='Train')
    axes[0].plot(epochs, history['val_loss'], label='Validation')
    axes[0].set_title('Total Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    
    # Reconstruction Loss
    axes[1].plot(epochs, history['train_recon'], label='Train')
    axes[1].plot(epochs, history['val_recon'], label='Validation')
    axes[1].set_title('Reconstruction Loss (MSE)')
    axes[1].set_xlabel('Epoch')
    
    # Physics/ICRP Loss
    axes[2].plot(epochs, history['train_icrp'], label='ICRP Consistency')
    axes[2].plot(epochs, history['train_physics'], label='Bio-Physics Constraint')
    axes[2].set_title('Physics Losses')
    axes[2].set_xlabel('Epoch')
    axes[2].legend()
    
    plt.tight_layout()
    plt.savefig(save_dir / 'loss_curves.png', dpi=300)
    plt.close()

def evaluate_reconstruction(model, test_expr, gene_names, save_dir):
    """Evaluate gene reconstruction accuracy."""
    with torch.no_grad():
        outputs = model(test_expr)
        recon = outputs['x_recon'].numpy()
        truth = test_expr.numpy()
    
    # Global metrics
    mse = np.mean((truth - recon)**2)
    r2 = manual_r2_score(truth.flatten(), recon.flatten())
    
    print(f"Test MSE: {mse:.4f}")
    print(f"Test R²: {r2:.4f}")
    
    # Parity Plot (Subset of points for clarity if needed, but 60 samples * 45 genes is small enough)
    plt.figure(figsize=(8, 8))
    plt.scatter(truth.flatten(), recon.flatten(), alpha=0.3, s=10, c='blue')
    
    min_val, max_val = truth.min(), truth.max()
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
    
    plt.title(f'Reconstruction Accuracy (R²={r2:.3f})')
    plt.xlabel('True Normalized Expression')
    plt.ylabel('Reconstructed Expression')
    plt.savefig(save_dir / 'reconstruction_parity.png', dpi=300)
    plt.close()
    
    return outputs

def analyze_latent_space(outputs, metadata, save_dir):
    """Visualize latent space with PCA."""
    z_phys = outputs['z_phys'].numpy()
    z_bio = outputs['z_bio'].numpy()
    
    # Combine just for checking total variance, but we visualize separately
    
    # 1. z_phys PCA
    if z_phys.shape[1] > 1:
        pca_phys = ManualPCA(n_components=2)
        z_phys_pca = pca_phys.fit_transform(z_phys)
        
        plt.figure(figsize=(10, 8))
        # Map categories to colors
        unique_ids = np.unique(metadata['geo_id'].values)
        colors = plt.cm.tab10(np.linspace(0, 1, len(unique_ids)))
        
        for i, uid in enumerate(unique_ids):
            mask = metadata['geo_id'].values == uid
            plt.scatter(
                z_phys_pca[mask, 0], z_phys_pca[mask, 1],
                label=uid, color=colors[i], s=100, alpha=0.8
            )
        
        plt.legend(title='Dataset')
        plt.title('Latent Physics Space (z_phys) PCA')
        plt.savefig(save_dir / 'latent_space_phys.png', dpi=300)
        plt.close()
    
    # 2. z_bio PCA
    if z_bio.shape[1] > 1:
        pca_bio = ManualPCA(n_components=min(2, z_bio.shape[1]))
        z_bio_pca = pca_bio.fit_transform(z_bio)
        
        plt.figure(figsize=(10, 8))
        for i, uid in enumerate(unique_ids):
            mask = metadata['geo_id'].values == uid
            plt.scatter(
                z_bio_pca[mask, 0], z_bio_pca[mask, 1],
                label=uid, color=colors[i], s=100, alpha=0.8
            )
            
        plt.legend(title='Dataset')
        plt.title('Latent Biological Space (z_bio) PCA')
        plt.savefig(save_dir / 'latent_space_bio.png', dpi=300)
        plt.close()

def analyze_physics_predictions(outputs, metadata, save_dir):
    """Analyze inferred physical parameters."""
    params = outputs['physics_params'] # Dict of lists/tensors
    
    # Convert to DataFrame
    df_params = pd.DataFrame({k: v.numpy() if hasattr(v, 'numpy') else v for k, v in params.items()})
    df_params['Dataset'] = metadata['geo_id'].values
    
    # Distribution of inferred MMAD
    plt.figure(figsize=(10, 6))
    
    unique_ids = df_params['Dataset'].unique()
    for uid in unique_ids:
        subset = df_params[df_params['Dataset'] == uid]
        plt.hist(subset['MMAD'], bins=20, alpha=0.5, label=uid, density=True)
        
    plt.title('Inferred Particle Size (MMAD) Distribution')
    plt.xlabel('MMAD (μm)')
    plt.legend(title='Dataset')
    plt.savefig(save_dir / 'inferred_mmad_dist.png', dpi=300)
    plt.close()
    
    # Deposition Predictions
    ftb = outputs['F_TB'].numpy()
    falv = outputs['F_ALV'].numpy()
    
    plt.figure(figsize=(10, 6))
    # Map dataset to colors
    geo_ids = metadata['geo_id'].astype('category').cat.codes
    scatter = plt.scatter(ftb, falv, c=geo_ids, cmap='viridis', s=50, alpha=0.7)
    plt.xlabel('Tracheobronchial Deposition Fraction')
    plt.ylabel('Alveolar Deposition Fraction')
    plt.title('Predicted Regional Deposition')
    plt.colorbar(scatter, label='Dataset Group')
    plt.savefig(save_dir / 'deposition_predictions.png', dpi=300)
    plt.close()

def main():
    save_dir = Path('results')
    save_dir.mkdir(exist_ok=True)
    
    # 1. Load
    model, test_expr, test_meta, history, gene_names = load_model_and_data()
    
    # 2. Loss Curves
    print("Generating loss curves...")
    plot_loss_curves(history, save_dir)
    
    # 3. Reconstruction Accuracy
    print("Evaluating reconstruction...")
    outputs = evaluate_reconstruction(model, test_expr, gene_names, save_dir)
    
    # 4. Latent Space
    print("Analyzing latent space...")
    analyze_latent_space(outputs, test_meta, save_dir)
    
    # 5. Physics Predictions
    print("Analyzing physics predictions...")
    analyze_physics_predictions(outputs, test_meta, save_dir)
    
    print("\nEvaluation Complete!")
    print(f"Results saved in {save_dir.absolute()}")

if __name__ == "__main__":
    main()
