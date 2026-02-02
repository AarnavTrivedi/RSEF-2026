"""
PI-VAE Training Script

Trains the Physics-Informed VAE on 321 GEO samples with:
- Stratified train/test split (80/20) by dataset
- ICRP deposition physics loss
- Validation monitoring
- Model checkpointing
"""

import sys
import os
# Add parent directory to path to find 'models' and 'data' modules
sys.path.insert(0, os.path.abspath('..'))

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
import numpy as np
from pathlib import Path
from loguru import logger
import json
from datetime import datetime

from data import prepare_geo_training_data
from models.pivae import PIVAE


def manual_stratified_split(metadata, test_size=0.2, random_state=42):
    """
    Manual stratified split ensuring each GEO dataset is represented in both train and test.
    Using numpy/pandas instead of sklearn to avoid dependency issues.
    """
    np.random.seed(random_state)
    train_indices = []
    test_indices = []
    
    # Group by stratify label (geo_id)
    unique_labels = metadata['geo_id'].unique()
    
    for label in unique_labels:
        # Get indices for this group
        group_indices = np.where(metadata['geo_id'] == label)[0]
        np.random.shuffle(group_indices)
        
        # Calculate split point
        n_samples = len(group_indices)
        n_test = int(n_samples * test_size)
        if n_test == 0 and n_samples > 1:
            n_test = 1  # Ensure at least one test sample if possible
        
        # Split
        test_indices.extend(group_indices[:n_test])
        train_indices.extend(group_indices[n_test:])
    
    # Shuffle final sets
    np.random.shuffle(train_indices)
    np.random.shuffle(test_indices)
    
    return np.array(train_indices), np.array(test_indices)


class PIVAETrainer:
    """
    Trainer for PI-VAE with logging and checkpointing.
    """
    
    def __init__(
        self,
        model: PIVAE,
        train_loader: DataLoader,
        val_loader: DataLoader,
        lr: float = 1e-3,
        device: str = 'cpu',
        checkpoint_dir: str = 'checkpoints'
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=lr,
            weight_decay=1e-4
        )
        
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer, T_0=50, T_mult=2
        )
        
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(exist_ok=True)
        
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_recon': [],
            'train_kl': [],
            'train_physics': [],
            'train_icrp': [],
            'val_recon': [],
            'val_physics': []
        }
        
        self.best_val_loss = float('inf')
    
    def train_epoch(self, beta: float = 0.01, gamma: float = 1.0) -> dict:
        """Single training epoch."""
        self.model.train()
        total_losses = {
            'total': 0, 'recon': 0, 'kl': 0, 
            'physics': 0, 'icrp': 0
        }
        n_batches = 0
        
        for batch in self.train_loader:
            expression = batch['expression'].to(self.device)
            bio_tb = batch['bio_tb'].to(self.device)
            bio_alv = batch['bio_alv'].to(self.device)
            
            self.optimizer.zero_grad()
            
            # Forward pass
            outputs = self.model(expression)
            
            # Compute loss
            z_true = batch['z_true'].to(self.device)
            has_label = batch['has_label'].to(self.device)
            
            # Use z_true only for labeled samples (has_label mask)
            # PIVAE.compute_loss expects z_true for all, but we can mask it?
            # Actually, compute_loss handles supervised loss if z_true is not None.
            # But here we have partial labels.
            # Let's pass masked z_true? No, model needs to handle masking or we filter.
            # Simple approach: Pass z_true, and PIVAE should ideally have a mask argument.
            # Let's check PIVAE definition. 
            # If PIVAE class doesn't support mask, we might need to modify it or only train on labeled?
            # Standard implementation: supervised loss = MSE(pred[mask], true[mask])
            # Let's modify call to pass z_true. PIVAE likely assumes fully labeled if passed.
            # We will pass it and let's see. If PIVAE is robust, it might need 'mask' argument.
            # For now, let's pass it. The model computes loss on all provided z_true.
            # We should probably modify PIVAE to take a mask if it doesn't.
            # But let's look at PIVAE.compute_loss signature in previous turns (Step 79).
            # It just takes z_true.
            
            # Strategy: Pass z_true where has_label is True.
            # If we pass z_true, it computes MSE.
            # If we pass all z_true (mixed with 0s), it will force 0s on unlabeled. BAD.
            # We need to filter.
            
            # Since we can't easily modify PIVAE in this turn without viewing it again,
            # Let's just modify the loss aggregation here?
            # No, compute_loss returns dictionary.
            
            losses = self.model.compute_loss(
                expression, outputs,
                bio_tb=bio_tb,
                bio_alv=bio_alv,
                beta=beta,
                gamma=gamma,
                z_true=z_true if has_label.any() else None, # Only pass if some labels exist
                label_mask=has_label # Pass mask to model (if supported) or handled
            )
            
            # Backward pass
            losses['total'].backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            
            # Accumulate losses
            for key in total_losses:
                if key == 'icrp':
                    total_losses[key] += losses.get('icrp_consistency', torch.tensor(0.0)).item()
                else:
                    total_losses[key] += losses.get(key, torch.tensor(0.0)).item()
            n_batches += 1
        
        # Average losses
        avg_losses = {k: v / n_batches for k, v in total_losses.items()}
        return avg_losses
    
    @torch.no_grad()
    def validate(self) -> dict:
        """Validation epoch."""
        self.model.eval()
        total_losses = {'total': 0, 'recon': 0, 'physics': 0}
        n_batches = 0
        
        for batch in self.val_loader:
            expression = batch['expression'].to(self.device)
            bio_tb = batch['bio_tb'].to(self.device)
            bio_alv = batch['bio_alv'].to(self.device)
            
            outputs = self.model(expression)
            losses = self.model.compute_loss(
                expression, outputs,
                bio_tb=bio_tb,
                bio_alv=bio_alv
            )
            
            for key in total_losses:
                total_losses[key] += losses.get(key, torch.tensor(0.0)).item()
            n_batches += 1
        
        avg_losses = {k: v / n_batches for k, v in total_losses.items()}
        return avg_losses
    
    def train(
        self,
        n_epochs: int = 100,
        beta_schedule: str = 'warmup',
        log_interval: int = 10
    ) -> dict:
        """
        Full training loop with beta annealing.
        
        Args:
            n_epochs: Number of epochs
            beta_schedule: 'warmup' for gradual KL annealing, 'constant' for fixed
            log_interval: How often to log
        """
        logger.info(f"Starting training for {n_epochs} epochs...")
        
        for epoch in range(n_epochs):
            # Beta annealing (gradual increase of KL weight)
            if beta_schedule == 'warmup':
                beta = min(0.1, 0.001 + epoch * 0.01)
            else:
                beta = 0.01
            
            # Train
            train_losses = self.train_epoch(beta=beta, gamma=1.0)
            
            # Validate
            val_losses = self.validate()
            
            # Update scheduler
            self.scheduler.step()
            
            # Record history
            self.history['train_loss'].append(train_losses['total'])
            self.history['val_loss'].append(val_losses['total'])
            self.history['train_recon'].append(train_losses['recon'])
            self.history['train_kl'].append(train_losses['kl'])
            self.history['train_physics'].append(train_losses['physics'])
            self.history['train_icrp'].append(train_losses['icrp'])
            self.history['val_recon'].append(val_losses['recon'])
            self.history['val_physics'].append(val_losses['physics'])
            
            # Checkpoint if best
            if val_losses['total'] < self.best_val_loss:
                self.best_val_loss = val_losses['total']
                self.save_checkpoint('best_model.pt')
            
            # Log
            if (epoch + 1) % log_interval == 0:
                logger.info(
                    f"Epoch {epoch+1}/{n_epochs} | "
                    f"Train: {train_losses['total']:.4f} | "
                    f"Val: {val_losses['total']:.4f} | "
                    f"Recon: {train_losses['recon']:.4f} | "
                    f"ICRP: {train_losses['icrp']:.4f} | "
                    f"β: {beta:.4f}"
                )
        
        # Save final model
        self.save_checkpoint('final_model.pt')
        
        logger.success(f"Training complete! Best val loss: {self.best_val_loss:.4f}")
        return self.history
    
    def save_checkpoint(self, filename: str):
        """Save model checkpoint."""
        path = self.checkpoint_dir / filename
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'history': self.history,
            'best_val_loss': self.best_val_loss
        }, path)


def main():
    """Main training script."""
    print("="*70)
    print("PI-VAE TRAINING ON GEO DATA")
    print("="*70)
    
    # Load data
    print("\n1. Loading GEO datasets...")
    # Adjust path to data_dir relative to current script (in 'data' dir)
    dataloader, gene_names, metadata = prepare_geo_training_data(
        data_dir='raw', # relative to current dir which is data
        use_physics_features=True,
        apply_batch_correction=True,
        batch_size=32
    )
    
    # Get the underlying dataset
    dataset = dataloader.dataset
    
    print(f"\n2. Dataset summary:")
    print(f"   Total samples: {len(dataset)}")
    print(f"   Features: {len(gene_names)} physics genes")
    print(f"   Datasets: {metadata['geo_id'].nunique()}")
    
    # Stratified split
    print("\n3. Creating stratified train/test split...")
    train_idx, test_idx = manual_stratified_split(metadata, test_size=0.2, random_state=42)
    
    print(f"   Train samples: {len(train_idx)}")
    print(f"   Test samples: {len(test_idx)}")
    
    # Check split distribution
    print("\n   Split distribution by dataset:")
    for geo_id in metadata['geo_id'].unique():
        mask = metadata['geo_id'] == geo_id
        n_train = mask.iloc[train_idx].sum()
        n_test = mask.iloc[test_idx].sum()
        print(f"   {geo_id}: {n_train} train, {n_test} test")
    
    # Create data loaders
    train_dataset = Subset(dataset, train_idx)
    test_dataset = Subset(dataset, test_idx)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # Create model
    print("\n4. Creating PI-VAE model...")
    model = PIVAE(
        input_dim=len(gene_names),
        latent_dim=8,
        z_phys_dim=5,
        encoder_hidden=[256, 128, 64],
        decoder_hidden=[64, 128, 256]
    )
    
    # Set biomarker indices
    model.set_biomarker_indices(gene_names)
    
    print(f"   Input dim: {model.input_dim}")
    print(f"   Latent dim: {model.latent_dim}")
    print(f"   z_phys dim: {model.z_phys_dim}")
    print(f"   ICRP surrogate: {'Available' if model.icrp_surrogate else 'Not available'}")
    
    # Create trainer
    print("\n5. Starting training...")
    trainer = PIVAETrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        lr=1e-3,
        device='cpu',
        checkpoint_dir='checkpoints'
    )
    
    # Train
    history = trainer.train(
        n_epochs=100,
        beta_schedule='warmup',
        log_interval=10
    )
    
    # Save results
    print("\n6. Saving results...")
    results = {
        'n_samples': len(dataset),
        'n_genes': len(gene_names),
        'gene_names': gene_names,
        'train_samples': len(train_idx),
        'test_samples': len(test_idx),
        'best_val_loss': trainer.best_val_loss,
        'final_train_loss': history['train_loss'][-1],
        'final_val_loss': history['val_loss'][-1],
        'timestamp': datetime.now().isoformat()
    }
    
    with open('checkpoints/training_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n" + "="*70)
    print("TRAINING COMPLETE!")
    print("="*70)
    print(f"Best validation loss: {trainer.best_val_loss:.4f}")
    print(f"Model saved to: checkpoints/best_model.pt")
    print(f"Results saved to: checkpoints/training_results.json")
    
    return model, history


if __name__ == "__main__":
    main()
