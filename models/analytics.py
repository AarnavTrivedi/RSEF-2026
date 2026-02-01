"""
PulmoTrace Analytics Module
----------------------------
Advanced analytical capabilities for comparative analysis, 
sensitivity analysis, and predictive modeling.
"""

import numpy as np
import pandas as pd
import torch
from typing import Dict, List, Tuple, Optional, Any
from scipy import stats
from scipy.spatial.distance import euclidean
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


class ComparativeAnalyzer:
    """
    Performs comparative analysis between multiple samples.
    Includes differential expression, latent space comparison, and statistical testing.
    """
    
    def __init__(self, engine):
        """
        Args:
            engine: PulmoTraceEngine instance with loaded model
        """
        self.engine = engine
        self.gene_names = engine.gene_names
        
    def compare_samples(self, 
                       sample_ids: List[str], 
                       group_labels: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Compare multiple samples across all dimensions.
        
        Args:
            sample_ids: List of sample IDs to compare (2-4 samples)
            group_labels: Optional labels for grouping (e.g., ['Control', 'Treated'])
            
        Returns:
            Dictionary containing:
                - differential_expression: DataFrame with gene-level statistics
                - latent_comparison: Latent space coordinates and distances
                - deposition_comparison: Regional deposition differences
                - cell_composition: Cell type proportions
                - statistics: Summary statistics
        """
        if len(sample_ids) < 2:
            raise ValueError("Need at least 2 samples for comparison")
        if len(sample_ids) > 4:
            raise ValueError("Maximum 4 samples supported for comparison")
            
        # Load sample data
        samples_data = []
        for sid in sample_ids:
            result = self.engine.predict_sample(sid)
            samples_data.append(result)
            
        # 1. Differential Expression Analysis
        diff_expr = self._differential_expression(samples_data, sample_ids, group_labels)
        
        # 2. Latent Space Comparison
        latent_comp = self._latent_space_comparison(samples_data, sample_ids)
        
        # 3. Deposition Comparison
        depo_comp = self._deposition_comparison(samples_data, sample_ids)
        
        # 4. Cell Composition Comparison
        cell_comp = self._cell_composition_comparison(samples_data, sample_ids)
        
        # 5. Summary Statistics
        summary = self._compute_summary_statistics(samples_data, sample_ids)
        
        return {
            'differential_expression': diff_expr,
            'latent_comparison': latent_comp,
            'deposition_comparison': depo_comp,
            'cell_composition': cell_comp,
            'summary_statistics': summary,
            'sample_ids': sample_ids,
            'group_labels': group_labels
        }
    
    def _differential_expression(self, 
                                 samples_data: List[Dict], 
                                 sample_ids: List[str],
                                 group_labels: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Compute differential expression statistics between samples.
        """
        # Extract expression matrices
        expr_matrices = [s['expression'] for s in samples_data]
        
        # Create DataFrame
        expr_df = pd.DataFrame(expr_matrices, 
                              columns=self.gene_names,
                              index=sample_ids)
        
        # Compute statistics
        results = []
        for gene in self.gene_names:
            values = expr_df[gene].values
            
            # Basic statistics
            mean_expr = np.mean(values)
            std_expr = np.std(values)
            cv = std_expr / mean_expr if mean_expr > 0 else 0
            
            # Pairwise comparisons
            if len(values) == 2:
                fold_change = values[1] / values[0] if values[0] > 0 else np.inf
                log2fc = np.log2(fold_change) if fold_change > 0 else np.nan
                
                # T-test (if we have groups)
                if group_labels and len(set(group_labels)) == 2:
                    group1 = values[np.array(group_labels) == group_labels[0]]
                    group2 = values[np.array(group_labels) == group_labels[1]]
                    if len(group1) > 0 and len(group2) > 0:
                        t_stat, p_value = stats.ttest_ind(group1, group2)
                    else:
                        t_stat, p_value = np.nan, np.nan
                else:
                    t_stat, p_value = np.nan, np.nan
            else:
                # Multiple samples - use ANOVA or range
                fold_change = np.max(values) / np.min(values) if np.min(values) > 0 else np.inf
                log2fc = np.log2(fold_change) if fold_change > 0 else np.nan
                
                if group_labels and len(set(group_labels)) >= 2:
                    # ANOVA
                    groups = [values[np.array(group_labels) == label] for label in set(group_labels)]
                    f_stat, p_value = stats.f_oneway(*groups)
                    t_stat = f_stat
                else:
                    t_stat, p_value = np.nan, np.nan
            
            results.append({
                'gene': gene,
                'mean_expression': mean_expr,
                'std_expression': std_expr,
                'cv': cv,
                'fold_change': fold_change,
                'log2_fold_change': log2fc,
                'statistic': t_stat,
                'p_value': p_value,
                **{f'sample_{sid}': expr_df.loc[sid, gene] for sid in sample_ids}
            })
        
        df = pd.DataFrame(results)
        
        # Add significance flag
        df['significant'] = df['p_value'] < 0.05
        
        # Sort by absolute log2 fold change
        df = df.sort_values('log2_fold_change', key=abs, ascending=False)
        
        return df
    
    def _latent_space_comparison(self, 
                                 samples_data: List[Dict], 
                                 sample_ids: List[str]) -> Dict[str, Any]:
        """
        Compare samples in latent space (physics and biology manifolds).
        """
        # Extract latent coordinates
        z_phys_list = [s['z_phys'] for s in samples_data]
        z_bio_list = [s['z_bio'] for s in samples_data]
        
        z_phys = np.array(z_phys_list)
        z_bio = np.array(z_bio_list)
        
        # Compute pairwise distances
        phys_distances = {}
        bio_distances = {}
        
        for i in range(len(sample_ids)):
            for j in range(i+1, len(sample_ids)):
                pair = f"{sample_ids[i]}_vs_{sample_ids[j]}"
                phys_distances[pair] = euclidean(z_phys[i], z_phys[j])
                bio_distances[pair] = euclidean(z_bio[i], z_bio[j])
        
        # Compute centroids
        phys_centroid = np.mean(z_phys, axis=0)
        bio_centroid = np.mean(z_bio, axis=0)
        
        # Compute spread (variance)
        phys_spread = np.var(z_phys, axis=0)
        bio_spread = np.var(z_bio, axis=0)
        
        return {
            'z_phys': z_phys,
            'z_bio': z_bio,
            'phys_distances': phys_distances,
            'bio_distances': bio_distances,
            'phys_centroid': phys_centroid,
            'bio_centroid': bio_centroid,
            'phys_spread': phys_spread,
            'bio_spread': bio_spread,
            'sample_ids': sample_ids
        }
    
    def _deposition_comparison(self, 
                              samples_data: List[Dict], 
                              sample_ids: List[str]) -> pd.DataFrame:
        """
        Compare regional deposition patterns.
        """
        # Extract deposition predictions
        depo_list = [s['deposition'] for s in samples_data]
        
        # Create DataFrame
        regions = ['Head', 'Tracheobronchial', 'Alveolar', 'Total_Lung', 'Total_Respiratory']
        depo_df = pd.DataFrame(depo_list, 
                              columns=regions,
                              index=sample_ids)
        
        # Compute statistics
        stats_df = pd.DataFrame({
            'region': regions,
            'mean': depo_df.mean(),
            'std': depo_df.std(),
            'min': depo_df.min(),
            'max': depo_df.max(),
            'range': depo_df.max() - depo_df.min(),
            'cv': depo_df.std() / depo_df.mean()
        })
        
        # Add individual sample values
        for sid in sample_ids:
            stats_df[f'sample_{sid}'] = depo_df.loc[sid]
        
        return stats_df
    
    def _cell_composition_comparison(self, 
                                    samples_data: List[Dict], 
                                    sample_ids: List[str]) -> pd.DataFrame:
        """
        Compare cell type composition across samples.
        """
        # Extract cell proportions
        cell_list = [s['cell_proportions'] for s in samples_data]
        
        # Get cell type names from first sample
        cell_types = list(cell_list[0].keys())
        
        # Create matrix
        cell_matrix = np.array([[c[ct] for ct in cell_types] for c in cell_list])
        
        # Create DataFrame
        cell_df = pd.DataFrame(cell_matrix,
                              columns=cell_types,
                              index=sample_ids)
        
        # Compute statistics
        stats_df = pd.DataFrame({
            'cell_type': cell_types,
            'mean': cell_df.mean(),
            'std': cell_df.std(),
            'min': cell_df.min(),
            'max': cell_df.max(),
            'range': cell_df.max() - cell_df.min()
        })
        
        # Add individual sample values
        for sid in sample_ids:
            stats_df[f'sample_{sid}'] = cell_df.loc[sid]
        
        return stats_df
    
    def _compute_summary_statistics(self, 
                                   samples_data: List[Dict], 
                                   sample_ids: List[str]) -> Dict[str, Any]:
        """
        Compute overall summary statistics.
        """
        return {
            'n_samples': len(sample_ids),
            'n_genes': len(self.gene_names),
            'sample_ids': sample_ids,
            'timestamp': pd.Timestamp.now().isoformat()
        }


class SensitivityAnalyzer:
    """
    Performs sensitivity analysis to identify key factors influencing outcomes.
    Uses gradient-based methods and perturbation analysis.
    """
    
    def __init__(self, engine):
        """
        Args:
            engine: PulmoTraceEngine instance with loaded model
        """
        self.engine = engine
        self.gene_names = engine.gene_names
        
    def analyze_gene_sensitivity(self, 
                                 sample_id: str,
                                 outcome: str = 'deposition',
                                 perturbation_size: float = 0.1) -> pd.DataFrame:
        """
        Analyze sensitivity of outcome to each gene.
        
        Args:
            sample_id: Sample to analyze
            outcome: Which outcome to analyze ('deposition', 'z_phys', 'z_bio')
            perturbation_size: Size of perturbation (fraction of expression)
            
        Returns:
            DataFrame with sensitivity metrics for each gene
        """
        # Get baseline prediction
        baseline = self.engine.predict_sample(sample_id)
        baseline_expr = baseline['expression']
        
        # Compute sensitivity for each gene
        results = []
        
        for i, gene in enumerate(self.gene_names):
            # Perturb gene up
            expr_up = baseline_expr.copy()
            expr_up[i] *= (1 + perturbation_size)
            
            # Perturb gene down
            expr_down = baseline_expr.copy()
            expr_down[i] *= (1 - perturbation_size)
            
            # Get predictions
            pred_up = self._predict_from_expression(expr_up)
            pred_down = self._predict_from_expression(expr_down)
            
            # Compute sensitivity based on outcome
            if outcome == 'deposition':
                # Use total lung deposition
                baseline_val = baseline['deposition'][3]  # Total_Lung
                up_val = pred_up['deposition'][3]
                down_val = pred_down['deposition'][3]
            elif outcome == 'z_phys':
                # Use L2 norm of physics latent
                baseline_val = np.linalg.norm(baseline['z_phys'])
                up_val = np.linalg.norm(pred_up['z_phys'])
                down_val = np.linalg.norm(pred_down['z_phys'])
            elif outcome == 'z_bio':
                # Use L2 norm of biology latent
                baseline_val = np.linalg.norm(baseline['z_bio'])
                up_val = np.linalg.norm(pred_up['z_bio'])
                down_val = np.linalg.norm(pred_down['z_bio'])
            else:
                raise ValueError(f"Unknown outcome: {outcome}")
            
            # Compute sensitivity metrics
            delta_up = up_val - baseline_val
            delta_down = baseline_val - down_val
            sensitivity = (delta_up + delta_down) / (2 * perturbation_size * baseline_expr[i])
            
            # Absolute sensitivity
            abs_sensitivity = abs(sensitivity)
            
            # Normalized sensitivity (elasticity)
            elasticity = sensitivity * baseline_expr[i] / baseline_val if baseline_val != 0 else 0
            
            results.append({
                'gene': gene,
                'baseline_expression': baseline_expr[i],
                'baseline_outcome': baseline_val,
                'delta_up': delta_up,
                'delta_down': delta_down,
                'sensitivity': sensitivity,
                'abs_sensitivity': abs_sensitivity,
                'elasticity': elasticity,
                'direction': 'positive' if sensitivity > 0 else 'negative'
            })
        
        df = pd.DataFrame(results)
        df = df.sort_values('abs_sensitivity', ascending=False)
        
        return df
    
    def _predict_from_expression(self, expression: np.ndarray) -> Dict[str, Any]:
        """
        Helper to get predictions from expression vector.
        """
        # Convert to tensor
        x = torch.FloatTensor(expression).unsqueeze(0).to(self.engine.device)
        
        # Get latent representation
        with torch.no_grad():
            z, z_phys, z_bio = self.engine.model.encode(x)
            
            # Get deposition prediction
            depo_pred = self.engine.model.surrogate(z_phys)
            
        return {
            'z_phys': z_phys.cpu().numpy()[0],
            'z_bio': z_bio.cpu().numpy()[0],
            'deposition': depo_pred.cpu().numpy()[0]
        }
    
    def compute_global_sensitivity(self, 
                                   sample_ids: List[str],
                                   outcome: str = 'deposition',
                                   n_samples: int = 100) -> pd.DataFrame:
        """
        Compute global sensitivity across multiple samples using Sobol-like sampling.
        
        Args:
            sample_ids: List of samples to analyze
            outcome: Which outcome to analyze
            n_samples: Number of Monte Carlo samples
            
        Returns:
            DataFrame with global sensitivity indices
        """
        # For simplicity, average local sensitivities
        all_sensitivities = []
        
        for sid in sample_ids:
            sens = self.analyze_gene_sensitivity(sid, outcome)
            all_sensitivities.append(sens)
        
        # Combine results
        combined = pd.concat(all_sensitivities, ignore_index=True)
        
        # Aggregate by gene
        global_sens = combined.groupby('gene').agg({
            'sensitivity': ['mean', 'std'],
            'abs_sensitivity': ['mean', 'std'],
            'elasticity': ['mean', 'std']
        }).reset_index()
        
        # Flatten column names
        global_sens.columns = ['_'.join(col).strip('_') for col in global_sens.columns.values]
        
        # Sort by mean absolute sensitivity
        global_sens = global_sens.sort_values('abs_sensitivity_mean', ascending=False)
        
        return global_sens
