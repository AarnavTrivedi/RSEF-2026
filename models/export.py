"""
PulmoTrace Export Module
------------------------
Generate publication-ready reports and export data in various formats.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.backends.backend_pdf as pdf_backend
from datetime import datetime
from typing import Dict, List, Optional, Any
import io


class ReportGenerator:
    """
    Generates comprehensive PDF reports with visualizations and statistics.
    """
    
    def __init__(self, engine):
        """
        Args:
            engine: PulmoTraceEngine instance
        """
        self.engine = engine
        
    def generate_sample_report(self, 
                               sample_id: str,
                               output_path: str,
                               include_3d: bool = False) -> str:
        """
        Generate a comprehensive report for a single sample.
        
        Args:
            sample_id: Sample to analyze
            output_path: Path to save PDF report
            include_3d: Whether to include 3D visualization (requires screenshot)
            
        Returns:
            Path to generated report
        """
        # Get prediction results
        results = self.engine.predict_sample(sample_id)
        
        # Create PDF
        pdf = pdf_backend.PdfPages(output_path)
        
        try:
            # Page 1: Title and Summary
            self._create_title_page(pdf, sample_id, results)
            
            # Page 2: Gene Expression Heatmap
            self._create_expression_page(pdf, sample_id, results)
            
            # Page 3: Latent Space Visualization
            self._create_latent_page(pdf, sample_id, results)
            
            # Page 4: Deposition Analysis
            self._create_deposition_page(pdf, sample_id, results)
            
            # Page 5: Cell Composition
            self._create_cell_composition_page(pdf, sample_id, results)
            
        finally:
            pdf.close()
            
        return output_path
    
    def generate_comparison_report(self,
                                   comparison_results: Dict[str, Any],
                                   output_path: str) -> str:
        """
        Generate a comparative analysis report for multiple samples.
        
        Args:
            comparison_results: Results from ComparativeAnalyzer.compare_samples()
            output_path: Path to save PDF report
            
        Returns:
            Path to generated report
        """
        pdf = pdf_backend.PdfPages(output_path)
        
        try:
            # Page 1: Title
            self._create_comparison_title_page(pdf, comparison_results)
            
            # Page 2: Differential Expression
            self._create_diff_expr_page(pdf, comparison_results)
            
            # Page 3: Latent Space Comparison
            self._create_latent_comparison_page(pdf, comparison_results)
            
            # Page 4: Deposition Comparison
            self._create_deposition_comparison_page(pdf, comparison_results)
            
            # Page 5: Cell Composition Comparison
            self._create_cell_comparison_page(pdf, comparison_results)
            
        finally:
            pdf.close()
            
        return output_path
    
    def _create_title_page(self, pdf, sample_id: str, results: Dict):
        """Create title page with summary statistics."""
        fig, ax = plt.subplots(figsize=(8.5, 11))
        ax.axis('off')
        
        # Title
        title_text = f"PulmoTrace Analysis Report\nSample: {sample_id}"
        ax.text(0.5, 0.9, title_text, ha='center', va='top', 
                fontsize=20, fontweight='bold', transform=ax.transAxes)
        
        # Timestamp
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        ax.text(0.5, 0.85, f"Generated: {timestamp}", ha='center', va='top',
                fontsize=10, transform=ax.transAxes)
        
        # Summary statistics
        summary_y = 0.75
        ax.text(0.1, summary_y, "Summary Statistics", fontsize=14, fontweight='bold',
                transform=ax.transAxes)
        
        summary_y -= 0.05
        
        # Deposition
        total_depo = results['deposition'][3]  # Total lung
        ax.text(0.1, summary_y, f"Total Lung Deposition: {total_depo:.2f}%",
                fontsize=11, transform=ax.transAxes)
        summary_y -= 0.04
        
        # Top expressed genes
        expr = results['expression']
        gene_names = self.engine.gene_names
        top_genes_idx = np.argsort(expr)[-5:][::-1]
        top_genes = [gene_names[i] for i in top_genes_idx]
        
        ax.text(0.1, summary_y, f"Top 5 Expressed Genes:", fontsize=11, fontweight='bold',
                transform=ax.transAxes)
        summary_y -= 0.03
        
        for i, gene in enumerate(top_genes):
            ax.text(0.15, summary_y, f"{i+1}. {gene}: {expr[gene_names.index(gene)]:.2f}",
                    fontsize=10, transform=ax.transAxes)
            summary_y -= 0.03
        
        # Cell composition
        summary_y -= 0.02
        ax.text(0.1, summary_y, "Cell Type Composition:", fontsize=11, fontweight='bold',
                transform=ax.transAxes)
        summary_y -= 0.03
        
        for cell_type, prop in results['cell_proportions'].items():
            ax.text(0.15, summary_y, f"{cell_type}: {prop*100:.1f}%",
                    fontsize=10, transform=ax.transAxes)
            summary_y -= 0.03
        
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)
    
    def _create_expression_page(self, pdf, sample_id: str, results: Dict):
        """Create gene expression heatmap page."""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8.5, 11))
        
        # Top: Heatmap
        expr = results['expression']
        gene_names = self.engine.gene_names
        
        # Sort by expression
        sorted_idx = np.argsort(expr)[::-1]
        sorted_expr = expr[sorted_idx]
        sorted_genes = [gene_names[i] for i in sorted_idx]
        
        # Plot top 30 genes as heatmap
        top_n = min(30, len(sorted_genes))
        im = ax1.imshow(sorted_expr[:top_n].reshape(-1, 1), 
                       aspect='auto', cmap='YlOrRd')
        ax1.set_yticks(range(top_n))
        ax1.set_yticklabels(sorted_genes[:top_n], fontsize=8)
        ax1.set_xticks([])
        ax1.set_title(f"Top {top_n} Expressed Genes - {sample_id}", fontsize=12, fontweight='bold')
        plt.colorbar(im, ax=ax1, label='Expression Level')
        
        # Bottom: Bar plot of all genes
        ax2.barh(range(len(sorted_genes)), sorted_expr, color='steelblue', alpha=0.7)
        ax2.set_xlabel('Expression Level', fontsize=10)
        ax2.set_ylabel('Gene Rank', fontsize=10)
        ax2.set_title('All Genes Ranked by Expression', fontsize=12, fontweight='bold')
        ax2.grid(axis='x', alpha=0.3)
        
        plt.tight_layout()
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)
    
    def _create_latent_page(self, pdf, sample_id: str, results: Dict):
        """Create latent space visualization page."""
        fig = plt.figure(figsize=(8.5, 11))
        
        # Physics latent (5D)
        ax1 = plt.subplot(2, 1, 1)
        z_phys = results['z_phys']
        ax1.bar(range(len(z_phys)), z_phys, color='coral', alpha=0.7)
        ax1.set_xlabel('Physics Dimension', fontsize=10)
        ax1.set_ylabel('Latent Value', fontsize=10)
        ax1.set_title(f'Physics Latent Space (5D) - {sample_id}', fontsize=12, fontweight='bold')
        ax1.grid(axis='y', alpha=0.3)
        ax1.axhline(0, color='black', linewidth=0.5)
        
        # Biology latent (3D)
        ax2 = plt.subplot(2, 1, 2)
        z_bio = results['z_bio']
        ax2.bar(range(len(z_bio)), z_bio, color='mediumseagreen', alpha=0.7)
        ax2.set_xlabel('Biology Dimension', fontsize=10)
        ax2.set_ylabel('Latent Value', fontsize=10)
        ax2.set_title(f'Biology Latent Space (3D) - {sample_id}', fontsize=12, fontweight='bold')
        ax2.grid(axis='y', alpha=0.3)
        ax2.axhline(0, color='black', linewidth=0.5)
        
        plt.tight_layout()
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)
    
    def _create_deposition_page(self, pdf, sample_id: str, results: Dict):
        """Create deposition analysis page."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 8.5))
        
        # Regional deposition bar chart
        depo = results['deposition']
        regions = ['Head', 'Tracheobronchial', 'Alveolar', 'Total Lung', 'Total Respiratory']
        
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8']
        ax1.barh(regions, depo, color=colors, alpha=0.8)
        ax1.set_xlabel('Deposition (%)', fontsize=11)
        ax1.set_title(f'Regional Deposition - {sample_id}', fontsize=13, fontweight='bold')
        ax1.grid(axis='x', alpha=0.3)
        
        # Add value labels
        for i, v in enumerate(depo):
            ax1.text(v + 0.5, i, f'{v:.2f}%', va='center', fontsize=9)
        
        # Pie chart for distribution
        lung_regions = depo[:3]  # Head, TB, Alveolar
        lung_labels = regions[:3]
        ax2.pie(lung_regions, labels=lung_labels, autopct='%1.1f%%',
                colors=colors[:3], startangle=90)
        ax2.set_title('Deposition Distribution', fontsize=13, fontweight='bold')
        
        plt.tight_layout()
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)
    
    def _create_cell_composition_page(self, pdf, sample_id: str, results: Dict):
        """Create cell composition page."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 8.5))
        
        cell_props = results['cell_proportions']
        cell_types = list(cell_props.keys())
        proportions = list(cell_props.values())
        
        # Bar chart
        colors_palette = plt.cm.Set3(range(len(cell_types)))
        ax1.barh(cell_types, proportions, color=colors_palette, alpha=0.8)
        ax1.set_xlabel('Proportion', fontsize=11)
        ax1.set_title(f'Cell Type Proportions - {sample_id}', fontsize=13, fontweight='bold')
        ax1.grid(axis='x', alpha=0.3)
        
        # Add percentage labels
        for i, v in enumerate(proportions):
            ax1.text(v + 0.01, i, f'{v*100:.1f}%', va='center', fontsize=9)
        
        # Pie chart
        ax2.pie(proportions, labels=cell_types, autopct='%1.1f%%',
                colors=colors_palette, startangle=90)
        ax2.set_title('Cell Composition', fontsize=13, fontweight='bold')
        
        plt.tight_layout()
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)
    
    def _create_comparison_title_page(self, pdf, comparison_results: Dict):
        """Create title page for comparison report."""
        fig, ax = plt.subplots(figsize=(8.5, 11))
        ax.axis('off')
        
        sample_ids = comparison_results['sample_ids']
        
        # Title
        title_text = f"PulmoTrace Comparative Analysis\n{len(sample_ids)} Samples"
        ax.text(0.5, 0.9, title_text, ha='center', va='top',
                fontsize=20, fontweight='bold', transform=ax.transAxes)
        
        # Timestamp
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        ax.text(0.5, 0.85, f"Generated: {timestamp}", ha='center', va='top',
                fontsize=10, transform=ax.transAxes)
        
        # Sample list
        y_pos = 0.75
        ax.text(0.1, y_pos, "Samples Compared:", fontsize=14, fontweight='bold',
                transform=ax.transAxes)
        y_pos -= 0.05
        
        for i, sid in enumerate(sample_ids):
            ax.text(0.15, y_pos, f"{i+1}. {sid}", fontsize=11,
                    transform=ax.transAxes)
            y_pos -= 0.04
        
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)
    
    def _create_diff_expr_page(self, pdf, comparison_results: Dict):
        """Create differential expression page."""
        diff_expr = comparison_results['differential_expression']
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8.5, 11))
        
        # Top: Volcano plot (log2FC vs -log10(p-value))
        log2fc = diff_expr['log2_fold_change'].fillna(0)
        pval = diff_expr['p_value'].fillna(1)
        neg_log_p = -np.log10(pval + 1e-10)
        
        # Color by significance
        colors = ['red' if (abs(fc) > 1 and p < 0.05) else 'gray' 
                 for fc, p in zip(log2fc, pval)]
        
        ax1.scatter(log2fc, neg_log_p, c=colors, alpha=0.6, s=30)
        ax1.axhline(-np.log10(0.05), color='blue', linestyle='--', linewidth=1, label='p=0.05')
        ax1.axvline(-1, color='green', linestyle='--', linewidth=1)
        ax1.axvline(1, color='green', linestyle='--', linewidth=1, label='|log2FC|=1')
        ax1.set_xlabel('log2(Fold Change)', fontsize=11)
        ax1.set_ylabel('-log10(p-value)', fontsize=11)
        ax1.set_title('Differential Expression Volcano Plot', fontsize=13, fontweight='bold')
        ax1.legend()
        ax1.grid(alpha=0.3)
        
        # Bottom: Top 20 genes bar plot
        top_genes = diff_expr.nlargest(20, 'abs_sensitivity' if 'abs_sensitivity' in diff_expr.columns else 'log2_fold_change', keep='first')
        
        ax2.barh(range(len(top_genes)), top_genes['log2_fold_change'], 
                color=['red' if x > 0 else 'blue' for x in top_genes['log2_fold_change']],
                alpha=0.7)
        ax2.set_yticks(range(len(top_genes)))
        ax2.set_yticklabels(top_genes['gene'], fontsize=8)
        ax2.set_xlabel('log2(Fold Change)', fontsize=11)
        ax2.set_title('Top 20 Differentially Expressed Genes', fontsize=13, fontweight='bold')
        ax2.axvline(0, color='black', linewidth=0.5)
        ax2.grid(axis='x', alpha=0.3)
        
        plt.tight_layout()
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)
    
    def _create_latent_comparison_page(self, pdf, comparison_results: Dict):
        """Create latent space comparison page."""
        latent_comp = comparison_results['latent_comparison']
        sample_ids = comparison_results['sample_ids']
        
        fig = plt.figure(figsize=(8.5, 11))
        
        # Physics latent comparison
        ax1 = plt.subplot(2, 1, 1)
        z_phys = latent_comp['z_phys']
        
        for i, sid in enumerate(sample_ids):
            ax1.plot(range(5), z_phys[i], marker='o', label=sid, linewidth=2)
        
        ax1.set_xlabel('Physics Dimension', fontsize=11)
        ax1.set_ylabel('Latent Value', fontsize=11)
        ax1.set_title('Physics Latent Space Comparison', fontsize=13, fontweight='bold')
        ax1.legend()
        ax1.grid(alpha=0.3)
        ax1.axhline(0, color='black', linewidth=0.5)
        
        # Biology latent comparison
        ax2 = plt.subplot(2, 1, 2)
        z_bio = latent_comp['z_bio']
        
        for i, sid in enumerate(sample_ids):
            ax2.plot(range(3), z_bio[i], marker='s', label=sid, linewidth=2)
        
        ax2.set_xlabel('Biology Dimension', fontsize=11)
        ax2.set_ylabel('Latent Value', fontsize=11)
        ax2.set_title('Biology Latent Space Comparison', fontsize=13, fontweight='bold')
        ax2.legend()
        ax2.grid(alpha=0.3)
        ax2.axhline(0, color='black', linewidth=0.5)
        
        plt.tight_layout()
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)
    
    def _create_deposition_comparison_page(self, pdf, comparison_results: Dict):
        """Create deposition comparison page."""
        depo_comp = comparison_results['deposition_comparison']
        sample_ids = comparison_results['sample_ids']
        
        fig, ax = plt.subplots(figsize=(11, 8.5))
        
        regions = depo_comp['region'].tolist()
        x = np.arange(len(regions))
        width = 0.8 / len(sample_ids)
        
        colors = plt.cm.Set2(range(len(sample_ids)))
        
        for i, sid in enumerate(sample_ids):
            values = depo_comp[f'sample_{sid}'].values
            ax.bar(x + i * width, values, width, label=sid, color=colors[i], alpha=0.8)
        
        ax.set_xlabel('Region', fontsize=12)
        ax.set_ylabel('Deposition (%)', fontsize=12)
        ax.set_title('Regional Deposition Comparison', fontsize=14, fontweight='bold')
        ax.set_xticks(x + width * (len(sample_ids) - 1) / 2)
        ax.set_xticklabels(regions, rotation=15, ha='right')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)
    
    def _create_cell_comparison_page(self, pdf, comparison_results: Dict):
        """Create cell composition comparison page."""
        cell_comp = comparison_results['cell_composition']
        sample_ids = comparison_results['sample_ids']
        
        fig, ax = plt.subplots(figsize=(11, 8.5))
        
        cell_types = cell_comp['cell_type'].tolist()
        x = np.arange(len(cell_types))
        width = 0.8 / len(sample_ids)
        
        colors = plt.cm.Set3(range(len(sample_ids)))
        
        for i, sid in enumerate(sample_ids):
            values = cell_comp[f'sample_{sid}'].values
            ax.bar(x + i * width, values, width, label=sid, color=colors[i], alpha=0.8)
        
        ax.set_xlabel('Cell Type', fontsize=12)
        ax.set_ylabel('Proportion', fontsize=12)
        ax.set_title('Cell Type Composition Comparison', fontsize=14, fontweight='bold')
        ax.set_xticks(x + width * (len(sample_ids) - 1) / 2)
        ax.set_xticklabels(cell_types, rotation=30, ha='right', fontsize=9)
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)


class DataExporter:
    """
    Export data in various formats (CSV, Excel, JSON).
    """
    
    @staticmethod
    def export_to_csv(data: pd.DataFrame, output_path: str) -> str:
        """Export DataFrame to CSV."""
        data.to_csv(output_path, index=True)
        return output_path
    
    @staticmethod
    def export_to_excel(data_dict: Dict[str, pd.DataFrame], output_path: str) -> str:
        """
        Export multiple DataFrames to Excel with separate sheets.
        
        Args:
            data_dict: Dictionary mapping sheet names to DataFrames
            output_path: Path to save Excel file
        """
        with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
            for sheet_name, df in data_dict.items():
                df.to_excel(writer, sheet_name=sheet_name, index=True)
        
        return output_path
    
    @staticmethod
    def export_comparison_data(comparison_results: Dict[str, Any], 
                              output_dir: str) -> Dict[str, str]:
        """
        Export all comparison data to separate files.
        
        Args:
            comparison_results: Results from ComparativeAnalyzer
            output_dir: Directory to save files
            
        Returns:
            Dictionary mapping data types to file paths
        """
        os.makedirs(output_dir, exist_ok=True)
        
        exported_files = {}
        
        # Differential expression
        diff_expr_path = os.path.join(output_dir, 'differential_expression.csv')
        comparison_results['differential_expression'].to_csv(diff_expr_path, index=False)
        exported_files['differential_expression'] = diff_expr_path
        
        # Deposition comparison
        depo_path = os.path.join(output_dir, 'deposition_comparison.csv')
        comparison_results['deposition_comparison'].to_csv(depo_path, index=False)
        exported_files['deposition_comparison'] = depo_path
        
        # Cell composition
        cell_path = os.path.join(output_dir, 'cell_composition.csv')
        comparison_results['cell_composition'].to_csv(cell_path, index=False)
        exported_files['cell_composition'] = cell_path
        
        # Latent space coordinates
        latent_comp = comparison_results['latent_comparison']
        latent_df = pd.DataFrame({
            'sample_id': latent_comp['sample_ids'],
            **{f'z_phys_{i}': latent_comp['z_phys'][:, i] for i in range(5)},
            **{f'z_bio_{i}': latent_comp['z_bio'][:, i] for i in range(3)}
        })
        latent_path = os.path.join(output_dir, 'latent_coordinates.csv')
        latent_df.to_csv(latent_path, index=False)
        exported_files['latent_coordinates'] = latent_path
        
        return exported_files
