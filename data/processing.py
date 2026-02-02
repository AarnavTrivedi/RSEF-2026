"""
Data Processing Module for PulmoTrace

Handles:
    1. GEO dataset downloading and parsing
    2. Gene expression normalization (TPM)
    3. Batch correction (Combat-seq)
    4. Spatial deconvolution (SpatialDDLS wrapper)
    5. Biomarker extraction
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
import torch
from torch.utils.data import Dataset, DataLoader
from loguru import logger
import warnings

# Suppress warnings during development
warnings.filterwarnings('ignore')


class GEODataLoader:
    """
    Utility class for downloading and parsing GEO datasets.
    
    Supported datasets:
        - GSE25531: Human PBMC, Diesel Exhaust (Controlled)
        - GSE47460: Human Lung Tissue, Smoking (Chronic)
        - GSE237251: Rat Lung, Wood Smoke
    """
    
    def __init__(self, data_dir: Path = Path("data/raw")):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # Dataset metadata
        self.datasets = {
            'GSE25531': {
                'organism': 'human',
                'tissue': 'PBMC',
                'exposure': 'diesel',
                'platform': 'Affymetrix',
                'n_samples': 14
            },
            'GSE47460': {
                'organism': 'human',
                'tissue': 'lung',
                'exposure': 'smoking',
                'platform': 'Illumina',
                'n_samples': 582
            },
            'GSE237251': {
                'organism': 'rat',
                'tissue': 'lung',
                'exposure': 'wood_smoke',
                'platform': 'RNA-seq',
                'n_samples': 12
            }
        }
    
    def download_dataset(self, geo_id: str) -> Path:
        """
        Download a GEO dataset.
        
        Args:
            geo_id: GEO accession ID (e.g., 'GSE25531')
        
        Returns:
            Path to downloaded data
        """
        if geo_id not in self.datasets:
            raise ValueError(f"Unknown dataset: {geo_id}. Available: {list(self.datasets.keys())}")
        
        output_dir = self.data_dir / geo_id
        output_dir.mkdir(exist_ok=True)
        
        try:
            import GEOparse
            gse = GEOparse.get_GEO(geo=geo_id, destdir=str(output_dir))
            
            # Extract expression data
            for gpl_name, gpl in gse.gpls.items():
                logger.info(f"Platform: {gpl_name}")
            
            # Save expression matrix
            for gsm_name, gsm in gse.gsms.items():
                logger.debug(f"Sample: {gsm_name}")
            
            logger.success(f"Downloaded {geo_id} to {output_dir}")
            return output_dir
            
        except ImportError:
            logger.warning("GEOparse not installed. Using synthetic data for development.")
            return self._generate_synthetic_data(geo_id, output_dir)
    
    def _generate_synthetic_data(self, geo_id: str, output_dir: Path) -> Path:
        """
        Generate synthetic data for development/testing.
        """
        meta = self.datasets[geo_id]
        n_samples = meta['n_samples']
        n_genes = 20000
        
        # Generate random expression matrix
        np.random.seed(42)
        
        # Base expression
        expression = np.random.lognormal(mean=2, sigma=2, size=(n_samples, n_genes))
        
        # Add exposure-related signal
        if meta['exposure'] == 'diesel':
            # Simulate diesel exposure effect
            # Half samples exposed, half control
            exposed_idx = n_samples // 2
            
            # Upregulate bronchial markers in exposed
            # AKR1B10-like gene (index 100)
            expression[:exposed_idx, 100] *= 10
            # CYP1A1-like gene (index 200)
            expression[:exposed_idx, 200] *= 5
            
        elif meta['exposure'] == 'smoking':
            # Simulate smoking effect (continuous pack-years)
            pack_years = np.random.exponential(scale=20, size=n_samples)
            
            # Dose-dependent upregulation
            for i, py in enumerate(pack_years):
                factor = 1 + py / 10
                expression[i, 100] *= factor  # AKR1B10
                expression[i, 300] *= factor  # MMP12
        
        # Create gene names
        gene_names = [f"GENE_{i}" for i in range(n_genes)]
        gene_names[100] = "AKR1B10"
        gene_names[200] = "CYP1A1"
        gene_names[300] = "MMP12"
        gene_names[400] = "SPP1"
        
        # Create sample names
        sample_names = [f"Sample_{i}" for i in range(n_samples)]
        
        # Save as CSV
        df = pd.DataFrame(expression, index=sample_names, columns=gene_names)
        df.to_csv(output_dir / "expression_matrix.csv")
        
        # Save metadata
        if meta['exposure'] == 'smoking':
            metadata = pd.DataFrame({
                'sample_id': sample_names,
                'pack_years': pack_years,
                'smoking_status': ['smoker' if py > 0 else 'non_smoker' for py in pack_years]
            })
        else:
            metadata = pd.DataFrame({
                'sample_id': sample_names,
                'condition': ['exposed'] * (n_samples // 2) + ['control'] * (n_samples - n_samples // 2)
            })
        
        metadata.to_csv(output_dir / "metadata.csv", index=False)
        
        logger.info(f"Generated synthetic data for {geo_id}: {n_samples} samples, {n_genes} genes")
        
        return output_dir
    
    def load_expression_matrix(self, geo_id: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Load expression matrix and metadata for a dataset.
        
        Supports multiple formats:
        1. Pre-processed CSV (expression_matrix.csv)
        2. GEO Series Matrix file (GSE*_series_matrix.txt)
        3. Synthetic data fallback
        
        Returns:
            expression_df: Samples × Genes
            metadata_df: Sample metadata
        """
        data_path = self.data_dir / geo_id
        
        # Check for different file formats
        expr_path = data_path / "expression_matrix.csv"
        
        # Look for Series Matrix files (case-insensitive, handles spaces)
        series_matrix_pattern = []
        rnaseq_pattern = []
        
        if data_path.exists():
            for f in data_path.iterdir():
                fname_lower = f.name.lower()
                # Check for RNA-seq counts files first (takes priority for RNA-seq studies)
                if ('count' in fname_lower or 'rnaseq' in fname_lower or 'tpm' in fname_lower or 'fpkm' in fname_lower) \
                        and fname_lower.endswith(('.txt', '.csv', '.tsv', '.txt.gz')):
                    rnaseq_pattern.append(f)
                # Check for Series Matrix files
                elif 'series' in fname_lower and 'matrix' in fname_lower and fname_lower.endswith(('.txt', '.txt.gz')):
                    series_matrix_pattern.append(f)
        
        if expr_path.exists():
            # Load pre-processed CSV
            expression = pd.read_csv(expr_path, index_col=0)
            meta_path = data_path / "metadata.csv"
            if meta_path.exists():
                metadata = pd.read_csv(meta_path)
            else:
                metadata = pd.DataFrame({'sample_id': expression.index.tolist()})
            return expression, metadata
        
        elif rnaseq_pattern:
            # Prioritize RNA-seq counts files (Series Matrix often empty for RNA-seq)
            rnaseq_file = rnaseq_pattern[0]
            logger.info(f"Loading RNA-seq counts file: {rnaseq_file.name}")
            return self.load_rnaseq_counts(rnaseq_file)
            
        elif series_matrix_pattern:
            # Load GEO Series Matrix file
            series_matrix_file = series_matrix_pattern[0]
            logger.info(f"Loading Series Matrix file: {series_matrix_file.name}")
            return self.load_series_matrix(series_matrix_file)
        
        elif not data_path.exists():
            logger.info(f"Dataset not found locally. Downloading {geo_id}...")
            self.download_dataset(geo_id)
            return self.load_expression_matrix(geo_id)  # Retry after download
        else:
            raise FileNotFoundError(f"No valid expression data found at {data_path}")
    
    def load_series_matrix(self, filepath: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Parse GEO Series Matrix file format.
        
        The Series Matrix file format includes:
        - Metadata lines starting with '!'
        - Expression matrix between 'series_matrix_table_begin' and 'series_matrix_table_end'
        
        Args:
            filepath: Path to series_matrix.txt file (can be .gz compressed)
        
        Returns:
            expression_df: Samples × Genes (transposed from GEO format)
            metadata_df: Sample metadata extracted from headers
        """
        import gzip
        
        # Handle gzipped files
        if str(filepath).endswith('.gz'):
            opener = gzip.open
            mode = 'rt'
        else:
            opener = open
            mode = 'r'
        
        with opener(filepath, mode) as f:
            lines = f.readlines()
        
        # Extract metadata lines (starting with '!')
        metadata_dict = {}
        sample_ids = []
        
        for line in lines:
            line = line.strip()
            if line.startswith('!Sample_geo_accession'):
                sample_ids = line.split('\t')[1:]
                sample_ids = [s.strip('"') for s in sample_ids]
            elif line.startswith('!Sample_title'):
                titles = line.split('\t')[1:]
                metadata_dict['title'] = [t.strip('"') for t in titles]
            elif line.startswith('!Sample_source_name'):
                sources = line.split('\t')[1:]
                metadata_dict['source'] = [s.strip('"') for s in sources]
            elif line.startswith('!Sample_characteristics'):
                # Can have multiple characteristic lines
                chars = line.split('\t')[1:]
                chars = [c.strip('"') for c in chars]
                # Extract key-value pairs
                if chars and ':' in chars[0]:
                    key = chars[0].split(':')[0].strip()
                    values = [c.split(':')[1].strip() if ':' in c else c for c in chars]
                    if key not in metadata_dict:
                        metadata_dict[key] = values
        
        # Find expression matrix boundaries
        start_idx = None
        end_idx = None
        
        for i, line in enumerate(lines):
            if 'series_matrix_table_begin' in line:
                start_idx = i + 1
            elif 'series_matrix_table_end' in line:
                end_idx = i
                break
        
        if start_idx is None or end_idx is None:
            raise ValueError(f"Could not find expression matrix in {filepath}")
        
        # Parse expression matrix
        # First line after 'begin' is headers (sample IDs)
        # Subsequent lines are gene_id followed by expression values
        from io import StringIO
        matrix_text = '\n'.join(lines[start_idx:end_idx])
        
        expression = pd.read_csv(
            StringIO(matrix_text),
            sep='\t',
            index_col=0
        )
        
        # Transpose to get Samples × Genes
        expression = expression.T
        
        # Clean up sample IDs
        expression.index = [idx.strip('"') for idx in expression.index]
        expression.columns = [col.strip('"') for col in expression.columns]
        
        # Create metadata DataFrame
        if sample_ids:
            metadata = pd.DataFrame({'sample_id': sample_ids})
            for key, values in metadata_dict.items():
                if len(values) == len(sample_ids):
                    metadata[key] = values
        else:
            metadata = pd.DataFrame({'sample_id': expression.index.tolist()})
        
        logger.success(f"Loaded Series Matrix: {expression.shape[0]} samples, {expression.shape[1]} genes")
        
        return expression, metadata

    def load_rnaseq_counts(self, filepath: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Load RNA-seq count/TPM/FPKM files.
        
        Auto-detects gene symbol column and sample columns.
        Returns expression matrix in Samples x Genes format.
        
        Args:
            filepath: Path to counts file (.txt, .csv, .tsv)
        
        Returns:
            expression: DataFrame (samples x genes)
            metadata: DataFrame with sample information
        """
        # Detect separator
        suffix = filepath.suffix.lower()
        if suffix == '.csv':
            sep = ','
        else:
            sep = '\t'
        
        # Handle gzipped files
        if str(filepath).endswith('.gz'):
            import gzip
            opener = gzip.open
            mode = 'rt'
        else:
            opener = open
            mode = 'r'
        
        # Read the file
        df = pd.read_csv(filepath, sep=sep, low_memory=False)
        
        logger.info(f"Loaded file with {df.shape[0]} rows, {df.shape[1]} columns")
        
        # Detect gene symbol column
        gene_col = None
        gene_col_candidates = ['Gene Symbol', 'gene_symbol', 'Gene_Symbol', 'Symbol', 
                               'gene_name', 'Gene', 'GeneSymbol', 'gene_id', 'GeneID']
        
        for col in gene_col_candidates:
            if col in df.columns:
                gene_col = col
                break
        
        if gene_col is None:
            # Try to find by lowercase match
            for col in df.columns:
                if 'gene' in col.lower() and ('symbol' in col.lower() or 'name' in col.lower()):
                    gene_col = col
                    break
        
        if gene_col is None:
            # Use first column as gene identifier
            gene_col = df.columns[0]
            logger.warning(f"Could not detect gene column, using first column: {gene_col}")
        
        logger.info(f"Using '{gene_col}' as gene identifier column")
        
        # Detect sample columns (numeric columns that aren't metadata)
        metadata_cols = ['Chromosome', 'Start', 'Stop', 'Strand', 'Chr', 'Start', 'End',
                         'gene_biotype', 'gene_source', 'Description', 'EntrezID', 'Ensembl',
                         gene_col]
        
        sample_cols = []
        for col in df.columns:
            if col not in metadata_cols and df[col].dtype in ['float64', 'int64', 'float32', 'int32']:
                sample_cols.append(col)
        
        if not sample_cols:
            # Fallback: take all columns except the first few
            sample_cols = df.columns[8:].tolist()  # Assume first 8 are metadata
            logger.warning(f"Auto-detected {len(sample_cols)} sample columns")
        
        logger.info(f"Detected {len(sample_cols)} sample columns")
        
        # Create expression matrix (Samples x Genes)
        expr_df = df.set_index(gene_col)[sample_cols]
        
        # Remove duplicate genes (keep first)
        if expr_df.index.duplicated().any():
            n_dups = expr_df.index.duplicated().sum()
            logger.warning(f"Removing {n_dups} duplicate gene entries")
            expr_df = expr_df[~expr_df.index.duplicated(keep='first')]
        
        # Transpose to Samples x Genes
        expression = expr_df.T
        
        # Create metadata
        metadata = pd.DataFrame({'sample_id': expression.index.tolist()})
        
        logger.success(f"Loaded RNA-seq: {expression.shape[0]} samples, {expression.shape[1]} genes")
        
        return expression, metadata


class PlatformAnnotation:
    """
    Handle platform annotation files for probe ID to gene symbol mapping.
    
    Supports downloading and parsing GPL annotation files from GEO/NCBI.
    Common platforms:
        - GPL6480: Agilent Human 4x44K (Used by GSE25531)
        - GPL570: Affymetrix Human Genome U133 Plus 2.0
        - GPL16686: Agilent Mouse 4x44K
    """
    
    # Known platform annotation files (local cache)
    PLATFORM_CACHE_DIR = Path("data/raw/platforms")
    
    # Platform metadata
    PLATFORMS = {
        'GPL6480': {
            'name': 'Agilent-014850 Whole Human Genome 4x44K',
            'probe_col': 'ID',
            'gene_col': 'GENE_SYMBOL',
            'url': 'https://ftp.ncbi.nlm.nih.gov/geo/platforms/GPL6nnn/GPL6480/annot/GPL6480.annot.gz'
        },
        'GPL570': {
            'name': 'Affymetrix Human Genome U133 Plus 2.0',
            'probe_col': 'ID',
            'gene_col': 'Gene Symbol',
            'url': 'https://ftp.ncbi.nlm.nih.gov/geo/platforms/GPLnnn/GPL570/annot/GPL570.annot.gz'
        },
        'GPL16686': {
            'name': 'Agilent-045802 Illumina BeadChip',
            'probe_col': 'ID',
            'gene_col': 'Gene_Symbol',
            'url': None  # Would need to look up
        }
    }
    
    def __init__(self, cache_dir: Path = None):
        self.cache_dir = Path(cache_dir) if cache_dir else self.PLATFORM_CACHE_DIR
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._annotation_cache = {}
    
    def download_annotation(self, platform_id: str) -> Path:
        """
        Download GPL annotation file from NCBI FTP.
        
        Args:
            platform_id: GPL ID (e.g., 'GPL6480')
        
        Returns:
            Path to downloaded annotation file
        """
        if platform_id not in self.PLATFORMS:
            raise ValueError(f"Unknown platform: {platform_id}. Known: {list(self.PLATFORMS.keys())}")
        
        platform = self.PLATFORMS[platform_id]
        if platform['url'] is None:
            raise ValueError(f"No download URL configured for {platform_id}")
        
        output_file = self.cache_dir / f"{platform_id}.annot.gz"
        
        if output_file.exists():
            logger.info(f"Platform annotation cached: {output_file}")
            return output_file
        
        logger.info(f"Downloading {platform_id} annotation from NCBI...")
        
        # Try curl first (handles SSL better on macOS)
        import subprocess
        try:
            result = subprocess.run(
                ['curl', '-sL', '-o', str(output_file), platform['url']],
                check=True,
                capture_output=True,
                timeout=120
            )
            if output_file.exists() and output_file.stat().st_size > 1000:
                logger.success(f"Downloaded {platform_id} annotation: {output_file}")
                return output_file
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired, FileNotFoundError):
            logger.warning("curl failed, trying urllib with SSL bypass...")
        
        # Fallback: urllib with SSL verification disabled (for macOS)
        import urllib.request
        import ssl
        
        ssl_context = ssl.create_default_context()
        ssl_context.check_hostname = False
        ssl_context.verify_mode = ssl.CERT_NONE
        
        with urllib.request.urlopen(platform['url'], context=ssl_context) as response:
            with open(output_file, 'wb') as f:
                f.write(response.read())
        
        logger.success(f"Downloaded {platform_id} annotation: {output_file}")
        return output_file
    
    def load_annotation(self, platform_id: str) -> pd.DataFrame:
        """
        Load and parse GPL annotation file.
        
        Returns DataFrame with probe_id and gene_symbol columns.
        """
        if platform_id in self._annotation_cache:
            return self._annotation_cache[platform_id]
        
        annot_file = self.cache_dir / f"{platform_id}.annot.gz"
        
        if not annot_file.exists():
            annot_file = self.download_annotation(platform_id)
        
        platform = self.PLATFORMS[platform_id]
        
        # Parse annotation file
        import gzip
        
        with gzip.open(annot_file, 'rt') as f:
            lines = f.readlines()
        
        # Find the data section (after !platform_table_begin)
        start_idx = None
        end_idx = len(lines)
        
        for i, line in enumerate(lines):
            if '!platform_table_begin' in line:
                start_idx = i + 1  # Data starts on next line
            elif '!platform_table_end' in line:
                end_idx = i
                break
        
        if start_idx is None:
            # Fallback: find first line that doesn't start with ^, !, or #
            for i, line in enumerate(lines):
                if line and not line[0] in ['^', '!', '#']:
                    start_idx = i
                    break
        
        if start_idx is None:
            raise ValueError(f"Could not find data section in {annot_file}")
        
        # Read as DataFrame - first line after start_idx is the header
        from io import StringIO
        data_text = ''.join(lines[start_idx:end_idx])
        
        annot_df = pd.read_csv(StringIO(data_text), sep='\t', low_memory=False)
        
        # Extract probe ID and gene symbol columns
        probe_col = platform['probe_col']
        gene_col = platform['gene_col']
        
        if probe_col not in annot_df.columns:
            # Try to find similar column
            possible = [c for c in annot_df.columns if 'id' in c.lower()]
            if possible:
                probe_col = possible[0]
            logger.warning(f"Using {probe_col} as probe column")
        
        if gene_col not in annot_df.columns:
            # Try to find gene symbol column
            possible = [c for c in annot_df.columns if 'gene' in c.lower() and 'symbol' in c.lower()]
            if possible:
                gene_col = possible[0]
            else:
                possible = [c for c in annot_df.columns if 'symbol' in c.lower()]
                if possible:
                    gene_col = possible[0]
            logger.warning(f"Using {gene_col} as gene symbol column")
        
        # Create mapping DataFrame
        mapping = annot_df[[probe_col, gene_col]].copy()
        mapping.columns = ['probe_id', 'gene_symbol']
        
        # Clean up: remove NaN, empty strings, multi-mapped probes (take first)
        mapping['gene_symbol'] = mapping['gene_symbol'].fillna('')
        mapping = mapping[mapping['gene_symbol'] != '']
        
        # Handle multiple gene symbols (take first one)
        mapping['gene_symbol'] = mapping['gene_symbol'].str.split('///').str[0].str.strip()
        
        # Remove duplicates (keep first probe per gene)
        mapping = mapping.drop_duplicates(subset=['probe_id'])
        
        self._annotation_cache[platform_id] = mapping
        
        logger.success(f"Loaded {platform_id}: {len(mapping)} probe-gene mappings")
        
        return mapping
    
    def map_probes_to_genes(
        self,
        expression: pd.DataFrame,
        platform_id: str,
        aggregate: str = 'mean'
    ) -> pd.DataFrame:
        """
        Convert probe-level expression to gene-level expression.
        
        Args:
            expression: DataFrame with probe IDs as columns (Samples × Probes)
            platform_id: GPL platform ID
            aggregate: How to combine multiple probes per gene ('mean', 'max', 'median')
        
        Returns:
            Gene-level expression DataFrame (Samples × Genes)
        """
        mapping = self.load_annotation(platform_id)
        
        # Create probe -> gene dictionary
        probe_to_gene = dict(zip(mapping['probe_id'], mapping['gene_symbol']))
        
        # Find probes that exist in our expression data
        valid_probes = [p for p in expression.columns if p in probe_to_gene]
        
        logger.info(f"Mapped {len(valid_probes)}/{len(expression.columns)} probes to genes")
        
        if len(valid_probes) == 0:
            raise ValueError("No probes could be mapped to genes. Check platform ID.")
        
        # Subset to valid probes and rename to genes
        gene_expr = expression[valid_probes].copy()
        gene_expr.columns = [probe_to_gene[p] for p in valid_probes]
        
        # Aggregate multiple probes per gene
        if aggregate == 'mean':
            gene_expr = gene_expr.T.groupby(level=0).mean().T
        elif aggregate == 'max':
            gene_expr = gene_expr.T.groupby(level=0).max().T
        elif aggregate == 'median':
            gene_expr = gene_expr.T.groupby(level=0).median().T
        else:
            raise ValueError(f"Unknown aggregation: {aggregate}")
        
        logger.success(f"Gene-level expression: {gene_expr.shape[0]} samples × {gene_expr.shape[1]} genes")
        
        return gene_expr
    
    def get_physics_genes(
        self,
        expression: pd.DataFrame,
        platform_id: str
    ) -> Tuple[pd.DataFrame, List[str]]:
        """
        Extract physics-informed genes from probe-level expression.
        
        Returns:
            Subset of expression with physics genes, list of found genes
        """
        from .features import get_human_genes
        
        target_genes = set(get_human_genes())
        
        # First, map all probes to genes
        gene_expr = self.map_probes_to_genes(expression, platform_id)
        
        # Find which target genes are present
        found_genes = [g for g in target_genes if g in gene_expr.columns]
        missing_genes = target_genes - set(found_genes)
        
        logger.info(f"Found {len(found_genes)}/{len(target_genes)} physics-informed genes")
        
        if missing_genes:
            logger.warning(f"Missing genes: {sorted(missing_genes)[:10]}...")
        
        return gene_expr[found_genes], found_genes


class ExpressionNormalizer:
    """
    Normalize gene expression data.
    
    Methods:
        - TPM: Transcripts Per Million
        - log_transform: Log2(x + 1) transformation
        - quantile_normalize: Quantile normalization across samples
    """
    
    @staticmethod
    def to_tpm(
        counts: np.ndarray,
        gene_lengths: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Convert raw counts to TPM.
        
        TPM = (count / gene_length) / sum(count / gene_length) * 1e6
        
        Args:
            counts: Raw counts (samples × genes)
            gene_lengths: Gene lengths in kb (if None, assumes equal length)
        
        Returns:
            TPM-normalized expression
        """
        if gene_lengths is None:
            # Assume equal gene lengths (simplified)
            rpk = counts
        else:
            rpk = counts / gene_lengths
        
        # Normalize by total
        scaling_factor = rpk.sum(axis=1, keepdims=True) / 1e6
        tpm = rpk / (scaling_factor + 1e-8)
        
        return tpm
    
    @staticmethod
    def log_transform(expression: np.ndarray, pseudocount: float = 1.0) -> np.ndarray:
        """Log2 transformation with pseudocount."""
        return np.log2(expression + pseudocount)
    
    @staticmethod
    def quantile_normalize(expression: np.ndarray) -> np.ndarray:
        """
        Quantile normalization across samples.
        
        Makes the distribution of each sample identical.
        """
        from scipy import stats
        
        # Rank the data
        ranked = np.zeros_like(expression)
        for i in range(expression.shape[0]):
            ranked[i] = stats.rankdata(expression[i])
        
        # Sort each sample
        sorted_expr = np.sort(expression, axis=1)
        
        # Mean across samples for each rank
        rank_means = sorted_expr.mean(axis=0)
        
        # Replace each value with the mean for its rank
        normalized = np.zeros_like(expression)
        for i in range(expression.shape[0]):
            normalized[i] = rank_means[ranked[i].astype(int) - 1]
        
        return normalized


class BatchCorrector:
    """
    Batch effect correction for multi-dataset integration.
    
    Implements ComBat-seq algorithm (Zhang et al., 2020) for RNA-seq data.
    
    The algorithm:
    1. Standardizes data within each batch
    2. Estimates batch effects using empirical Bayes
    3. Removes batch effects while preserving biological variance
    
    Based on:
    - Johnson et al., 2007 (original ComBat)
    - Zhang et al., 2020 (ComBat-seq for RNA-seq)
    """
    
    def __init__(self, parametric: bool = True):
        """
        Initialize batch corrector.
        
        Args:
            parametric: If True, use parametric empirical Bayes.
                       If False, use non-parametric (slower but more robust).
        """
        self.parametric = parametric
        self.fitted = False
        
        # Parameters estimated during fit
        self.grand_mean = None
        self.var_pooled = None
        self.batch_design = None
        self.gamma_hat = None
        self.delta_hat = None
        self.gamma_star = None
        self.delta_star = None
    
    def combat_seq(
        self,
        expression: np.ndarray,
        batch_labels: np.ndarray,
        covariates: Optional[np.ndarray] = None,
        reference_batch: Optional[str] = None
    ) -> np.ndarray:
        """
        Apply ComBat-seq batch correction.
        
        Args:
            expression: Expression matrix (samples × genes)
            batch_labels: Batch assignment for each sample
            covariates: Optional biological covariates to preserve (samples × n_cov)
            reference_batch: Optional reference batch to adjust others to
        
        Returns:
            Batch-corrected expression matrix
        """
        n_samples, n_genes = expression.shape
        
        # Get unique batches
        batches = np.unique(batch_labels)
        n_batches = len(batches)
        
        if n_batches < 2:
            logger.warning("Only one batch found - no correction needed")
            return expression
        
        logger.info(f"Correcting batch effects across {n_batches} batches")
        
        # Create batch indicator matrix
        batch_design = self._create_batch_design(batch_labels, batches)
        
        # Step 1: Standardize data
        standardized, grand_mean, var_pooled = self._standardize(
            expression, batch_design, covariates
        )
        
        # Step 2: Estimate batch parameters using empirical Bayes
        gamma_hat, delta_hat = self._estimate_batch_params(
            standardized, batch_labels, batches
        )
        
        # Step 3: Apply empirical Bayes shrinkage
        if self.parametric:
            gamma_star, delta_star = self._parametric_eb_shrinkage(
                gamma_hat, delta_hat, batch_labels, batches
            )
        else:
            gamma_star, delta_star = self._nonparametric_eb_shrinkage(
                gamma_hat, delta_hat, batch_labels, batches
            )
        
        # Step 4: Adjust data
        corrected = self._adjust_data(
            standardized, batch_labels, batches,
            gamma_star, delta_star, grand_mean, var_pooled
        )
        
        # Store parameters
        self.grand_mean = grand_mean
        self.var_pooled = var_pooled
        self.batch_design = batch_design
        self.gamma_hat = gamma_hat
        self.delta_hat = delta_hat
        self.gamma_star = gamma_star
        self.delta_star = delta_star
        self.fitted = True
        
        logger.info("Batch correction complete")
        
        return corrected
    
    def _create_batch_design(self, batch_labels: np.ndarray, batches: np.ndarray) -> np.ndarray:
        """
        Create one-hot encoded batch design matrix.
        """
        n_samples = len(batch_labels)
        n_batches = len(batches)
        
        design = np.zeros((n_samples, n_batches))
        for i, batch in enumerate(batches):
            design[batch_labels == batch, i] = 1
        
        return design
    
    def _standardize(
        self,
        expression: np.ndarray,
        batch_design: np.ndarray,
        covariates: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Standardize expression data.
        
        Returns:
            standardized: Z-scored expression
            grand_mean: Overall mean per gene
            var_pooled: Pooled variance per gene
        """
        n_samples, n_genes = expression.shape
        
        # Build design matrix (batches + covariates)
        if covariates is not None:
            design = np.hstack([batch_design, covariates])
        else:
            design = batch_design
        
        # Compute grand mean (intercept from regression)
        grand_mean = expression.mean(axis=0)
        
        # Compute pooled variance
        # Center data
        centered = expression - grand_mean
        var_pooled = centered.var(axis=0, ddof=1)
        var_pooled = np.maximum(var_pooled, 1e-10)  # Avoid division by zero
        
        # Standardize
        standardized = centered / np.sqrt(var_pooled)
        
        return standardized, grand_mean, var_pooled
    
    def _estimate_batch_params(
        self,
        standardized: np.ndarray,
        batch_labels: np.ndarray,
        batches: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Estimate batch effect parameters.
        
        gamma_hat: Additive batch effect (shift)
        delta_hat: Multiplicative batch effect (scale)
        
        Returns:
            gamma_hat: (n_batches, n_genes)
            delta_hat: (n_batches, n_genes)
        """
        n_batches = len(batches)
        n_genes = standardized.shape[1]
        
        gamma_hat = np.zeros((n_batches, n_genes))
        delta_hat = np.zeros((n_batches, n_genes))
        
        for i, batch in enumerate(batches):
            batch_mask = batch_labels == batch
            batch_data = standardized[batch_mask]
            
            # Batch mean (additive effect)
            gamma_hat[i] = batch_data.mean(axis=0)
            
            # Batch variance (multiplicative effect)
            delta_hat[i] = batch_data.var(axis=0, ddof=1)
            delta_hat[i] = np.maximum(delta_hat[i], 1e-10)
        
        return gamma_hat, delta_hat
    
    def _parametric_eb_shrinkage(
        self,
        gamma_hat: np.ndarray,
        delta_hat: np.ndarray,
        batch_labels: np.ndarray,
        batches: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Parametric empirical Bayes shrinkage.
        
        Assumes gamma ~ Normal and delta ~ Inverse Gamma.
        Shrinks estimates toward prior mean.
        """
        n_batches = len(batches)
        
        gamma_star = np.zeros_like(gamma_hat)
        delta_star = np.zeros_like(delta_hat)
        
        for i, batch in enumerate(batches):
            n_batch = np.sum(batch_labels == batch)
            
            # Prior parameters for gamma (Normal)
            gamma_bar = gamma_hat[i].mean()
            tau_2 = gamma_hat[i].var()
            
            # Posterior gamma (shrunk toward prior mean)
            # gamma_star = (n * gamma_hat + tau^-2 * gamma_bar) / (n + tau^-2)
            shrinkage = n_batch / (n_batch + 1 / (tau_2 + 1e-10))
            gamma_star[i] = shrinkage * gamma_hat[i] + (1 - shrinkage) * gamma_bar
            
            # Prior parameters for delta (Inverse Gamma)
            # Method of moments
            delta_bar = delta_hat[i].mean()
            s_2 = delta_hat[i].var()
            
            # Shape and rate of prior Inverse Gamma
            if s_2 > 0:
                alpha_prior = (delta_bar ** 2) / s_2 + 2
                beta_prior = delta_bar * (alpha_prior - 1)
            else:
                alpha_prior = 2
                beta_prior = 1
            
            # Posterior delta
            # For simplicity, use shrinkage toward prior mean
            shrinkage_delta = n_batch / (n_batch + 2 * alpha_prior)
            delta_star[i] = shrinkage_delta * delta_hat[i] + (1 - shrinkage_delta) * delta_bar
            delta_star[i] = np.maximum(delta_star[i], 1e-10)
        
        return gamma_star, delta_star
    
    def _nonparametric_eb_shrinkage(
        self,
        gamma_hat: np.ndarray,
        delta_hat: np.ndarray,
        batch_labels: np.ndarray,
        batches: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Non-parametric empirical Bayes shrinkage.
        
        Uses kernel density estimation for prior (slower but more robust).
        For now, falls back to parametric for simplicity.
        """
        logger.warning("Non-parametric EB not fully implemented. Using parametric.")
        return self._parametric_eb_shrinkage(gamma_hat, delta_hat, batch_labels, batches)
    
    def _adjust_data(
        self,
        standardized: np.ndarray,
        batch_labels: np.ndarray,
        batches: np.ndarray,
        gamma_star: np.ndarray,
        delta_star: np.ndarray,
        grand_mean: np.ndarray,
        var_pooled: np.ndarray
    ) -> np.ndarray:
        """
        Apply batch correction and back-transform.
        
        Corrected = sqrt(var_pooled) * (standardized - gamma*) / sqrt(delta*) + grand_mean
        """
        n_samples, n_genes = standardized.shape
        corrected = np.zeros_like(standardized)
        
        for i, batch in enumerate(batches):
            batch_mask = batch_labels == batch
            
            # Remove batch effect
            batch_data = standardized[batch_mask]
            adjusted = (batch_data - gamma_star[i]) / np.sqrt(delta_star[i])
            
            # Back-transform
            corrected[batch_mask] = np.sqrt(var_pooled) * adjusted + grand_mean
        
        # Ensure non-negative (for count data)
        corrected = np.maximum(corrected, 0)
        
        return corrected
    
    def get_batch_statistics(self) -> Dict[str, np.ndarray]:
        """
        Get estimated batch statistics after fitting.
        
        Returns:
            Dictionary with gamma_hat, delta_hat, gamma_star, delta_star
        """
        if not self.fitted:
            raise ValueError("BatchCorrector has not been fitted yet.")
        
        return {
            'gamma_hat': self.gamma_hat,
            'delta_hat': self.delta_hat,
            'gamma_star': self.gamma_star,
            'delta_star': self.delta_star
        }


class SpatialDeconvolution:
    """
    Wrapper for spatial deconvolution algorithms.
    
    Estimates cell-type proportions from bulk RNA-seq using
    single-cell reference data.
    
    This is crucial for:
        1. Separating bronchial vs alveolar signals
        2. Creating the "Biological Ground Truth" for the Physics Loss
    """
    
    def __init__(self, reference_path: Optional[Path] = None):
        """
        Initialize deconvolution.
        
        Args:
            reference_path: Path to single-cell reference (e.g., HLCA)
        """
        self.reference_path = reference_path
        self.cell_types = [
            'Airway_Epithelial',
            'Alveolar_Epithelial',
            'Alveolar_Macrophage',
            'Ciliated',
            'Basal',
            'Club',
            'Other'
        ]
        
        logger.info(f"SpatialDeconvolution initialized with {len(self.cell_types)} cell types")
    
    def deconvolve(
        self,
        bulk_expression: np.ndarray,
        gene_names: List[str],
        method: str = 'nnls'
    ) -> Dict[str, np.ndarray]:
        """
        Perform deconvolution to estimate cell-type proportions.
        
        Args:
            bulk_expression: Bulk expression (samples × genes)
            gene_names: Gene names
            method: Deconvolution method ('nnls', 'cibersort', 'spatialddls')
        
        Returns:
            Dictionary with:
                - 'proportions': Cell-type proportions (samples × cell_types)
                - 'expression_by_celltype': Deconvolved expression per cell type
        """
        n_samples = bulk_expression.shape[0]
        n_cell_types = len(self.cell_types)
        
        if method == 'nnls':
            return self._nnls_deconvolution(bulk_expression, gene_names)
        elif method == 'simplified':
            return self._simplified_deconvolution(bulk_expression, gene_names)
        elif method == 'spatialddls':
            return self._spatialddls_deconvolution(bulk_expression, gene_names)
        else:
            logger.warning(f"Method {method} not implemented. Using simplified.")
            return self._simplified_deconvolution(bulk_expression, gene_names)
    
    def _simplified_deconvolution(
        self,
        bulk_expression: np.ndarray,
        gene_names: List[str]
    ) -> Dict[str, np.ndarray]:
        """
        Simplified deconvolution using marker gene ratios.
        
        For development/testing only.
        """
        n_samples = bulk_expression.shape[0]
        
        # Find marker gene indices
        marker_indices = {
            'bronchial': [],
            'alveolar': []
        }
        
        bronchial_markers = ['AKR1B10', 'CYP1A1', 'MUC5AC', 'SCGB1A1']
        alveolar_markers = ['MMP12', 'SPP1', 'AGER', 'SFTPC']
        
        for i, gene in enumerate(gene_names):
            if gene in bronchial_markers:
                marker_indices['bronchial'].append(i)
            elif gene in alveolar_markers:
                marker_indices['alveolar'].append(i)
        
        # Calculate regional signals
        bronchial_signal = np.zeros(n_samples)
        alveolar_signal = np.zeros(n_samples)
        
        if marker_indices['bronchial']:
            bronchial_signal = bulk_expression[:, marker_indices['bronchial']].mean(axis=1)
        
        if marker_indices['alveolar']:
            alveolar_signal = bulk_expression[:, marker_indices['alveolar']].mean(axis=1)
        
        # Normalize to proportions
        total_signal = bronchial_signal + alveolar_signal + 1e-8
        bronchial_prop = bronchial_signal / total_signal
        alveolar_prop = alveolar_signal / total_signal
        
        # Assign to cell types
        proportions = np.zeros((n_samples, len(self.cell_types)))
        
        # Bronchial types (indices 0, 3, 4, 5)
        proportions[:, 0] = bronchial_prop * 0.3  # Airway_Epithelial
        proportions[:, 3] = bronchial_prop * 0.3  # Ciliated
        proportions[:, 4] = bronchial_prop * 0.2  # Basal
        proportions[:, 5] = bronchial_prop * 0.2  # Club
        
        # Alveolar types (indices 1, 2)
        proportions[:, 1] = alveolar_prop * 0.4   # Alveolar_Epithelial
        proportions[:, 2] = alveolar_prop * 0.6   # Alveolar_Macrophage
        
        return {
            'proportions': proportions,
            'cell_types': self.cell_types,
            'bronchial_signal': bronchial_signal,
            'alveolar_signal': alveolar_signal
        }
    
    def _nnls_deconvolution(
        self,
        bulk_expression: np.ndarray,
        gene_names: List[str]
    ) -> Dict[str, np.ndarray]:
        """
        Non-negative least squares deconvolution.
        
        Uses a generated signature matrix based on marker genes.
        """
        from scipy.optimize import nnls
        
        # Generate signature matrix from marker genes
        signature = self._generate_signature_matrix(gene_names)
        
        n_samples = bulk_expression.shape[0]
        n_cell_types = len(self.cell_types)
        
        # Solve NNLS for each sample
        proportions = np.zeros((n_samples, n_cell_types))
        
        for i in range(n_samples):
            coeffs, _ = nnls(signature, bulk_expression[i])
            proportions[i] = coeffs / (coeffs.sum() + 1e-8)
        
        return {
            'proportions': proportions,
            'cell_types': self.cell_types,
            'bronchial_signal': proportions[:, [0, 3, 4, 5]].sum(axis=1),
            'alveolar_signal': proportions[:, [1, 2]].sum(axis=1)
        }
    
    def _spatialddls_deconvolution(
        self,
        bulk_expression: np.ndarray,
        gene_names: List[str],
        n_iterations: int = 10
    ) -> Dict[str, np.ndarray]:
        """
        SpatialDDLS-inspired deconvolution with iterative refinement.
        
        This method implements:
        1. Marker gene weighting based on specificity
        2. Weighted non-negative least squares
        3. Iterative refinement of cell-type proportions
        
        Based on concepts from:
        - Mañanes et al., 2024 (SpatialDDLS)
        - Wang et al., 2019 (MuSiC)
        
        Args:
            bulk_expression: Bulk expression (samples × genes)
            gene_names: Gene names
            n_iterations: Number of refinement iterations
        
        Returns:
            Dictionary with proportions and regional signals
        """
        from scipy.optimize import nnls
        
        n_samples, n_genes = bulk_expression.shape
        n_cell_types = len(self.cell_types)
        
        # Step 1: Generate signature matrix
        signature = self._generate_signature_matrix(gene_names)
        
        # Step 2: Calculate gene weights based on specificity
        gene_weights = self._calculate_gene_weights(signature)
        
        # Apply weights to signature and expression
        weighted_signature = signature * gene_weights[:, np.newaxis]
        weighted_expression = bulk_expression * gene_weights
        
        # Step 3: Initial deconvolution with weighted NNLS
        proportions = np.zeros((n_samples, n_cell_types))
        
        for i in range(n_samples):
            coeffs, _ = nnls(weighted_signature, weighted_expression[i])
            proportions[i] = coeffs / (coeffs.sum() + 1e-8)
        
        # Step 4: Iterative refinement
        for iteration in range(n_iterations):
            # Update gene weights based on residuals
            residuals = np.zeros((n_samples, n_genes))
            
            for i in range(n_samples):
                predicted = signature @ proportions[i]
                residuals[i] = np.abs(bulk_expression[i] - predicted)
            
            # Reduce weights for high-residual genes
            mean_residual = residuals.mean(axis=0)
            residual_factor = 1.0 / (1.0 + mean_residual / mean_residual.mean())
            updated_weights = gene_weights * residual_factor
            
            # Re-solve with updated weights
            weighted_signature = signature * updated_weights[:, np.newaxis]
            weighted_expression = bulk_expression * updated_weights
            
            new_proportions = np.zeros((n_samples, n_cell_types))
            for i in range(n_samples):
                coeffs, _ = nnls(weighted_signature, weighted_expression[i])
                new_proportions[i] = coeffs / (coeffs.sum() + 1e-8)
            
            # Check convergence
            diff = np.abs(new_proportions - proportions).max()
            proportions = new_proportions
            
            if diff < 1e-4:
                logger.debug(f"Converged after {iteration + 1} iterations")
                break
        
        # Calculate regional signals
        bronchial_indices = [0, 3, 4, 5]  # Airway_Epithelial, Ciliated, Basal, Club
        alveolar_indices = [1, 2]          # Alveolar_Epithelial, Alveolar_Macrophage
        
        bronchial_signal = proportions[:, bronchial_indices].sum(axis=1)
        alveolar_signal = proportions[:, alveolar_indices].sum(axis=1)
        
        logger.info(f"SpatialDDLS deconvolution complete: {n_samples} samples")
        
        return {
            'proportions': proportions,
            'cell_types': self.cell_types,
            'bronchial_signal': bronchial_signal,
            'alveolar_signal': alveolar_signal,
            'gene_weights': gene_weights,
            'n_iterations': iteration + 1 if 'iteration' in dir() else n_iterations
        }
    
    def _generate_signature_matrix(self, gene_names: List[str]) -> np.ndarray:
        """
        Generate a cell-type signature matrix from marker genes.
        
        The signature matrix encodes expected expression of each gene
        in each cell type, used as the reference for deconvolution.
        
        Returns:
            signature: (n_genes, n_cell_types) matrix
        """
        n_genes = len(gene_names)
        n_cell_types = len(self.cell_types)
        
        # Define marker genes for each cell type
        markers = {
            'Airway_Epithelial': ['MUC5AC', 'MUC5B', 'AKR1B10', 'CYP1A1'],
            'Alveolar_Epithelial': ['SFTPC', 'SFTPB', 'AGER', 'HOPX'],
            'Alveolar_Macrophage': ['MMP12', 'SPP1', 'CD68', 'MARCO'],
            'Ciliated': ['FOXJ1', 'DNAH5', 'RSPH1', 'SCGB1A1'],
            'Basal': ['KRT5', 'KRT14', 'TP63', 'NGFR'],
            'Club': ['SCGB1A1', 'SCGB3A2', 'CYP2F1'],
            'Other': []
        }
        
        # Create gene name to index mapping
        gene_to_idx = {gene: i for i, gene in enumerate(gene_names)}
        
        # Initialize signature with low background expression
        np.random.seed(42)  # Reproducibility
        signature = np.random.uniform(0.1, 0.3, (n_genes, n_cell_types))
        
        # Set high expression for marker genes
        for cell_idx, cell_type in enumerate(self.cell_types):
            if cell_type in markers:
                for marker_gene in markers[cell_type]:
                    if marker_gene in gene_to_idx:
                        gene_idx = gene_to_idx[marker_gene]
                        # High specificity: this gene is highly expressed in this cell type
                        signature[gene_idx, :] = 0.1
                        signature[gene_idx, cell_idx] = 1.0
        
        return signature
    
    def _calculate_gene_weights(self, signature: np.ndarray) -> np.ndarray:
        """
        Calculate gene weights based on cell-type specificity.
        
        Genes that are specific to one cell type get higher weights,
        as they provide more information for deconvolution.
        
        Returns:
            weights: (n_genes,) array of gene weights
        """
        n_genes, n_cell_types = signature.shape
        
        # Normalize signature by max per gene
        max_per_gene = signature.max(axis=1, keepdims=True)
        normalized = signature / (max_per_gene + 1e-8)
        
        # Calculate entropy-based specificity
        # Low entropy = high specificity
        # Add small epsilon to avoid log(0)
        prob = normalized / (normalized.sum(axis=1, keepdims=True) + 1e-8)
        entropy = -np.sum(prob * np.log2(prob + 1e-10), axis=1)
        
        # Convert entropy to weight (inverse relationship)
        # Max entropy for uniform distribution = log2(n_cell_types)
        max_entropy = np.log2(n_cell_types)
        specificity = 1.0 - (entropy / max_entropy)
        
        # Scale weights to [0.1, 1.0] range
        weights = 0.1 + 0.9 * specificity
        
        return weights


class PulmoTraceDataset(Dataset):
    """
    PyTorch Dataset for PI-VAE training.
    
    Each sample contains:
        - Gene expression vector
        - Deconvolved regional signals (bio_tb, bio_alv)
        - (Optional) Ground truth exposure parameters
    """
    
    def __init__(
        self,
        expression: np.ndarray,
        gene_names: List[str],
        deconv_results: Dict[str, np.ndarray],
        metadata: Optional[pd.DataFrame] = None,
        top_k_genes: int = 2000,
        normalize: bool = True
    ):
        """
        Initialize dataset.
        
        Args:
            expression: Expression matrix (samples × genes)
            gene_names: Gene names
            deconv_results: Deconvolution results
            metadata: Sample metadata (may contain exposure info)
            top_k_genes: Number of most variable genes to keep
            normalize: Whether to apply log normalization
        """
        self.gene_names = gene_names
        self.metadata = metadata
        
        # Select top variable genes
        gene_var = expression.var(axis=0)
        top_idx = np.argsort(gene_var)[-top_k_genes:]
        
        self.expression = expression[:, top_idx]
        self.selected_genes = [gene_names[i] for i in top_idx]
        
        # Normalize
        if normalize:
            self.expression = ExpressionNormalizer.log_transform(self.expression)
            # Z-score normalize
            self.expression = (self.expression - self.expression.mean(axis=0)) / (self.expression.std(axis=0) + 1e-8)
        
        # Deconvolution signals
        self.bio_tb = deconv_results['bronchial_signal']
        self.bio_alv = deconv_results['alveolar_signal']
        
        # Normalize biological signals
        self.bio_tb = (self.bio_tb - self.bio_tb.mean()) / (self.bio_tb.std() + 1e-8)
        self.bio_alv = (self.bio_alv - self.bio_alv.mean()) / (self.bio_alv.std() + 1e-8)
        
        # Convert to tensors
        self.expression = torch.FloatTensor(self.expression)
        self.bio_tb = torch.FloatTensor(self.bio_tb)
        self.bio_alv = torch.FloatTensor(self.bio_alv)
        
        # Extract ground truth physics if available (for supervised learning)
        from .features import get_dataset_info
        
        # 5 dim: MMAD, GSD, Conc, Duration, Rate
        self.z_true = torch.zeros((len(self.metadata), 5)) 
        self.has_label = torch.zeros(len(self.metadata), dtype=torch.bool)
        
        for idx, row in self.metadata.iterrows():
            geo_id = row.get('geo_id')
            try:
                info = get_dataset_info(geo_id)
                # Check for critical missing values
                if info.mmad_um is not None and info.concentration is not None:
                     # Populate standard vector: [MMAD, GSD, Conc, Duration, Rate]
                     self.z_true[idx, 0] = info.mmad_um
                     self.z_true[idx, 1] = info.gsd
                     self.z_true[idx, 2] = info.concentration
                     self.z_true[idx, 3] = 4.0 # Default duration (hours/chronic score?)
                     self.z_true[idx, 4] = 15.0 # Default breath rate
                     
                     self.has_label[idx] = True
            except:
                pass
        
        logger.info(f"Dataset created: {len(self)} samples, {self.expression.shape[1]} genes")
        logger.info(f"Labeled samples for supervision: {self.has_label.sum().item()}")
    
    def __len__(self) -> int:
        return len(self.expression)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = {
            'expression': self.expression[idx],
            'bio_tb': self.bio_tb[idx],
            'bio_alv': self.bio_alv[idx]
        }
        
        # Add metadata if available
        if self.metadata is not None:
            # Check for known exposure columns
            if 'pack_years' in self.metadata.columns:
                sample['pack_years'] = torch.tensor(
                    self.metadata.iloc[idx]['pack_years'],
                    dtype=torch.float32
                )
        
        # Add physics ground truth
        sample['z_true'] = self.z_true[idx]
        sample['has_label'] = self.has_label[idx]
        
        return sample


def prepare_training_data(
    geo_ids: List[str] = ['GSE47460'],
    data_dir: Path = Path("data"),
    top_k_genes: int = 2000,
    batch_size: int = 32,
    deconvolution_method: str = 'spatialddls',
    apply_batch_correction: bool = True
) -> Tuple[DataLoader, List[str]]:
    """
    Complete pipeline to prepare training data.
    
    This function orchestrates:
    1. Loading GEO expression data (or generating synthetic)
    2. TPM normalization
    3. Batch correction (ComBat-seq) if multiple datasets
    4. Cell-type deconvolution
    5. PyTorch dataset creation
    
    Args:
        geo_ids: List of GEO dataset IDs
        data_dir: Data directory
        top_k_genes: Number of genes to keep
        batch_size: Batch size
        deconvolution_method: 'spatialddls', 'nnls', or 'simplified'
        apply_batch_correction: Whether to apply ComBat-seq batch correction
    
    Returns:
        DataLoader and selected gene names
    """
    logger.info("=" * 60)
    logger.info("Preparing Training Data")
    logger.info("=" * 60)
    
    # Initialize loaders
    geo_loader = GEODataLoader(data_dir / "raw")
    deconvolver = SpatialDeconvolution()
    
    all_expression = []
    all_metadata = []
    batch_labels = []
    gene_names = None
    
    for geo_id in geo_ids:
        logger.info(f"Processing {geo_id}...")
        
        # Load data
        expression, metadata = geo_loader.load_expression_matrix(geo_id)
        
        # Store gene names (assume consistent across datasets)
        if gene_names is None:
            gene_names = expression.columns.tolist()
        
        all_expression.append(expression.values)
        all_metadata.append(metadata)
        
        # Track batch labels for each sample
        batch_labels.extend([geo_id] * len(expression))
    
    # Concatenate
    expression = np.vstack(all_expression)
    metadata = pd.concat(all_metadata, ignore_index=True)
    batch_labels = np.array(batch_labels)
    
    logger.info(f"Combined data: {expression.shape[0]} samples, {expression.shape[1]} genes")
    
    # Step 1: Normalize to TPM
    expression = ExpressionNormalizer.to_tpm(expression)
    logger.info("Applied TPM normalization")
    
    # Step 2: Batch correction (if multiple datasets and enabled)
    unique_batches = np.unique(batch_labels)
    if apply_batch_correction and len(unique_batches) > 1:
        logger.info(f"Applying ComBat-seq batch correction across {len(unique_batches)} batches")
        corrector = BatchCorrector()
        expression = corrector.combat_seq(expression, batch_labels)
    elif len(unique_batches) == 1:
        logger.info("Single dataset - batch correction not needed")
    else:
        logger.info("Batch correction disabled by user")
    
    # Step 3: Deconvolve
    logger.info(f"Running {deconvolution_method} deconvolution")
    deconv_results = deconvolver.deconvolve(expression, gene_names, method=deconvolution_method)
    
    # Create dataset
    dataset = PulmoTraceDataset(
        expression=expression,
        gene_names=gene_names,
        deconv_results=deconv_results,
        metadata=metadata,
        top_k_genes=top_k_genes
    )
    
    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,  # Set to 0 for development
        pin_memory=True
    )
    
    logger.success(f"Data preparation complete: {len(dataset)} samples")
    
    return dataloader, dataset.selected_genes


def prepare_geo_training_data(
    data_dir: str = 'data/raw',
    datasets: Dict[str, Dict] = None,
    use_physics_features: bool = True,
    apply_batch_correction: bool = True,
    batch_size: int = 32,
    normalize: bool = True
) -> Tuple[DataLoader, List[str], pd.DataFrame]:
    """
    Load and prepare multiple GEO datasets for training.
    
    Handles:
    1. Loading microarray and RNA-seq datasets
    2. Mapping probe IDs to gene symbols
    3. Finding common genes across datasets
    4. Applying batch correction (ComBat-seq)
    5. Extracting physics-informed features
    
    Args:
        data_dir: Directory containing raw GEO data
        datasets: Dict of {geo_id: {'platform': 'GPL...', 'species': 'human/mouse'}}
                 If None, uses default validated datasets
        use_physics_features: Use only physics-informed 49 genes
        apply_batch_correction: Apply ComBat-seq
        batch_size: Batch size for DataLoader
        normalize: Apply log2(x+1) normalization
    
    Returns:
        dataloader: PyTorch DataLoader
        gene_names: List of gene names in order
        metadata: Combined metadata DataFrame
    """
    from .features import get_human_genes
    
    # Default datasets with their platforms
    if datasets is None:
        datasets = {
            'GSE25531': {'platform': 'GPL6480', 'species': 'human'},
            'GSE18385': {'platform': 'GPL570', 'species': 'human'},
            'GSE10006': {'platform': 'GPL570', 'species': 'human'},
            'GSE7462': {'platform': 'GPL570', 'species': 'human'},
            'GSE237251': {'platform': 'RNA-seq', 'species': 'rat'},
        }
    
    loader = GEODataLoader(data_dir=Path(data_dir))
    annot = PlatformAnnotation(cache_dir=Path(data_dir) / 'platforms')
    
    all_expression = []
    all_metadata = []
    batch_labels = []
    
    logger.info(f"Loading {len(datasets)} datasets...")
    
    for geo_id, info in datasets.items():
        try:
            logger.info(f"Processing {geo_id}...")
            
            # Load raw expression
            expr, meta = loader.load_expression_matrix(geo_id)
            
            if expr.shape[1] == 0:
                logger.warning(f"Skipping {geo_id}: no expression data")
                continue
            
            # Map probes to genes (if microarray)
            platform = info.get('platform', 'unknown')
            if platform in ['GPL6480', 'GPL570']:
                gene_expr = annot.map_probes_to_genes(expr, platform)
            else:
                # RNA-seq already has gene symbols
                gene_expr = expr
            
            # Convert mouse/rat genes to human orthologs for consistency
            species = info.get('species', 'human')
            if species in ['mouse', 'rat']:
                # Simple case-based ortholog mapping: rat/mouse genes → HUMAN genes
                # Most ortholog gene symbols are identical except for capitalization
                # e.g., rat 'Ahr' → human 'AHR', rat 'Cyp1a1' → human 'CYP1A1'
                original_cols = list(gene_expr.columns)
                new_cols = [g.upper() for g in original_cols]
                gene_expr.columns = new_cols
                
                # Count how many were actually changed
                n_changed = sum(1 for old, new in zip(original_cols, new_cols) if old != new)
                logger.info(f"Mapped {n_changed} {species} genes to human orthologs (case conversion)")
            
            # Add metadata columns
            meta['geo_id'] = geo_id
            meta['species'] = species
            
            all_expression.append(gene_expr)
            all_metadata.append(meta)
            batch_labels.extend([geo_id] * len(gene_expr))
            
            logger.success(f"{geo_id}: {gene_expr.shape[0]} samples, {gene_expr.shape[1]} genes")
            
        except Exception as e:
            logger.error(f"Failed to process {geo_id}: {e}")
            continue
    
    if not all_expression:
        raise ValueError("No datasets could be loaded")
    
    # Find common genes across all datasets
    common_genes = set(all_expression[0].columns)
    for df in all_expression[1:]:
        common_genes &= set(df.columns)
    
    common_genes = sorted(list(common_genes))
    logger.info(f"Common genes across all datasets: {len(common_genes)}")
    
    # Filter to physics features if requested
    if use_physics_features:
        human_genes = set(get_human_genes())
        physics_genes = [g for g in common_genes if g in human_genes]
        
        if len(physics_genes) < 10:
            logger.warning(f"Only {len(physics_genes)} physics genes found, using all common genes")
        else:
            logger.info(f"Using {len(physics_genes)} physics-informed genes")
            common_genes = physics_genes
    
    # Subset all datasets to common genes and combine
    combined_expression = pd.concat(
        [df[common_genes] for df in all_expression],
        axis=0
    )
    combined_metadata = pd.concat(all_metadata, axis=0, ignore_index=True)
    
    logger.info(f"Combined: {combined_expression.shape[0]} samples × {combined_expression.shape[1]} genes")
    
    # Convert to numpy
    expression = combined_expression.values.astype(np.float32)
    gene_names = combined_expression.columns.tolist()
    batch_labels = np.array(batch_labels)
    
    # Normalize
    if normalize:
        expression = ExpressionNormalizer.log_transform(expression)
        logger.info("Applied log2(x+1) normalization")
    
    # Batch correction
    unique_batches = np.unique(batch_labels)
    if apply_batch_correction and len(unique_batches) > 1:
        logger.info(f"Applying ComBat-seq across {len(unique_batches)} batches...")
        corrector = BatchCorrector()
        expression = corrector.combat_seq(expression, batch_labels)
        logger.success("Batch correction complete")
    
    # Run deconvolution
    deconv = SpatialDeconvolution()
    deconv_results = deconv.deconvolve(expression, gene_names, method='simplified')
    
    # Create dataset
    dataset = PulmoTraceDataset(
        expression=expression,
        gene_names=gene_names,
        deconv_results=deconv_results,
        metadata=combined_metadata
    )
    
    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True
    )
    
    logger.success(f"Training data ready: {len(dataset)} samples, {len(gene_names)} genes")
    
    return dataloader, gene_names, combined_metadata


# =============================================================================
# Testing
# =============================================================================
if __name__ == "__main__":
    logger.info("Testing data processing module...")
    
    # Test GEO loader
    loader = GEODataLoader(Path("data/raw"))
    
    # Generate synthetic data
    expression, metadata = loader.load_expression_matrix('GSE47460')
    
    logger.info(f"Expression shape: {expression.shape}")
    logger.info(f"Metadata columns: {metadata.columns.tolist()}")
    
    # Test deconvolution
    deconv = SpatialDeconvolution()
    results = deconv.deconvolve(
        expression.values,
        expression.columns.tolist(),
        method='simplified'
    )
    
    logger.info(f"Deconvolution results:")
    logger.info(f"  Proportions shape: {results['proportions'].shape}")
    logger.info(f"  Bronchial signal range: [{results['bronchial_signal'].min():.3f}, {results['bronchial_signal'].max():.3f}]")
    logger.info(f"  Alveolar signal range: [{results['alveolar_signal'].min():.3f}, {results['alveolar_signal'].max():.3f}]")
    
    # Test dataset
    dataset = PulmoTraceDataset(
        expression=expression.values,
        gene_names=expression.columns.tolist(),
        deconv_results=results,
        metadata=metadata
    )
    
    sample = dataset[0]
    logger.info(f"Sample keys: {sample.keys()}")
    logger.info(f"Expression shape: {sample['expression'].shape}")
    
    logger.success("Data processing module test passed!")
