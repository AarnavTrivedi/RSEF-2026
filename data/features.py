"""
Physics-Informed Feature List for PulmoTrace PI-VAE.

Contains the 50 highly conserved genes across human and mouse
representing distinct physical response modules for reverse-dosimetry.

Based on forensic audit of GEO datasets (GSE25531, GSE18385, GSE237251, etc.)
"""

from typing import Dict, List, NamedTuple
from dataclasses import dataclass


@dataclass
class GeneOrthologs:
    """Paired human-mouse ortholog genes."""
    human: str
    mouse: str
    module: str
    mechanism: str
    physical_driver: str


# =============================================================================
# Core Physics-Informed Feature Set (50 Genes)
# =============================================================================

PHYSICS_FEATURES: List[GeneOrthologs] = [
    # --- Metabolic Module: Gas Phase (Aldehydes) ---
    GeneOrthologs("AKR1B10", "Akr1b8", "Metabolic", "Detoxification", "Gas Phase (Aldehydes)"),
    GeneOrthologs("ALDH3A1", "Aldh3a1", "Metabolic", "Aldehyde oxidation", "Gas Phase (Aldehydes)"),
    GeneOrthologs("ALDH1A1", "Aldh1a1", "Metabolic", "Aldehyde oxidation", "Gas Phase (Aldehydes)"),
    GeneOrthologs("ADH7", "Adh7", "Metabolic", "Alcohol dehydrogenase", "Gas Phase (Aldehydes)"),
    GeneOrthologs("GPX2", "Gpx2", "Metabolic", "Peroxide reduction", "Gas Phase (Aldehydes)"),
    
    # --- Xenobiotic Module: Particle Phase (PAHs) ---
    GeneOrthologs("CYP1A1", "Cyp1a1", "Xenobiotic", "Hydroxylation", "Particle Phase (PAHs)"),
    GeneOrthologs("CYP1B1", "Cyp1b1", "Xenobiotic", "Hydroxylation", "Particle Phase (PAHs)"),
    GeneOrthologs("AHR", "Ahr", "Xenobiotic", "Sensor", "Particle Phase (PAHs)"),
    GeneOrthologs("ARNT", "Arnt", "Xenobiotic", "AHR partner", "Particle Phase (PAHs)"),
    GeneOrthologs("TIPARP", "Tiparp", "Xenobiotic", "AHR repressor", "Particle Phase (PAHs)"),
    GeneOrthologs("AHRR", "Ahrr", "Xenobiotic", "AHR repressor", "Particle Phase (PAHs)"),
    GeneOrthologs("UGT1A1", "Ugt1a1", "Xenobiotic", "Glucuronidation", "Particle Phase (PAHs)"),
    GeneOrthologs("EPHX1", "Ephx1", "Xenobiotic", "Epoxide hydrolase", "Particle Phase (PAHs)"),
    
    # --- Oxidative Stress Module: Free Radicals (ROS) ---
    GeneOrthologs("HMOX1", "Hmox1", "Oxidative", "Heme degradation", "Free Radicals (ROS)"),
    GeneOrthologs("NQO1", "Nqo1", "Oxidative", "Quinone reduction", "Free Radicals (ROS)"),
    GeneOrthologs("GCLC", "Gclc", "Oxidative", "Glutathione synthesis", "Free Radicals (ROS)"),
    GeneOrthologs("GCLM", "Gclm", "Oxidative", "Glutathione synthesis", "Free Radicals (ROS)"),
    GeneOrthologs("GSR", "Gsr", "Oxidative", "Glutathione reductase", "Free Radicals (ROS)"),
    GeneOrthologs("SOD1", "Sod1", "Oxidative", "Superoxide dismutase", "Free Radicals (ROS)"),
    GeneOrthologs("SOD2", "Sod2", "Oxidative", "Superoxide dismutase", "Free Radicals (ROS)"),
    GeneOrthologs("CAT", "Cat", "Oxidative", "Catalase", "Free Radicals (ROS)"),
    GeneOrthologs("NFE2L2", "Nfe2l2", "Oxidative", "NRF2 master regulator", "Free Radicals (ROS)"),
    GeneOrthologs("KEAP1", "Keap1", "Oxidative", "NRF2 inhibitor", "Free Radicals (ROS)"),
    
    # --- Inflammatory Module: Particle Burden ---
    GeneOrthologs("CXCL8", "Cxcl1", "Inflammatory", "Neutrophil recruit (KC)", "Particle Burden"),
    GeneOrthologs("IL1B", "Il1b", "Inflammatory", "Acute phase", "Particle Burden"),
    GeneOrthologs("IL6", "Il6", "Inflammatory", "Acute phase", "Particle Burden"),
    GeneOrthologs("TNF", "Tnf", "Inflammatory", "Acute phase", "Particle Burden"),
    GeneOrthologs("S100A8", "S100a8", "Inflammatory", "DAMP/Alarmin", "Particle Burden"),
    GeneOrthologs("S100A9", "S100a9", "Inflammatory", "DAMP/Alarmin", "Particle Burden"),
    GeneOrthologs("CCL2", "Ccl2", "Inflammatory", "Monocyte recruit", "Particle Burden"),
    GeneOrthologs("CXCL2", "Cxcl2", "Inflammatory", "Neutrophil recruit", "Particle Burden"),
    GeneOrthologs("PTGS2", "Ptgs2", "Inflammatory", "COX-2", "Particle Burden"),
    GeneOrthologs("NLRP3", "Nlrp3", "Inflammatory", "Inflammasome", "Particle Burden"),
    
    # --- Structural/Remodeling Module: Injury ---
    GeneOrthologs("MMP12", "Mmp12", "Structural", "Elastase (Macrophage)", "Remodeling/Injury"),
    GeneOrthologs("MMP9", "Mmp9", "Structural", "Gelatinase", "Remodeling/Injury"),
    GeneOrthologs("MMP2", "Mmp2", "Structural", "Gelatinase", "Remodeling/Injury"),
    GeneOrthologs("SPP1", "Spp1", "Structural", "Fibrosis/Osteopontin", "Remodeling/Injury"),
    GeneOrthologs("TGFB1", "Tgfb1", "Structural", "Fibrosis master", "Remodeling/Injury"),
    GeneOrthologs("COL1A1", "Col1a1", "Structural", "Collagen", "Remodeling/Injury"),
    GeneOrthologs("TIMP1", "Timp1", "Structural", "MMP inhibitor", "Remodeling/Injury"),
    
    # --- Epithelial/Mucosal Module ---
    GeneOrthologs("MUC5AC", "Muc5ac", "Epithelial", "Goblet hyperplasia", "Mucosal Defense"),
    GeneOrthologs("MUC5B", "Muc5b", "Epithelial", "Mucin", "Mucosal Defense"),
    GeneOrthologs("SCGB1A1", "Scgb1a1", "Epithelial", "Club cell marker", "Mucosal Defense"),
    GeneOrthologs("SFTPC", "Sftpc", "Epithelial", "AT2 marker", "Alveolar"),
    GeneOrthologs("AGER", "Ager", "Epithelial", "AT1 marker/RAGE", "Alveolar"),
    
    # --- Apoptosis/DNA Damage Module ---
    GeneOrthologs("TP53", "Trp53", "Apoptosis", "Tumor suppressor", "DNA Damage"),
    GeneOrthologs("CDKN1A", "Cdkn1a", "Apoptosis", "p21 cell cycle arrest", "DNA Damage"),
    GeneOrthologs("BAX", "Bax", "Apoptosis", "Pro-apoptotic", "DNA Damage"),
    GeneOrthologs("BCL2", "Bcl2", "Apoptosis", "Anti-apoptotic", "DNA Damage"),
]

# Convenience accessors
def get_human_genes() -> List[str]:
    """Get list of human gene symbols."""
    return [g.human for g in PHYSICS_FEATURES]


def get_mouse_genes() -> List[str]:
    """Get list of mouse gene symbols."""
    return [g.mouse for g in PHYSICS_FEATURES]


def get_ortholog_map() -> Dict[str, str]:
    """Get human -> mouse ortholog mapping."""
    return {g.human: g.mouse for g in PHYSICS_FEATURES}


def get_reverse_ortholog_map() -> Dict[str, str]:
    """Get mouse -> human ortholog mapping."""
    return {g.mouse: g.human for g in PHYSICS_FEATURES}


def get_genes_by_module(module: str) -> List[GeneOrthologs]:
    """Get all genes for a specific module."""
    return [g for g in PHYSICS_FEATURES if g.module == module]


def get_module_names() -> List[str]:
    """Get unique module names."""
    return list(set(g.module for g in PHYSICS_FEATURES))


# Module-specific subsets for deconvolution
BRONCHIAL_MARKERS = ['AKR1B10', 'CYP1A1', 'MUC5AC', 'SCGB1A1', 'MUC5B']
ALVEOLAR_MARKERS = ['MMP12', 'SPP1', 'AGER', 'SFTPC', 'S100A8']


# =============================================================================
# Validated Dataset Registry
# =============================================================================

@dataclass
class DatasetMetadata:
    """Metadata for validated GEO datasets."""
    geo_id: str
    species: str
    exposure_agent: str
    delivery_method: str
    timepoint: str
    mmad_um: float  # Mass Median Aerodynamic Diameter in μm
    gsd: float      # Geometric Standard Deviation
    concentration: float  # μg/m³
    n_samples: int
    platform: str
    tissue: str
    notes: str = ""


VALIDATED_DATASETS: Dict[str, DatasetMetadata] = {
    # Human Datasets
    "GSE25531": DatasetMetadata(
        geo_id="GSE25531",
        species="Homo sapiens",
        exposure_agent="Diesel Exhaust",
        delivery_method="Controlled Chamber Inhalation",
        timepoint="Acute (24h post)",
        mmad_um=0.3,
        gsd=2.2,
        concentration=300.0,  # μg/m³ PM2.5
        n_samples=28,  # 14 subjects × 2 conditions
        platform="GPL6480 Agilent",
        tissue="PBMC",
        notes="Gold standard crossover design"
    ),
    
    "GSE18385": DatasetMetadata(
        geo_id="GSE18385",
        species="Homo sapiens",
        exposure_agent="Cigarette Smoke",
        delivery_method="Natural Inhalation",
        timepoint="Chronic",
        mmad_um=0.45,  # Wet MMAD after hygroscopic growth
        gsd=1.8,
        concentration=0.0,  # Pack-years based
        n_samples=38,
        platform="Affymetrix HG-U133",
        tissue="Bronchial Epithelium",
        notes="Smokers vs non-smokers, lobar separation"
    ),
    
    "GSE10006": DatasetMetadata(
        geo_id="GSE10006",
        species="Homo sapiens",
        exposure_agent="Smoking Cessation",
        delivery_method="Natural Inhalation (History)",
        timepoint="Recovery",
        mmad_um=0.45,
        gsd=1.8,
        concentration=0.0,
        n_samples=75,
        platform="Affymetrix HG-U133",
        tissue="Large/Small Airway Epithelium",
        notes="Spatial resolution: 3rd-4th vs 10th-12th generation"
    ),
    
    # Mouse/Rat Datasets
    "GSE153896": DatasetMetadata(
        geo_id="GSE153896",
        species="Mus musculus",
        exposure_agent="PM2.5 (Ambient)",
        delivery_method="Whole Body Chamber",
        timepoint="Chronic (3-6 months)",
        mmad_um=1.5,  # Typical urban PM2.5
        gsd=2.5,
        concentration=80.0,  # μg/m³
        n_samples=24,
        platform="Illumina",
        tissue="Lung",
        notes="Real-world urban dust composition"
    ),
    
    "GSE237251": DatasetMetadata(
        geo_id="GSE237251",
        species="Rattus norvegicus",
        exposure_agent="Eucalyptus/Wood Smoke",
        delivery_method="Whole Body Chamber",
        timepoint="Sub-Acute (2 weeks)",
        mmad_um=2.0,  # Broad range 32nm-10μm
        gsd=3.0,
        concentration=11000.0,  # 11 mg/m³
        n_samples=48,
        platform="RNA-seq",
        tissue="Lung",
        notes="Multi-omics, validated mass concentration"
    ),
    
    "GSE55270": DatasetMetadata(
        geo_id="GSE55270",
        species="Mus musculus",
        exposure_agent="Roadside Traffic PM",
        delivery_method="Environmental Exposure",
        timepoint="Chronic",
        mmad_um=0.8,  # Mixed mode
        gsd=2.5,
        concentration=50.0,  # Variable
        n_samples=20,
        platform="Affymetrix Mouse 430",
        tissue="Lung",
        notes="Complex mixture (Gas + Particle)"
    ),
}


def get_dataset_info(geo_id: str) -> DatasetMetadata:
    """Get metadata for a validated dataset."""
    if geo_id not in VALIDATED_DATASETS:
        raise ValueError(f"Dataset {geo_id} not in validated registry")
    return VALIDATED_DATASETS[geo_id]


def list_validated_datasets() -> List[str]:
    """List all validated GSE IDs."""
    return list(VALIDATED_DATASETS.keys())


def get_human_datasets() -> List[str]:
    """Get GSE IDs for human datasets."""
    return [k for k, v in VALIDATED_DATASETS.items() if v.species == "Homo sapiens"]


def get_mouse_datasets() -> List[str]:
    """Get GSE IDs for mouse/rat datasets."""
    return [k for k, v in VALIDATED_DATASETS.items() if v.species != "Homo sapiens"]
