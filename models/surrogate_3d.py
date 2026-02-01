"""
3D CFD Surrogate Model - Physics-Informed Heuristics
-----------------------------------------------------
Maps aerosol parameters (MMAD, Flow Rate, GSD) to 3D deposition patterns.

Uses ICRP 66 / MPPD-inspired physics without requiring CFD training data:
- Impaction: Stokes number scaling at bifurcations
- Sedimentation: Gravitational settling in horizontal airways
- Diffusion: Brownian motion for ultrafine particles

Author: PulmoTrace Engineering Team
"""

import numpy as np
from typing import Optional


class Surrogate3D:
    """
    Physics-informed surrogate for 3D deposition prediction.
    
    This model uses analytical physics (not ML) to predict deposition patterns
    based on particle and airflow parameters. It's designed for real-time
    visualization feedback before full CFD training data is available.
    
    Physics Basis:
    - Stokes Number (Stk): Dimensionless number for inertial impaction
    - Peclet Number (Pe): Ratio of convective to diffusive transport
    - Settling velocity: Terminal velocity under gravity
    """
    
    # ICRP 66 Reference Constants
    RHO_PARTICLE = 1000  # kg/m³ (unit density sphere convention)
    MU_AIR = 1.81e-5     # Pa·s (air viscosity at 37°C)
    RHO_AIR = 1.1        # kg/m³
    DIFFUSION_COEF_REF = 1e-9  # m²/s reference for 1µm particle
    G = 9.81             # m/s² gravity
    
    def __init__(self, mesh_points: Optional[np.ndarray] = None):
        """
        Initialize surrogate with optional mesh geometry.
        
        Args:
            mesh_points: Nx3 array of mesh vertex coordinates [mm]
        """
        self.mesh_points = mesh_points
        self.n_points = len(mesh_points) if mesh_points is not None else 0
        
        # Pre-computed geometric features
        self._bifurcation_field = None
        self._depth_field = None
        self._horizontal_field = None
        
        if mesh_points is not None:
            self._precompute_geometry()
            
    def _precompute_geometry(self):
        """
        Compute geometric features from mesh for physics calculations.
        
        Features computed:
        1. Bifurcation proximity (high curvature regions)
        2. Airway depth (generation proxy)
        3. Horizontal orientation (for sedimentation)
        """
        if self.mesh_points is None:
            return
            
        z = self.mesh_points[:, 2]  # Depth coordinate
        x = self.mesh_points[:, 0]
        y = self.mesh_points[:, 1]
        
        # 1. Bifurcation field - peaks at known branching depths
        # For procedural mesh: bifurcations at z = -120, -222, -308, -380 mm
        self._bifurcation_field = np.zeros(self.n_points)
        
        # Weibel model bifurcation depths (cumulative)
        L0 = 120  # Trachea length mm
        scale = 0.85  # Length scaling per generation
        bifurcation_z = []
        cumulative = 0
        for gen in range(5):
            cumulative += L0 * (scale ** gen)
            bifurcation_z.append(-cumulative)
            
        # Gaussian kernels at bifurcations (wider for deeper generations)
        for i, bif_z in enumerate(bifurcation_z):
            sigma = 15 + i * 5  # Wider spread for deeper airways
            dist_sq = (z - bif_z) ** 2
            self._bifurcation_field += np.exp(-dist_sq / (2 * sigma ** 2))
            
        # Normalize to [0, 1]
        max_val = self._bifurcation_field.max()
        if max_val > 0:
            self._bifurcation_field /= max_val
            
        # 2. Depth field - normalized z-coordinate (0 = mouth, 1 = deep)
        z_min, z_max = z.min(), z.max()
        self._depth_field = (z_max - z) / (z_max - z_min + 1e-6)
        
        # 3. Horizontal field - proxy for sedimentation potential
        # Higher in regions where airways run horizontally
        # Use |x| as proxy (lateral spread = more horizontal)
        r = np.sqrt(x**2 + y**2)
        self._horizontal_field = r / (r.max() + 1e-6)
        
    def set_mesh(self, mesh_points: np.ndarray):
        """Update mesh and recompute geometry."""
        self.mesh_points = mesh_points
        self.n_points = len(mesh_points)
        self._precompute_geometry()
        
    def forward(self, mmad: float, q_flow: float, gsd: float = 1.5) -> np.ndarray:
        """
        Predict 3D deposition field.
        
        Args:
            mmad: Mass Median Aerodynamic Diameter [µm]
            q_flow: Inspiratory Flow Rate [L/min]
            gsd: Geometric Standard Deviation
            
        Returns:
            Deposition probability at each mesh vertex [0, 1]
        """
        if self.n_points == 0:
            return np.zeros(100)
            
        # Convert units
        d_p = mmad * 1e-6  # µm -> m
        Q = q_flow / 60000  # L/min -> m³/s
        
        # Estimate mean velocity in trachea (D ~ 18mm)
        D_trachea = 0.018  # m
        A_trachea = np.pi * (D_trachea / 2) ** 2
        U = Q / A_trachea  # m/s
        
        # ===== 1. IMPACTION (Stokes Number) =====
        # Stk = (ρ_p * d_p² * U) / (18 * μ * D)
        # High Stk -> high impaction at bifurcations
        
        stokes = (self.RHO_PARTICLE * d_p**2 * U) / (18 * self.MU_AIR * D_trachea)
        
        # Impaction probability ~ Stk^0.5 for realistic scaling
        # Concentrated at bifurcations
        impaction_eff = min(1.0, np.sqrt(stokes) * 2)
        deposition_impaction = self._bifurcation_field * impaction_eff
        
        # ===== 2. SEDIMENTATION (Gravitational Settling) =====
        # V_settle = (ρ_p * d_p² * g) / (18 * μ)
        # More important for larger particles in horizontal airways
        
        v_settle = (self.RHO_PARTICLE * d_p**2 * self.G) / (18 * self.MU_AIR)
        
        # Sedimentation efficiency increases with particle size and in horizontal regions
        sed_eff = min(1.0, v_settle * 100)  # Scale to reasonable range
        deposition_sedimentation = self._horizontal_field * sed_eff * 0.3
        
        # ===== 3. DIFFUSION (Brownian Motion) =====
        # D_diff ~ kT / (3πμd_p) - inversely proportional to size
        # More important for ultrafine particles (< 0.5 µm)
        
        diff_coef = self.DIFFUSION_COEF_REF * (1e-6 / (d_p + 1e-9))
        
        # Diffusion more uniform, slightly higher in smaller airways (deeper)
        diff_eff = min(0.5, diff_coef * 1e6)  # Scale appropriately
        deposition_diffusion = (1 + self._depth_field * 0.5) * diff_eff * 0.2
        
        # ===== 4. COMBINE MECHANISMS =====
        # Total deposition is sum of mechanisms (simplified additive model)
        total = deposition_impaction + deposition_sedimentation + deposition_diffusion
        
        # Apply GSD effect - broader distribution = more uniform deposition
        # GSD = 1.0 is monodisperse, GSD > 2 is highly polydisperse
        gsd_factor = 1.0 / (1 + (gsd - 1.5) * 0.2)  # Reduces peaks for high GSD
        total *= gsd_factor
        
        # Normalize and clip
        total = np.clip(total, 0.0, 1.0)
        
        return total
    
    def get_physics_summary(self, mmad: float, q_flow: float) -> dict:
        """
        Return physics parameters for display in UI.
        
        Args:
            mmad: MMAD in µm
            q_flow: Flow rate in L/min
            
        Returns:
            Dictionary of computed physics values
        """
        d_p = mmad * 1e-6
        Q = q_flow / 60000
        D_trachea = 0.018
        A_trachea = np.pi * (D_trachea / 2) ** 2
        U = Q / A_trachea
        
        stokes = (self.RHO_PARTICLE * d_p**2 * U) / (18 * self.MU_AIR * D_trachea)
        v_settle = (self.RHO_PARTICLE * d_p**2 * self.G) / (18 * self.MU_AIR)
        
        # Particle regime classification
        if mmad < 0.5:
            regime = "Diffusion-Dominated"
        elif mmad < 3:
            regime = "Mixed (Sedimentation + Diffusion)"
        elif mmad < 8:
            regime = "Impaction-Dominated"
        else:
            regime = "Heavy Particle (Nasal/Oral Capture)"
            
        return {
            "stokes_number": stokes,
            "settling_velocity_mm_s": v_settle * 1000,
            "trachea_velocity_m_s": U,
            "deposition_regime": regime,
            "reynolds_number": (self.RHO_AIR * U * D_trachea) / self.MU_AIR
        }
