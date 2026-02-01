"""
Mesh Converter - STL/VTK/OBJ to GLTF/GLB
-----------------------------------------
Converts iLADDER and other lung meshes to web-optimized format.
"""

import os
import numpy as np
from pathlib import Path

try:
    import trimesh
    TRIMESH_AVAILABLE = True
except ImportError:
    TRIMESH_AVAILABLE = False
    print("Warning: trimesh not available. Install with: pip install trimesh[easy]")


def convert_to_gltf(input_path: str, output_path: str, simplify_ratio: float = 0.1):
    """
    Convert lung mesh to web-optimized GLTF/GLB format.
    
    Args:
        input_path: Path to input mesh (STL/VTK/OBJ/PLY)
        output_path: Path to output file (.glb or .gltf)
        simplify_ratio: Target ratio for decimation (0.1 = 10% of original faces)
        
    Returns:
        dict: Mesh statistics
    """
    if not TRIMESH_AVAILABLE:
        raise ImportError("trimesh is required: pip install trimesh[easy]")
        
    print(f"[Mesh Converter] Loading: {input_path}")
    mesh = trimesh.load(input_path)
    
    # Handle multi-mesh scenes
    if isinstance(mesh, trimesh.Scene):
        mesh = mesh.dump(concatenate=True)
        
    original_faces = len(mesh.faces)
    print(f"[Mesh Converter] Original: {original_faces:,} faces, {len(mesh.vertices):,} vertices")
    
    # Simplify for web
    if simplify_ratio < 1.0:
        target_faces = int(original_faces * simplify_ratio)
        print(f"[Mesh Converter] Simplifying to ~{target_faces:,} faces...")
        mesh = mesh.simplify_quadric_decimation(target_faces)
        print(f"[Mesh Converter] Simplified: {len(mesh.faces):,} faces")
    
    # Center and normalize scale
    mesh.vertices -= mesh.centroid
    scale = 200 / mesh.bounding_box.extents.max()
    mesh.vertices *= scale
    
    # Export
    print(f"[Mesh Converter] Exporting to: {output_path}")
    mesh.export(output_path)
    
    stats = {
        'original_faces': original_faces,
        'simplified_faces': len(mesh.faces),
        'vertices': len(mesh.vertices),
        'file_size_mb': os.path.getsize(output_path) / (1024 * 1024)
    }
    
    print(f"[Mesh Converter] ✓ Complete: {stats['simplified_faces']:,} faces, {stats['file_size_mb']:.2f} MB")
    return stats


def generate_enhanced_airway(output_path: str, generations: int = 5):
    """
    Generate an enhanced procedural airway mesh based on Weibel morphometry.
    
    Args:
        output_path: Output file path (.glb)
        generations: Number of airway generations (0-5)
        
    Returns:
        trimesh.Trimesh: Generated mesh
    """
    if not TRIMESH_AVAILABLE:
        raise ImportError("trimesh is required")
    
    print(f"[Mesh Generator] Generating {generations}-generation airway tree...")
    
    # Weibel morphometry parameters
    # Generation: (diameter mm, length mm)
    weibel = {
        0: (18.0, 120.0),   # Trachea
        1: (12.2, 47.6),    # Main bronchi
        2: (8.3, 19.0),     # Lobar
        3: (5.6, 7.6),      # Segmental
        4: (4.5, 12.7),     # Subsegmental
        5: (3.5, 10.7),     # Small bronchi
    }
    
    all_meshes = []
    
    def add_branch(start, direction, gen, parent_radius):
        """Recursively add bronchial branches."""
        if gen > generations:
            return
            
        diameter, length = weibel.get(gen, (2.0, 8.0))
        radius = diameter / 2
        
        # Create tube
        end = start + direction * length
        tube = create_tube(start, end, radius, segments=16, radial=12)
        all_meshes.append(tube)
        
        if gen < generations:
            # Bifurcation: two child branches
            # Branching angle based on Weibel
            branch_angle = np.radians(25 + gen * 3)  # Increases with generation
            
            # Rotate direction for left branch
            perp = np.array([1, 0, 0]) if abs(direction[0]) < 0.9 else np.array([0, 1, 0])
            perp = np.cross(direction, perp)
            perp = perp / np.linalg.norm(perp)
            
            # Rotation matrices
            cos_a, sin_a = np.cos(branch_angle), np.sin(branch_angle)
            
            # Left branch
            left_dir = direction * cos_a + perp * sin_a
            left_dir = left_dir / np.linalg.norm(left_dir)
            add_branch(end, left_dir, gen + 1, radius)
            
            # Right branch
            right_dir = direction * cos_a - perp * sin_a
            right_dir = right_dir / np.linalg.norm(right_dir)
            add_branch(end, right_dir, gen + 1, radius)
    
    # Start with trachea (generation 0)
    trachea_start = np.array([0.0, 0.0, 0.0])
    trachea_dir = np.array([0.0, 0.0, -1.0])
    add_branch(trachea_start, trachea_dir, 0, weibel[0][0] / 2)
    
    # Combine all meshes
    combined = trimesh.util.concatenate(all_meshes)
    
    # Center
    combined.vertices -= combined.centroid
    
    # Export
    combined.export(output_path)
    
    print(f"[Mesh Generator] ✓ Generated: {len(combined.faces):,} faces, {len(combined.vertices):,} vertices")
    return combined


def create_tube(start, end, radius, segments=16, radial=12):
    """Create a tube mesh between two points."""
    direction = end - start
    length = np.linalg.norm(direction)
    direction = direction / length
    
    # Perpendicular vectors
    if abs(direction[0]) < 0.9:
        perp1 = np.cross(direction, [1, 0, 0])
    else:
        perp1 = np.cross(direction, [0, 1, 0])
    perp1 = perp1 / np.linalg.norm(perp1)
    perp2 = np.cross(direction, perp1)
    
    vertices = []
    faces = []
    
    for i in range(segments + 1):
        t = i / segments
        center = start + direction * length * t
        
        for j in range(radial):
            theta = 2 * np.pi * j / radial
            offset = radius * (np.cos(theta) * perp1 + np.sin(theta) * perp2)
            vertices.append(center + offset)
    
    for i in range(segments):
        for j in range(radial):
            i0 = i * radial + j
            i1 = i * radial + (j + 1) % radial
            i2 = (i + 1) * radial + (j + 1) % radial
            i3 = (i + 1) * radial + j
            
            faces.append([i0, i1, i2])
            faces.append([i0, i2, i3])
    
    return trimesh.Trimesh(vertices=np.array(vertices), faces=np.array(faces))


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Mesh Converter for PulmoTrace')
    parser.add_argument('--convert', type=str, help='Input mesh to convert')
    parser.add_argument('--output', type=str, help='Output path')
    parser.add_argument('--ratio', type=float, default=0.1, help='Simplification ratio')
    parser.add_argument('--generate', action='store_true', help='Generate procedural mesh')
    parser.add_argument('--generations', type=int, default=5, help='Number of generations')
    
    args = parser.parse_args()
    
    if args.generate:
        output = args.output or 'meshes/lung_procedural.glb'
        generate_enhanced_airway(output, args.generations)
    elif args.convert:
        output = args.output or args.convert.replace('.stl', '.glb')
        convert_to_gltf(args.convert, output, args.ratio)
    else:
        print("Usage:")
        print("  Generate procedural: python mesh_converter.py --generate --generations 5")
        print("  Convert iLADDER:     python mesh_converter.py --convert lung.stl --output lung.glb --ratio 0.1")
