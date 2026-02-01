"""
OpenFOAM polyMesh to GLTF Converter
------------------------------------
Extracts wall surfaces from OpenFOAM mesh and converts to web-ready GLTF.
Optimized for large CFD meshes (handles 100M+ elements).
"""

import numpy as np
import os
import re
from pathlib import Path

try:
    import trimesh
    TRIMESH_AVAILABLE = True
except ImportError:
    TRIMESH_AVAILABLE = False


def parse_openfoam_file(filepath):
    """Parse OpenFOAM ASCII file and extract data."""
    with open(filepath, 'r') as f:
        content = f.read()
    
    # Find the number of entries (e.g., "251684382" before "(")
    # Skip header and find the count
    lines = content.split('\n')
    
    # Find where data starts (after FoamFile block)
    data_start = 0
    in_header = True
    paren_count = 0
    
    for i, line in enumerate(lines):
        line = line.strip()
        if '{' in line:
            paren_count += 1
        if '}' in line:
            paren_count -= 1
            if paren_count == 0:
                in_header = False
                data_start = i + 1
                break
    
    return lines[data_start:]


def read_points(points_file):
    """Read OpenFOAM points file."""
    print("[OpenFOAM] Reading points file...")
    
    with open(points_file, 'r') as f:
        content = f.read()
    
    # Find the count and data section
    match = re.search(r'(\d+)\s*\(', content)
    if not match:
        raise ValueError("Could not find point count")
    
    count = int(match.group(1))
    print(f"[OpenFOAM] Found {count:,} points")
    
    # Extract points between ( )
    start = content.find('(', match.end() - 1) + 1
    end = content.rfind(')')
    data = content[start:end]
    
    # Parse points: (x y z)
    points = []
    for line in data.split('\n'):
        line = line.strip()
        if line.startswith('(') and line.endswith(')'):
            coords = line[1:-1].split()
            if len(coords) == 3:
                points.append([float(c) for c in coords])
    
    print(f"[OpenFOAM] Parsed {len(points):,} points")
    return np.array(points, dtype=np.float32)


def read_faces(faces_file, max_faces=None):
    """Read OpenFOAM faces file."""
    print("[OpenFOAM] Reading faces file (this may take a while)...")
    
    with open(faces_file, 'r') as f:
        content = f.read()
    
    # Find count
    match = re.search(r'(\d+)\s*\(', content)
    if not match:
        raise ValueError("Could not find face count")
    
    count = int(match.group(1))
    print(f"[OpenFOAM] Found {count:,} faces")
    
    if max_faces and count > max_faces:
        print(f"[OpenFOAM] Limiting to {max_faces:,} faces")
        count = max_faces
    
    # Extract face data
    start = content.find('(', match.end() - 1) + 1
    end = content.rfind(')')
    data = content[start:end]
    
    # Parse faces: N(v0 v1 v2 ... vN-1)
    faces = []
    for line in data.split('\n'):
        line = line.strip()
        if not line:
            continue
        
        # Format: 4(0 1 2 3) or just (0 1 2 3)
        match = re.match(r'(\d*)\(([^)]+)\)', line)
        if match:
            vertex_str = match.group(2)
            vertices = [int(v) for v in vertex_str.split()]
            faces.append(vertices)
            
            if max_faces and len(faces) >= max_faces:
                break
    
    print(f"[OpenFOAM] Parsed {len(faces):,} faces")
    return faces


def read_boundary(boundary_file):
    """Read OpenFOAM boundary file."""
    print("[OpenFOAM] Reading boundary patches...")
    
    with open(boundary_file, 'r') as f:
        content = f.read()
    
    patches = []
    
    # Parse patch definitions
    patch_pattern = r'(\w+)\s*\{\s*type\s+(\w+);[^}]*nFaces\s+(\d+);\s*startFace\s+(\d+);'
    
    for match in re.finditer(patch_pattern, content, re.DOTALL):
        name, ptype, nfaces, start = match.groups()
        patches.append({
            'name': name,
            'type': ptype,
            'nFaces': int(nfaces),
            'startFace': int(start)
        })
    
    print(f"[OpenFOAM] Found {len(patches)} boundary patches:")
    for p in patches:
        print(f"  - {p['name']}: {p['nFaces']:,} faces ({p['type']})")
    
    return patches


def extract_wall_surfaces(polymesh_dir, output_path, simplify_ratio=0.05):
    """
    Extract wall surfaces from OpenFOAM polyMesh and convert to GLTF.
    
    Args:
        polymesh_dir: Path to polyMesh directory
        output_path: Output GLTF/GLB file path
        simplify_ratio: Target ratio for decimation (0.05 = 5% of faces)
    """
    if not TRIMESH_AVAILABLE:
        raise ImportError("trimesh required: pip install trimesh[easy]")
    
    polymesh_dir = Path(polymesh_dir)
    
    # Read boundary to find wall patches
    boundaries = read_boundary(polymesh_dir / 'boundary')
    
    # Get wall patches
    wall_patches = [p for p in boundaries if p['type'] == 'wall']
    
    if not wall_patches:
        print("[OpenFOAM] No wall patches found!")
        return None
    
    total_wall_faces = sum(p['nFaces'] for p in wall_patches)
    print(f"[OpenFOAM] Total wall faces to extract: {total_wall_faces:,}")
    
    # Read all points
    points = read_points(polymesh_dir / 'points')
    
    # Read faces - only what we need for walls
    # We need faces from min(startFace) to max(startFace + nFaces)
    min_start = min(p['startFace'] for p in wall_patches)
    max_end = max(p['startFace'] + p['nFaces'] for p in wall_patches)
    
    print(f"[OpenFOAM] Reading faces {min_start:,} to {max_end:,}...")
    
    # For large files, read faces incrementally
    all_faces = read_faces_range(polymesh_dir / 'faces', min_start, max_end)
    
    # Extract wall faces
    wall_vertices = []
    wall_faces = []
    vertex_map = {}  # Global to local vertex mapping
    
    for patch in wall_patches:
        start = patch['startFace'] - min_start
        end = start + patch['nFaces']
        
        for face_idx in range(start, end):
            if face_idx >= len(all_faces):
                break
                
            face = all_faces[face_idx]
            
            # Map vertices
            local_face = []
            for global_v in face:
                if global_v not in vertex_map:
                    vertex_map[global_v] = len(wall_vertices)
                    wall_vertices.append(points[global_v])
                local_face.append(vertex_map[global_v])
            
            # Triangulate if needed (fan triangulation for convex polygons)
            if len(local_face) >= 3:
                for i in range(1, len(local_face) - 1):
                    wall_faces.append([local_face[0], local_face[i], local_face[i + 1]])
    
    print(f"[OpenFOAM] Extracted: {len(wall_vertices):,} vertices, {len(wall_faces):,} triangles")
    
    # Create mesh
    vertices = np.array(wall_vertices, dtype=np.float32)
    faces = np.array(wall_faces, dtype=np.int32)
    
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
    
    # Center and scale
    mesh.vertices -= mesh.centroid
    scale = 200 / mesh.bounding_box.extents.max()
    mesh.vertices *= scale
    
    # Simplify if too large
    if simplify_ratio < 1.0 and len(mesh.faces) > 50000:
        target = int(len(mesh.faces) * simplify_ratio)
        target = max(target, 30000)  # Minimum quality
        print(f"[OpenFOAM] Simplifying to {target:,} faces...")
        mesh = mesh.simplify_quadric_decimation(face_count=target)
    
    # Export
    mesh.export(output_path)
    
    file_size = os.path.getsize(output_path) / (1024 * 1024)
    print(f"[OpenFOAM] ✓ Exported: {output_path}")
    print(f"[OpenFOAM] Final mesh: {len(mesh.faces):,} faces, {file_size:.2f} MB")
    
    return mesh


def read_faces_range(faces_file, start_idx, end_idx):
    """Read a range of faces from OpenFOAM faces file."""
    print(f"[OpenFOAM] Reading face range {start_idx:,} to {end_idx:,}...")
    
    faces = []
    current_idx = 0
    
    with open(faces_file, 'r') as f:
        # Skip header
        in_data = False
        
        for line in f:
            line = line.strip()
            
            # Find start of data
            if not in_data:
                if line.startswith('(') and not line.startswith('//'):
                    in_data = True
                    # Check if this line has face data too
                    if len(line) > 1:
                        line = line[1:]  # Remove opening paren
                    else:
                        continue
            
            if in_data:
                if line.startswith(')'):
                    break
                
                # Skip if before our range
                if current_idx < start_idx:
                    current_idx += 1
                    continue
                
                # Stop if past our range
                if current_idx >= end_idx:
                    break
                
                # Parse face: N(v0 v1 v2 ...)
                match = re.match(r'(\d*)\(([^)]+)\)', line)
                if match:
                    vertex_str = match.group(2)
                    vertices = [int(v) for v in vertex_str.split()]
                    faces.append(vertices)
                
                current_idx += 1
    
    print(f"[OpenFOAM] Read {len(faces):,} faces")
    return faces


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='OpenFOAM polyMesh to GLTF Converter')
    parser.add_argument('--polymesh', type=str, required=True, help='Path to polyMesh directory')
    parser.add_argument('--output', type=str, default='lung_openfoam.glb', help='Output file')
    parser.add_argument('--ratio', type=float, default=0.05, help='Simplification ratio')
    
    args = parser.parse_args()
    
    extract_wall_surfaces(args.polymesh, args.output, args.ratio)
