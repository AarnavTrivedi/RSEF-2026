
"""
Procedural Airway Mesh Generator
--------------------------------
Generates a synthetic 3D airway tree based on the Weibel symmetric branching model.
Used for visualization when no patient-specific CT mesh is available.
"""

import numpy as np
import pyvista as pv

def generate_airway_tree(generations=4, branch_angle=35, length_scale=0.85, radius_scale=0.75):
    """
    Generate a branching airway tree mesh.
    
    Args:
        generations (int): Number of branching generations.
        branch_angle (float): Branching angle in degrees.
        length_scale (float): Scaling factor for length of child branches.
        radius_scale (float): Scaling factor for radius of child branches.
        
    Returns:
        pv.PolyData: Combined mesh of the airway tree.
    """
    
    # Weibel Model Dimensions (Approximate for human adult)
    # Generation 0 (Trachea): L=12cm, D=1.8cm
    l_0 = 120.0  # mm
    r_0 = 9.0    # mm
    
    meshes = []
    
    # Recursive function to build branches
    def add_branch(start_point, direction, length, radius, current_gen):
        # Create cylinder for this segment
        # PyVista Cycle is centered at (0,0,0) aligned with X axis by default? 
        # Actually typically Z. Let's use `pv.Cylinder` or `pv.Line` then tube.
        # Tube is easier for smooth joints.
        
        end_point = start_point + direction * length
        
        # Create a line and tube it
        line = pv.Line(start_point, end_point)
        tube = line.tube(radius=radius, n_sides=20, capping=False)
        
        # Add generation data for coloring later
        tube.point_data['generation'] = np.full(tube.n_points, current_gen)
        
        meshes.append(tube)
        
        if current_gen < generations:
            # Calculate new dimensions
            new_len = length * length_scale
            new_rad = radius * radius_scale
            
            # Rotation matrices for left/right branches
            # Rotate around an axis perpendicular to 'direction' and 'some_up_vector'
            # Simple approach: bifurcate in the Y-Z plane relative to current direction
            
            # Normalize direction
            d = direction / np.linalg.norm(direction)
            
            # Arbitrary perpendicular axis (unless d is parallel to x)
            if np.abs(d[0]) < 0.9:
                perp = np.cross(d, [1, 0, 0])
            else:
                perp = np.cross(d, [0, 1, 0])
            perp = perp / np.linalg.norm(perp)
            
            # Rotation Axis: perp
            angle_rad = np.radians(branch_angle)
            
            # Rodrigues rotation formula or simple matrix
            # Let's use scipy if available, or manual quaternion/matrix to be safe without scipy
            # Actually, let's just perturb the vector manually for simplicity in this demo
            
            # Branch 1 (+angle)
            # Component parallel to d: d * cos(angle)
            # Component parallel to perp: perp * sin(angle)
            d1 = d * np.cos(angle_rad) + perp * np.sin(angle_rad)
            
            # Branch 2 (-angle)
            d2 = d * np.cos(angle_rad) - perp * np.sin(angle_rad)
            
            # Twist plane for next generation (90 deg rotation) to create 3D structure
            # Not fully implemented for simplicity, keeping planar for now or random twist
            
            add_branch(end_point, d1, new_len, new_rad, current_gen + 1)
            add_branch(end_point, d2, new_len, new_rad, current_gen + 1)
            
    # Start recursion (Trachea points down -Z or similar)
    # Let's point down Y (anatomical coordinates usually Y is vertical in some contexts, or Z)
    # Let's use Z-up convention: Head at +Z, Lungs at -Z?
    # Or medical: Z is axial (Head-Feet). So trachea goes -Z.
    
    start = np.array([0.0, 0.0, 0.0]) # Larynx
    direction = np.array([0.0, 0.0, -1.0])
    
    add_branch(start, direction, l_0, r_0, 0)
    
    # Combine all meshes
    full_mesh = meshes[0]
    for m in meshes[1:]:
        full_mesh = full_mesh.merge(m)
        
    # Smooth the joints (optional, might be slow)
    # full_mesh = full_mesh.smooth(n_iter=100)
    
    return full_mesh

if __name__ == "__main__":
    print("Generating Airway Mesh...")
    mesh = generate_airway_tree()
    print(f"Generated mesh with {mesh.n_points} points and {mesh.n_cells} cells.")
    
    output_file = "airway_gen0-4.vtk"
    mesh.save(output_file)
    print(f"Saved to {output_file}")
    
    # Preview (if interactive)
    # mesh.plot()
