#!/usr/bin/env python3
"""
Standalone test script for PyVista 3D rendering.
Run this to verify PyVista/VTK is working correctly.
"""

import sys
import os

# Add parent to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_pyvista_basic():
    """Test basic PyVista functionality."""
    print("=" * 50)
    print("Testing PyVista Installation")
    print("=" * 50)
    
    try:
        import pyvista as pv
        print(f"✓ PyVista version: {pv.__version__}")
    except ImportError as e:
        print(f"✗ PyVista not found: {e}")
        print("\nInstall with: pip install pyvista pyvistaqt")
        return False
        
    try:
        from pyvistaqt import QtInteractor
        print("✓ PyVistaQt available")
    except ImportError as e:
        print(f"✗ PyVistaQt not found: {e}")
        print("\nInstall with: pip install pyvistaqt")
        return False
        
    return True

def test_mesh_generation():
    """Test mesh generation."""
    print("\n" + "=" * 50)
    print("Testing Mesh Generation")
    print("=" * 50)
    
    import pyvista as pv
    import numpy as np
    
    # Simple tube test
    tube = pv.Tube(pointa=(0, 0, 0), pointb=(0, 0, -100), radius=10, resolution=20)
    print(f"✓ Created tube: {tube.n_points} points, {tube.n_cells} cells")
    
    # Test mesh generator
    try:
        from data.mesh_generator import generate_airway_tree
        mesh = generate_airway_tree(generations=3)
        print(f"✓ Generated airway tree: {mesh.n_points} points, {mesh.n_cells} cells")
        return mesh
    except Exception as e:
        print(f"✗ Mesh generator failed: {e}")
        return tube

def test_qt_integration():
    """Test Qt integration with PyVista."""
    print("\n" + "=" * 50)
    print("Testing Qt Integration")
    print("=" * 50)
    
    from PyQt6.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget
    from PyQt6.QtCore import QTimer
    import pyvista as pv
    from pyvistaqt import QtInteractor
    import numpy as np
    
    app = QApplication(sys.argv)
    
    window = QMainWindow()
    window.setWindowTitle("PyVista Test")
    window.resize(800, 600)
    
    central = QWidget()
    layout = QVBoxLayout(central)
    window.setCentralWidget(central)
    
    # Create plotter
    plotter = QtInteractor(central)
    layout.addWidget(plotter)
    
    # Add mesh
    mesh = pv.Tube(pointa=(0, 0, 0), pointb=(0, 0, -100), radius=10, resolution=20)
    mesh["test"] = np.linspace(0, 1, mesh.n_points)
    
    plotter.set_background("#1e1e1e")
    plotter.add_mesh(mesh, scalars="test", cmap="plasma", smooth_shading=True)
    plotter.view_isometric()
    plotter.reset_camera()
    
    print("✓ Qt integration successful")
    print("\nA window should appear with a 3D tube.")
    print("Close the window to exit.")
    
    window.show()
    
    # Auto-close after 5 seconds for CI
    if os.environ.get("CI"):
        QTimer.singleShot(2000, app.quit)
    
    sys.exit(app.exec())

if __name__ == "__main__":
    if not test_pyvista_basic():
        sys.exit(1)
        
    test_mesh_generation()
    test_qt_integration()
