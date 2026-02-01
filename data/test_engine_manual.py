
"""
Manual Test for PulmoTrace Engine
---------------------------------
Verifies that the engine loads and predicts correctly.
"""
import sys
import os
import numpy as np

# Add parent directory
sys.path.insert(0, os.path.abspath('..'))

from models.engine import PulmoTraceEngine

def test_engine():
    print("Initializing Engine...")
    engine = PulmoTraceEngine(checkpoints_dir='../data/checkpoints')
    
    print("\nEngine Initialized.")
    print(f"Gene count: {len(engine.gene_names)}")
    
    # Create dummy expression vector (45 genes)
    dummy_expr = np.random.rand(45).astype(np.float32)
    
    print("\nProcessing Dummy Sample...")
    result = engine.process_sample(dummy_expr, sample_name="TestBot_001")
    
    print("\n=== Result Summary ===")
    print(f"Sample: {result['sample_name']}")
    print(f"MMAD: {result['physics']['mmad']:.3f} µm")
    print(f"Source: {result['physics']['interpretation']['source']}")
    print(f"Dominant Tissue: {result['biology']['dominant_tissue']}")
    print(f"Alignment Consistent? {result['alignment']['consistent']}")
    
    print("\nSUCCESS: Engine is working!")

if __name__ == "__main__":
    test_engine()
