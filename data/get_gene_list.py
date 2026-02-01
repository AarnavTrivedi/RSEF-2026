
import sys
import os
sys.path.append('..')
from data import prepare_geo_training_data

print("Running pipeline to get gene list...")
dataloader, gene_names, metadata = prepare_geo_training_data(
    data_dir='raw',
    use_physics_features=True,
    apply_batch_correction=True,
    batch_size=32
)
print("FINAL GENE LIST:")
print(gene_names)
