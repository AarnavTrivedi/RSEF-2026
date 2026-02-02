
"""
Render Nature-Style Tables
--------------------------
Converts CSV result files into publication-quality high-resolution images.
Follows "Nature" style guidelines: minimalist lines, clean typography, header emphasis.
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Configuration
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'results')
TABLES_OUT_DIR = os.path.join(DATA_DIR, 'tables')
os.makedirs(TABLES_OUT_DIR, exist_ok=True)

# Style Settings
FONT_FAMILY = 'sans-serif' # or 'serif' for Times
FONT_SIZE_HEADER = 12
FONT_SIZE_BODY = 10
ROW_HEIGHT = 0.5
HEADER_HEIGHT = 0.8 # Slightly taller

def render_table(df, title, filename, top_n=None):
    """Renders a DataFrame as a high-quality table image."""
    print(f"   Rendering {filename}...")
    
    if top_n and len(df) > top_n:
        df = df.head(top_n)
        
    # Create figure
    # Estimate height: Header + Rows
    n_rows = len(df)
    n_cols = len(df.columns)
    
    # Heuristic width calc
    fig_width = max(8, n_cols * 2.0)
    fig_height = (n_rows * ROW_HEIGHT) + HEADER_HEIGHT + 1.0 # Padding
    
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    ax.axis('off')
    
    # Table Data
    table_data = [df.columns.values.tolist()] + df.values.tolist()
    
    # Create Table
    table = plt.table(cellText=table_data,
                      colLabels=None, # We include headers in data for custom styling
                      cellLoc='left',
                      loc='center',
                      bbox=[0, 0, 1, 1])
    
    table.auto_set_font_size(False)
    table.set_fontsize(FONT_SIZE_BODY)
    
    # Styling
    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor('white') # No vertical lines by default
        cell.set_linewidth(0)
        cell.set_height(ROW_HEIGHT / fig_height)
        
        # Header Styling (Row 0)
        if row == 0:
            cell.set_text_props(weight='bold', size=FONT_SIZE_HEADER)
            # Top line (Double thickness or just strong)
            # cell.set_edgecolor('black') # This draws box. We want lines.
            
            # Draw explicit lines later? No, Matplotlib table is limited. 
            # We simulate "Nature" style by drawing lines explicitly on axis.
        else:
            # Alternating rows (optional)
            # if row % 2 == 0:
            #    cell.set_facecolor('#f2f2f2')
            pass

    # Draw "Nature" style horizontal lines
    # 1. Top of Header
    ax.plot([0, 1], [1, 1], color='black', linewidth=2, transform=ax.transAxes)
    
    # 2. Bottom of Header (Row 0)
    # Height of header relative to box
    header_bottom_y = 1.0 - (1.0 / (n_rows + 1)) # Approx? No, use fixed layout
    # Better: Use Table coordinates.
    # But bbox method is easier: just drawing lines at specific Y coords
    
    # Let's rely on bbox= [0, 0, 1, 1] means table fills axis.
    # Row 0 is at top. Each row is 1/(n_rows+1) height.
    row_h_frac = 1.0 / (n_rows + 1)
    
    # Line under header
    ax.plot([0, 1], [1.0 - row_h_frac, 1.0 - row_h_frac], color='black', linewidth=1, transform=ax.transAxes)
    
    # 3. Bottom of Table
    ax.plot([0, 1], [0, 0], color='black', linewidth=2, transform=ax.transAxes)

    # Title
    # plt.title(title, loc='left', pad=20, fontsize=14, fontweight='bold')
    # Nature tables usually have title as caption text outside image, but let's include it for standalone
    # ax.text(0, 1.02, title, transform=ax.transAxes, fontsize=14, weight='bold', va='bottom')

    plt.tight_layout()
    plt.savefig(os.path.join(TABLES_OUT_DIR, filename), dpi=300, bbox_inches='tight', pad_inches=0.1)
    plt.close()

def main():
    print("🎨 Rendering Publication-Quality Tables...")
    
    # 1. Performance Metrics
    df1 = pd.read_csv(os.path.join(DATA_DIR, 'table1_performance_metrics.csv'))
    render_table(df1, "Table 1: System Performance Metrics", "table1_performance.png")
    
    # 2. Ablation Study
    df2 = pd.read_csv(os.path.join(DATA_DIR, 'table2_ablation_study.csv'))
    render_table(df2, "Table 2: Ablation Study Results", "table2_ablation.png")
    
    # 3. Forensic Validation (Top 10)
    df3 = pd.read_csv(os.path.join(DATA_DIR, 'table3_forensic_validation.csv'))
    render_table(df3, "Table 3: Forensic Case Validation (Subset)", "table3_forensic.png", top_n=10)
    
    # 4. Pathway Enrichment
    df4 = pd.read_csv(os.path.join(DATA_DIR, 'table4_biological_enrichment.csv'))
    render_table(df4, "Table 4: Biological Pathway Enrichment", "table4_enrichment.png")
    
    # 5. Deconvolution Fidelity (Top 10)
    df5 = pd.read_csv(os.path.join(DATA_DIR, 'table5_deconv_fidelity.csv'))
    render_table(df5, "Table 5: Deconvolution Fidelity Check", "table5_deconv.png", top_n=10)
    
    # 6. Train Test Generalization (Summary)
    # Create a summary DataFrame manually since the CSV is raw large data
    df6_raw = pd.read_csv(os.path.join(DATA_DIR, 'table6_train_test_comparison.csv'))
    summary_data = []
    for s in ['Train', 'Test']:
        sub = df6_raw[df6_raw['Set'] == s]
        summary_data.append({
            'Set': s,
            'N_Samples': len(sub),
            'Recon_Loss (Mean)': f"{sub['Recon_Loss'].mean():.4f}",
            'MMAD_Error (Mean)': f"{sub['MMAD_Abs_Error'].mean():.4f}",
            'Consistency_%': f"{sub['Physics_Consistent'].mean()*100:.1f}%"
        })
    df6 = pd.DataFrame(summary_data)
    render_table(df6, "Table 6: Train vs Test Generalization Summary", "table6_generalization.png")
    
    # 7. Detailed Alignment (Top 10)
    df7 = pd.read_csv(os.path.join(DATA_DIR, 'table7_detailed_alignment.csv'))
    # Shorten floating points for display
    # (Actually script generated raw float, checking CSV... generated np.round(..., 2/3). Good.)
    render_table(df7, "Table 7: Detailed Forensic Analysis", "table7_alignment.png", top_n=10)
    
    # 8. Side-by-Side Comparison
    df8 = pd.read_csv(os.path.join(DATA_DIR, 'table8_side_by_side_alignment.csv'))
    render_table(df8, "Table 8: Deconvolution vs Predicted Regional Fraction", "table8_alignment_comparison.png", top_n=20)
    
    print(f"✨ All tables rendered to {TABLES_OUT_DIR}")

if __name__ == "__main__":
    main()
