
"""
PulmoTrace Presentation Mode
----------------------------
High-fidelity interactive dashboard for demonstrating generative AI capabilities.
Uses Plotly + QWebEngineView for publication-quality rendering.
"""

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, 
    QComboBox, QGroupBox, QSplitter, QProgressBar, QMessageBox,
    QFrame
)
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QFont
from PyQt6.QtWebEngineWidgets import QWebEngineView

import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np

class PresentationTab(QWidget):
    """
    The 'WOW' factor tab.
    Generates synthetic data on the fly and visualizes it interactively.
    """
    
    def __init__(self, engine, parent=None):
        super().__init__(parent)
        self.engine = engine
        self.cohort_data = None
        
        self.init_ui()
        
    def init_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        
        # --- Control Panel (Top) ---
        controls_frame = QFrame()
        controls_frame.setStyleSheet("background-color: #2d2d2d; border-bottom: 1px solid #444;")
        controls_layout = QHBoxLayout(controls_frame)
        
        title_lbl = QLabel("Generative Synthesis:")
        title_lbl.setStyleSheet("color: #fff; font-weight: bold; font-size: 14px;")
        controls_layout.addWidget(title_lbl)
        
        self.scenario_combo = QComboBox()
        self.scenario_combo.addItems(["Mixed Cohort", "Smoker (High Exposure)", "Diesel (Ultrafine)"])
        self.scenario_combo.setStyleSheet("""
            QComboBox { background: #333; color: white; padding: 5px; border: 1px solid #555; }
            QComboBox::drop-down { border: none; }
        """)
        controls_layout.addWidget(self.scenario_combo)
        
        self.gen_btn = QPushButton("✨ Generate Synthetic Cohort")
        self.gen_btn.setStyleSheet("""
            QPushButton { 
                background-color: #4facfe; color: white; font-weight: bold; 
                padding: 6px 15px; border-radius: 4px; border: none;
            }
            QPushButton:hover { background-color: #00f260; }
        """)
        self.gen_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.gen_btn.clicked.connect(self.generate_cohort)
        controls_layout.addWidget(self.gen_btn)
        
        controls_layout.addStretch()
        main_layout.addWidget(controls_frame)
        
        # --- Visualization Area (Splitter) ---
        splitter = QSplitter(Qt.Orientation.Horizontal)
        
        # Left: 3D Latent Space
        self.latent_view = QWebEngineView()
        self.latent_view.setHtml(self._get_placeholder_html("Generative Latent Manifold"))
        splitter.addWidget(self.latent_view)
        
        # Right: Gene Heatmap
        self.heatmap_view = QWebEngineView()
        self.heatmap_view.setHtml(self._get_placeholder_html("Synthetic Expression Profile"))
        splitter.addWidget(self.heatmap_view)
        
        splitter.setSizes([600, 600])
        main_layout.addWidget(splitter)
        
    def _get_placeholder_html(self, title):
        return f"""
        <html>
        <body style='background-color: #1e1e1e; color: #666; display: flex; 
                     justify-content: center; align-items: center; height: 100vh; font-family: sans-serif;'>
            <h2>{title}</h2>
        </body>
        </html>
        """

    def generate_cohort(self):
        scenario_map = {
            "Mixed Cohort": "Mixed",
            "Smoker (High Exposure)": "Smoker",
            "Diesel (Ultrafine)": "Diesel Exhaust"
        }
        
        scenario = scenario_map[self.scenario_combo.currentText()]
        
        try:
            # Generate Data
            self.cohort_data = self.engine.generate_synthetic_cohort(n_samples=50, scenario=scenario)
            
            # Update Visuals
            self.update_charts()
            
        except Exception as e:
            QMessageBox.critical(self, "Generation Error", str(e))

    def update_charts(self):
        if self.cohort_data is None: return
        
        df = self.cohort_data
        
        # 1. 3D Scatter (Latent Space)
        fig_3d = px.scatter_3d(
            df, x='z_mmad', y=df.columns[1], z=df.columns[2], # Rough mapping to latent dims
            color='Scenario',
            hover_name='SampleID',
            title='Generative Latent Manifold (Physics-Informed)',
            template='plotly_dark',
            color_discrete_map={'Smoker-Like': '#f09819', 'Diesel-Like': '#4facfe'}
        )
        fig_3d.update_layout(
            margin=dict(l=0, r=0, b=0, t=40),
            scene=dict(
                xaxis_title='Physics (MMAD)',
                yaxis_title='Bio Dim 1',
                zaxis_title='Bio Dim 2'
            )
        )
        
        # 2. Heatmap
        # Select first 20 genes for clarity
        genes = [c for c in df.columns if c not in ['Scenario', 'z_mmad', 'SampleID']][:25]
        matrix = df[genes].values
        
        fig_hm = go.Figure(data=go.Heatmap(
            z=matrix,
            x=genes,
            y=df['SampleID'],
            colorscale='Viridis',
            colorbar=dict(title='Expression')
        ))
        
        fig_hm.update_layout(
            title='Synthetic Gene Expression Signatures',
            template='plotly_dark',
            margin=dict(l=0, r=0, b=0, t=40),
            xaxis_tickangle=-45
        )
        
        # Render to HTML
        self.latent_view.setHtml(fig_3d.to_html(include_plotlyjs='cdn'))
        self.heatmap_view.setHtml(fig_hm.to_html(include_plotlyjs='cdn'))
