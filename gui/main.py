# SOTA UI Implementation
import sys
import os
import numpy as np
import pandas as pd

# Add the parent directory to sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)
    
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
    QLabel, QPushButton, QComboBox, QTabWidget, QSplitter, 
    QFrame, QTableWidget, QTableWidgetItem, QHeaderView, QFileDialog, QMessageBox,
    QDockWidget, QSizePolicy, QGraphicsDropShadowEffect, QSlider, QGroupBox
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QSize
from PyQt6.QtGui import QIcon, QFont, QColor

# Matplotlib integration
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
import matplotlib.pyplot as plt
import seaborn as sns

from models.engine import PulmoTraceEngine
try:
    from gui.viz_3d import AirwayRenderer
    from gui.analytics_tab import AnalyticsTab
    from gui.results_tab import ResultsTab
except ImportError:
    from viz_3d import AirwayRenderer # Fallback for local execution
    from analytics_tab import AnalyticsTab
    from results_tab import ResultsTab

# --- CONFIGURATION ---
THEME_COLOR = "#4facfe"
SUCCESS_COLOR = "#00f260"
WARNING_COLOR = "#f09819"
DANGER_COLOR = "#ff5858"

# Worker Thread
class EngineLoaderThread(QThread):
    loaded = pyqtSignal(object)
    error = pyqtSignal(str)
    
    def run(self):
        try:
            # Checkpoints in data/checkpoints/
            ckpt_dir = os.path.join(parent_dir, 'data', 'checkpoints')
            engine = PulmoTraceEngine(checkpoints_dir=ckpt_dir)
            self.loaded.emit(engine)
        except Exception as e:
            self.error.emit(str(e))

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        
        self.setWindowTitle("PulmoTrace AI - Inverse Dosimetry Platform")
        self.resize(1600, 1000)
        
        # Load Stylesheet
        self._load_stylesheet()
        
        self.engine = None
        
        # UI Setup
        self._setup_ui()
        
        # Start Loading
        self.status_label.setText("🚀 Initializing Neural Surrogate...")
        self.loader_thread = EngineLoaderThread()
        self.loader_thread.loaded.connect(self._on_engine_loaded)
        self.loader_thread.error.connect(self._on_engine_error)
        self.loader_thread.start()
        
    def _load_stylesheet(self):
        style_path = os.path.join(current_dir, "styles.qss")
        if os.path.exists(style_path):
            with open(style_path, "r") as f:
                self.setStyleSheet(f.read())
        else:
            print("Warning: styles.qss not found!")

    def _setup_ui(self):
        # --- Sidebar ---
        self.sidebar_dock = QDockWidget("Core Controls", self)
        self.sidebar_dock.setAllowedAreas(Qt.DockWidgetArea.LeftDockWidgetArea)
        self.sidebar_dock.setFeatures(QDockWidget.DockWidgetFeature.NoDockWidgetFeatures)
        self.sidebar_dock.setTitleBarWidget(QWidget()) # Hide default title bar for cleaner look
        
        sidebar_widget = QWidget()
        sidebar_widget.setObjectName("sidebar")
        sidebar_layout = QVBoxLayout(sidebar_widget)
        sidebar_layout.setContentsMargins(20, 30, 20, 30)
        sidebar_layout.setSpacing(20)
        
        # Brand
        title = QLabel("PulmoTrace")
        title.setObjectName("title_label")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        sidebar_layout.addWidget(title)
        
        sidebar_layout.addSpacing(20)
        
        # Controls Group
        self._add_sidebar_label(sidebar_layout, "DATA SOURCE")
        self.source_combo = QComboBox()
        self.source_combo.addItems(["Demo Sample", "Official Test Set (GEO)", "Upload CSV"])
        self.source_combo.currentIndexChanged.connect(self._toggle_inputs)
        sidebar_layout.addWidget(self.source_combo)
        
        self._add_sidebar_label(sidebar_layout, "CONDITION PRESET")
        self.demo_combo = QComboBox()
        self.demo_combo.addItems(["Smoker (High Exposure)", "Non-Smoker", "Occupational (Diesel)"])
        sidebar_layout.addWidget(self.demo_combo)
        
        self.load_btn = QPushButton("📂 Load CSV File")
        self.load_btn.setVisible(False)
        self.load_btn.clicked.connect(self._load_csv_file)
        sidebar_layout.addWidget(self.load_btn)
        
        sidebar_layout.addSpacing(20)
        
        # --- Visualization Controls ---
        self._add_sidebar_label(sidebar_layout, "REAL-TIME VIZ")
        
        # MMAD
        self.mmad_lbl = QLabel("MMAD: 2.5 µm")
        self.mmad_lbl.setStyleSheet("color: #ccc; font-size: 11px;")
        sidebar_layout.addWidget(self.mmad_lbl)
        
        self.mmad_slider = QSlider(Qt.Orientation.Horizontal)
        self.mmad_slider.setRange(1, 100) # 0.1 to 10.0
        self.mmad_slider.setValue(25)
        self.mmad_slider.valueChanged.connect(self._update_viz)
        sidebar_layout.addWidget(self.mmad_slider)
        
        # Flow
        self.flow_lbl = QLabel("Flow: 30 L/min")
        self.flow_lbl.setStyleSheet("color: #ccc; font-size: 11px;")
        sidebar_layout.addWidget(self.flow_lbl)
        
        self.flow_slider = QSlider(Qt.Orientation.Horizontal)
        self.flow_slider.setRange(15, 60)
        self.flow_slider.setValue(30)
        self.flow_slider.valueChanged.connect(self._update_viz)
        sidebar_layout.addWidget(self.flow_slider)
        
        sidebar_layout.addSpacing(20)
        
        self.run_btn = QPushButton("▶ RUN ANALYSIS")
        self.run_btn.setObjectName("action_btn")
        self.run_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.run_btn.clicked.connect(self._run_analysis)
        self.run_btn.setEnabled(False)
        sidebar_layout.addWidget(self.run_btn)
        
        sidebar_layout.addStretch()
        
        # Status Footer
        self.status_label = QLabel("Status: Waiting")
        self.status_label.setWordWrap(True)
        self.status_label.setStyleSheet("color: #666; font-size: 11px; border-top: 1px solid #333; padding-top: 10px;")
        sidebar_layout.addWidget(self.status_label)
        
        self.sidebar_dock.setWidget(sidebar_widget)
        self.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, self.sidebar_dock)
        
        # --- Central Area ---
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        
        main_layout = QVBoxLayout(self.central_widget)
        main_layout.setContentsMargins(20, 20, 20, 20)
        main_layout.setSpacing(20)
        
        # 1. Metrics Grid (HUD)
        metrics_container = QFrame()
        metrics_container.setObjectName("metrics_container")
        metrics_layout = QHBoxLayout(metrics_container)
        metrics_layout.setContentsMargins(0,0,0,0)
        metrics_layout.setSpacing(15)
        
        self.card_mmad = self._create_metric_card("Inferred MMAD", "--", "Unknown Mode", "#4facfe")
        self.card_source = self._create_metric_card("Likely Source", "--", "Awaiting Data", "#00f260")
        self.card_tissue = self._create_metric_card("Dominant Tissue", "--", "Compartment", "#f09819")
        self.card_consistency = self._create_metric_card("Phys-Bio Consistency", "--", "Validation", "#ff5858")
        
        metrics_layout.addWidget(self.card_mmad)
        metrics_layout.addWidget(self.card_source)
        metrics_layout.addWidget(self.card_tissue)
        metrics_layout.addWidget(self.card_consistency)
        
        main_layout.addWidget(metrics_container)
        
        # 2. Visualization Area
        self.viz_tabs = QTabWidget()
        main_layout.addWidget(self.viz_tabs)
        
        # Tab 1: Dashboard
        self.dashboard_tab = QWidget()
        self._setup_dashboard_tab()
        self.viz_tabs.addTab(self.dashboard_tab, "Interactive Dashboard")
        
        # Tab 2: Deep Dive
        self.details_tab = QWidget()
        self._setup_details_tab()
        self.viz_tabs.addTab(self.details_tab, "Molecular Deep Dive")

        # Tab 3: 3D Airway Map (CFD Viz)
        self.viz_container = QWidget()
        self.viz_layout = QVBoxLayout(self.viz_container)
        self.viz_layout.setContentsMargins(0,0,0,0)
        
        self.viz_3d = AirwayRenderer()
        self.viz_layout.addWidget(self.viz_3d)
        
        self.viz_tabs.addTab(self.viz_container, "3D Airway Map")
        
        # Tab 4: Advanced Analytics
        self.analytics_tab_widget = None  # Will be initialized when engine loads
        self.analytics_placeholder = QWidget()
        analytics_placeholder_layout = QVBoxLayout(self.analytics_placeholder)
        analytics_placeholder_layout.addWidget(QLabel("Loading analytics module..."))
        self.viz_tabs.addTab(self.analytics_placeholder, "Advanced Analytics")
        
        self.viz_tabs.addTab(self.analytics_placeholder, "Advanced Analytics")
        
        # Tab 5: Validation Results (New)
        self.results_placeholder = QWidget()
        res_layout = QVBoxLayout(self.results_placeholder)
        res_layout.addWidget(QLabel("Loading validation module..."))
        self.viz_tabs.addTab(self.results_placeholder, "✅ Validation Results")
        
        # Connect tab change to trigger 3D init when tab is selected
        self.viz_tabs.currentChanged.connect(self._on_tab_changed)

    def _add_sidebar_label(self, layout, text):
        lbl = QLabel(text)
        lbl.setStyleSheet("color: #666; font-weight: bold; font-size: 10px; letter-spacing: 1px;")
        layout.addWidget(lbl)

    def _create_metric_card(self, title, value, subtitle, accent_color):
        frame = QFrame()
        frame.setObjectName("card")
        frame.setFrameShape(QFrame.Shape.StyledPanel)
        
        # Add a subtle shadow
        shadow = QGraphicsDropShadowEffect()
        shadow.setBlurRadius(15)
        shadow.setColor(QColor(0, 0, 0, 100))
        shadow.setOffset(0, 4)
        frame.setGraphicsEffect(shadow)
        
        layout = QVBoxLayout(frame)
        layout.setContentsMargins(20, 20, 20, 20)
        
        t = QLabel(title)
        t.setObjectName("metric_title")
        
        v = QLabel(value)
        v.setObjectName("metric_value")
        v.setStyleSheet(f"color: {accent_color};") # Dynamic accent
        
        s = QLabel(subtitle)
        s.setObjectName("metric_subtitle")
        
        layout.addWidget(t)
        layout.addWidget(v)
        layout.addWidget(s)
        layout.addStretch()
        
        return frame

    def _setup_dashboard_tab(self):
        layout = QHBoxLayout(self.dashboard_tab)
        layout.setContentsMargins(10, 20, 10, 10)
        layout.setSpacing(20)
        
        # We need a dark theme for matplotlib
        plt.style.use('dark_background')
        
        # Plot 1: Deposition Fate
        self.fig1, self.ax1 = plt.subplots(figsize=(6, 5))
        self.fig1.patch.set_facecolor('#1e1e1e')
        self.ax1.set_facecolor('#1e1e1e')
        self.canvas1 = FigureCanvas(self.fig1)
        self._style_canvas(self.canvas1)
        layout.addWidget(self._wrap_plot("Predicted Deposition Fate", self.canvas1))
        
        # Plot 2: Tissue Composition
        self.fig2, self.ax2 = plt.subplots(figsize=(6, 5))
        self.fig2.patch.set_facecolor('#1e1e1e')
        self.ax2.set_facecolor('#1e1e1e')
        self.canvas2 = FigureCanvas(self.fig2)
        self._style_canvas(self.canvas2)
        layout.addWidget(self._wrap_plot("Tissue Composition (Deconvolution)", self.canvas2))
        
    def _style_canvas(self, canvas):
        canvas.setStyleSheet("background-color: #1e1e1e; border-radius: 8px;")
        
    def _wrap_plot(self, title, canvas):
        frame = QFrame()
        frame.setObjectName("card")
        l = QVBoxLayout(frame)
        l.setContentsMargins(15, 15, 15, 15)
        
        lbl = QLabel(title)
        lbl.setStyleSheet("font-size: 14px; font-weight: bold; color: #fff; margin-bottom: 10px;")
        l.addWidget(lbl)
        l.addWidget(canvas)
        
        # Custom Toolbar logic could go here to hide it or style it
        toolbar = NavigationToolbar(canvas, frame)
        toolbar.setStyleSheet("background-color: #2d2d2d; border: none; color: white;")
        l.addWidget(toolbar)
        
        return frame

    def _setup_details_tab(self):
        layout = QHBoxLayout(self.details_tab)
        
        self.consistency_table = self._style_table(QTableWidget())
        self.consistency_table.setColumnCount(3)
        self.consistency_table.setHorizontalHeaderLabels(["Compartment", "Bio Signal", "Phys Dose"])
        self.consistency_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        layout.addWidget(self.consistency_table)
        
        self.gene_table = self._style_table(QTableWidget())
        self.gene_table.setColumnCount(2)
        self.gene_table.setHorizontalHeaderLabels(["Gene", "Expression (Log-Norm)"])
        self.gene_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        layout.addWidget(self.gene_table)
        
    def _style_table(self, table):
        table.verticalHeader().setVisible(False)
        table.setAlternatingRowColors(True)
        table.setShowGrid(False)
        return table

    def _toggle_inputs(self):
        source = self.source_combo.currentText()
        
        self.load_btn.setVisible(source == "Upload CSV")
        self.demo_combo.setVisible(source != "Upload CSV")
        
        # Populate Demo Combo based on Source
        self.demo_combo.blockSignals(True)
        self.demo_combo.clear()
        
        if source == "Demo Sample":
            self.demo_combo.addItems(["Smoker (High Exposure)", "Non-Smoker", "Occupational (Diesel)"])
            
        elif source == "Official Test Set (GEO)":
            if self.engine:
                self.status_label.setText("Status: Loading Test Set...")
                QApplication.processEvents()
                try:
                    meta = self.engine.load_test_data()
                    # Add items in format "GSM12345: Group"
                    if not meta.empty:
                        # meta likely has 'Group' column or similar
                        items = []
                        for idx, row in meta.iterrows():
                            lbl = f"{idx} ({row.get('Group', 'Unknown')})"
                            items.append(lbl)
                        self.demo_combo.addItems(items)
                        self.status_label.setText(f"Status: Loaded {len(items)} samples.")
                    else:
                        self.demo_combo.addItem("No samples found")
                except Exception as e:
                    self.status_label.setText("Status: Error Loading Data")
                    QMessageBox.warning(self, "Load Error", str(e))
            else:
                 self.demo_combo.addItem("Engine not ready...")
        
        self.demo_combo.blockSignals(False)

    def _on_engine_loaded(self, engine):
        self.engine = engine
        self.status_label.setText("Status: Engine Ready - Waiting for Input")
        self.run_btn.setEnabled(True)
        self.run_btn.setText("▶ RUN ANALYSIS")
        
        # Initialize Analytics Tab
        try:
            self.analytics_tab_widget = AnalyticsTab(engine)
            # Replace placeholder with actual analytics tab
            analytics_tab_index = self.viz_tabs.indexOf(self.analytics_placeholder)
            if analytics_tab_index >= 0:
                self.viz_tabs.removeTab(analytics_tab_index)
                self.viz_tabs.insertTab(analytics_tab_index, self.analytics_tab_widget, "Advanced Analytics")
            print("[Analytics] Advanced Analytics tab initialized successfully")
        except Exception as e:
            print(f"Warning: Could not initialize analytics tab: {e}")

        except Exception as e:
            print(f"Warning: Could not initialize analytics tab: {e}")

        # Initialize Results Tab
        try:
            self.results_tab = ResultsTab(engine)
            res_index = self.viz_tabs.indexOf(self.results_placeholder)
            if res_index >= 0:
                self.viz_tabs.removeTab(res_index)
                self.viz_tabs.insertTab(res_index, self.results_tab, "✅ Validation Results")
            print("[Validation] Results tab initialized successfully")
        except Exception as e:
            print(f"Warning: Could not initialize results tab: {e}")

    def _on_engine_error(self, err_msg):
        self.status_label.setText(f"Status: Error loading engine!")
        QMessageBox.critical(self, "Engine Error", f"Failed to initialize AI Engine:\n{err_msg}")

    def _load_csv_file(self):
        # Implementation for real file loading
        file_path, _ = QFileDialog.getOpenFileName(self, "Open CSV", "", "CSV Files (*.csv);;Text Files (*.txt)")
        if file_path:
            self.status_label.setText(f"Selected: {os.path.basename(file_path)}")
            # TODO: Store the path for _run_analysis to use

    def _update_card(self, card, value, subtitle=None):
        card.findChild(QLabel, "metric_value").setText(value)
        if subtitle:
            card.findChild(QLabel, "metric_subtitle").setText(subtitle)

    def _run_analysis(self):
        if not self.engine: return
        
        self.status_label.setText("Status: Running Physics-Informed Inference...")
        self.run_btn.setEnabled(False)
        self.run_btn.setText("Computing...")
        QApplication.processEvents()
        
        # --- Data Generation Logic ---
        sample_name = "Unknown"
        sample_data = None
        source = self.source_combo.currentText()
        
        if source == "Official Test Set (GEO)":
            # Extract ID from "GSM12345 (Group)"
            combo_text = self.demo_combo.currentText()
            sample_id = combo_text.split(' ')[0]
            
            sample_data = self.engine.get_test_sample(sample_id)
            sample_name = sample_id
            
            if sample_data is None:
                QMessageBox.warning(self, "Error", "Could not retrieve data for selected sample.")
                self.run_btn.setEnabled(True)
                return

        elif source == "Demo Sample":
            condition = self.demo_combo.currentText()
            if "Smoker" in condition:
                np.random.seed(42)
                vals = np.random.normal(0, 1, 45)
                for i, g in enumerate(self.engine.gene_names):
                    if g in ['CYP1A1', 'AHR', 'MMP12']: vals[i] += 2.0
                sample_data = vals
                sample_name = "Demo_Smoker_High"
            elif "Diesel" in condition:
                np.random.seed(123)
                vals = np.random.normal(0, 1, 45)
                for i, g in enumerate(self.engine.gene_names):
                    if g in ['IL6', 'TNF', 'HMOX1']: vals[i] += 1.5
                sample_data = vals
                sample_name = "Demo_Diesel_Occupational"
            else:
                np.random.seed(0)
                vals = np.random.normal(-0.5, 0.5, 45) 
                sample_data = vals
                sample_name = "Demo_Healthy_Control"
                
        try:
            result = self.engine.process_sample(sample_data, sample_name=sample_name)
            self._update_display(result)
            self.status_label.setText(f"Status: Analysis Complete ({sample_name})")
        except Exception as e:
            QMessageBox.critical(self, "Analysis Failed", str(e))
            self.status_label.setText("Status: Failure")
        finally:
            self.run_btn.setEnabled(True)
            self.run_btn.setText("▶ RUN ANALYSIS")

    def _update_display(self, result):
        phys = result['physics']
        bio = result['biology']
        align = result['alignment']
        
        # 1. Update Metrics
        self._update_card(self.card_mmad, f"{phys['mmad']:.2f} µm", phys['interpretation']['size_class'])
        self._update_card(self.card_source, phys['interpretation']['source'])
        self._update_card(self.card_tissue, bio['dominant_tissue'])
        
        consist_text = "Consistent" if align['consistent'] else "Divergent"
        self._update_card(self.card_consistency, consist_text, f"Score: {align['overall_score']:.2f}")
        
        # 2. Update Plots
        # Pie Chart - Improved Aesthetics
        self.ax1.clear()
        dep = phys['deposition']
        labels = list(dep.keys())
        sizes = list(dep.values())
        
        # Modern Neon Palette
        colors = ['#4facfe', '#bd34fe', '#00f260', '#f09819'] 
        wedges, texts, autotexts = self.ax1.pie(sizes, labels=labels, autopct='%1.1f%%', 
                                              startangle=90, colors=colors, 
                                              textprops=dict(color="white"))
        
        # Donut Chart Style
        self.ax1.add_artist(plt.Circle((0,0),0.70,fc='#1e1e1e'))
        self.ax1.axis('equal')
        self.canvas1.draw()
        
        # Bar Chart - Tissue
        self.ax2.clear()
        comp = bio.get('proportions', bio.get('composition', {}))
        types = list(comp.keys())
        vals = list(comp.values())
        
        sns.barplot(x=vals, y=types, ax=self.ax2, palette="viridis", orient='h')
        self.ax2.set_xlabel("Proportion")
        self.ax2.grid(True, color='#333', linestyle='--')
        self.canvas2.draw()
        
        # 3. Populate Molecular Deep Dive Tables
        # Consistency Table
        self.consistency_table.setRowCount(2)
        self.consistency_table.setItem(0, 0, QTableWidgetItem("Airways (Bronchial)"))
        self.consistency_table.setItem(0, 1, QTableWidgetItem(f"{align['bio_scores']['Airway']:.1%}"))
        self.consistency_table.setItem(0, 2, QTableWidgetItem(f"{align['phys_scores']['Airway']:.1%}"))
        
        self.consistency_table.setItem(1, 0, QTableWidgetItem("Deep Lung (Alveolar)"))
        self.consistency_table.setItem(1, 1, QTableWidgetItem(f"{align['bio_scores']['Alveolar']:.1%}"))
        self.consistency_table.setItem(1, 2, QTableWidgetItem(f"{align['phys_scores']['Alveolar']:.1%}"))
        
        # Gene Expression Table
        gene_data = result.get('gene_expression', {})
        if gene_data:
            self.gene_table.setRowCount(len(gene_data))
            for i, (gene, expr) in enumerate(sorted(gene_data.items(), key=lambda x: -x[1])[:20]):
                self.gene_table.setItem(i, 0, QTableWidgetItem(gene))
                self.gene_table.setItem(i, 1, QTableWidgetItem(f"{expr:.3f}"))
        else:
            # Fallback: show top genes from input
            sample_vals = result.get('input_expression', [])
            gene_names = self.engine.gene_names if self.engine else []
            if len(sample_vals) > 0 and len(gene_names) > 0:
                pairs = list(zip(gene_names, sample_vals))
                pairs.sort(key=lambda x: -abs(x[1]))
                self.gene_table.setRowCount(min(20, len(pairs)))
                for i, (gene, val) in enumerate(pairs[:20]):
                    self.gene_table.setItem(i, 0, QTableWidgetItem(gene))
                    self.gene_table.setItem(i, 1, QTableWidgetItem(f"{val:.3f}"))

    def _update_viz(self):
        """Update 3D Visualization based on sliders."""
        mmad = self.mmad_slider.value() / 10.0
        flow = self.flow_slider.value()
        
        # Update Labels
        self.mmad_lbl.setText(f"MMAD: {mmad:.1f} µm")
        self.flow_lbl.setText(f"Flow: {flow} L/min")
        
        # Update 3D Viz
        if hasattr(self, 'viz_3d'):
            self.viz_3d.update_deposition(mmad, flow)
            
            # Switch to 3D tab automatically on interaction? No, let user switch.
            # But if they drag sliders, they likely want to see it.
            # self.viz_tabs.setCurrentIndex(2) 
            
    def _on_tab_changed(self, index):
        """Handle tab change."""
        if index == 2:  # 3D Airway Map tab
            print("[MainWindow] 3D tab selected")
            # Matplotlib renderer initializes immediately, no deferred init needed

if __name__ == "__main__":
    app = QApplication(sys.argv)
    
    # Set Fusion Theme for consistent dark mode base
    app.setStyle("Fusion")
    
    window = MainWindow()
    window.show()
    
    sys.exit(app.exec())

