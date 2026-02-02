"""
PulmoTrace Analytics Tab
-------------------------
GUI components for advanced analytics features.
"""

from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QPushButton, 
                             QLabel, QComboBox, QListWidget, QGroupBox, 
                             QTableWidget, QTableWidgetItem, QTabWidget,
                             QFileDialog, QMessageBox, QProgressBar, QSplitter,
                             QAbstractItemView)
from PyQt6.QtCore import Qt, pyqtSignal, QThread
from PyQt6.QtGui import QFont
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Any
import os

from models.analytics import ComparativeAnalyzer, SensitivityAnalyzer
from models.export import ReportGenerator, DataExporter


class AnalyticsTab(QWidget):
    """
    Main analytics tab containing comparative analysis and sensitivity analysis.
    """
    
    def __init__(self, engine, parent=None):
        super().__init__(parent)
        self.engine = engine
        self.comparative_analyzer = ComparativeAnalyzer(engine)
        self.sensitivity_analyzer = SensitivityAnalyzer(engine)
        self.report_generator = ReportGenerator(engine)
        
        self.comparison_results = None
        self.sensitivity_results = None
        
        self.init_ui()
        
    def init_ui(self):
        """Initialize the UI components."""
        layout = QVBoxLayout()
        
        # Title
        title = QLabel("Advanced Analytics")
        title_font = QFont()
        title_font.setPointSize(16)
        title_font.setBold(True)
        title.setFont(title_font)
        layout.addWidget(title)
        
        # Tab widget for different analytics modes
        self.analytics_tabs = QTabWidget()
        
        # Comparative Analysis Tab
        self.comparative_widget = ComparativeAnalysisWidget(
            self.engine, self.comparative_analyzer, self.report_generator)
        self.analytics_tabs.addTab(self.comparative_widget, "Comparative Analysis")
        
        # Sensitivity Analysis Tab
        self.sensitivity_widget = SensitivityAnalysisWidget(
            self.engine, self.sensitivity_analyzer)
        self.analytics_tabs.addTab(self.sensitivity_widget, "Sensitivity Analysis")
        
        layout.addWidget(self.analytics_tabs)
        
        self.setLayout(layout)


class ComparativeAnalysisWidget(QWidget):
    """
    Widget for comparative analysis of multiple samples.
    """
    
    def __init__(self, engine, analyzer, report_generator, parent=None):
        super().__init__(parent)
        self.engine = engine
        self.analyzer = analyzer
        self.report_generator = report_generator
        self.comparison_results = None
        
        self.init_ui()
        
    def init_ui(self):
        """Initialize UI components."""
        layout = QVBoxLayout()
        
        # Sample selection panel
        selection_group = QGroupBox("Sample Selection")
        selection_layout = QHBoxLayout()
        
        # Available samples list
        left_panel = QVBoxLayout()
        left_panel.addWidget(QLabel("Available Samples:"))
        self.available_list = QListWidget()
        self.available_list.setSelectionMode(QAbstractItemView.SelectionMode.MultiSelection)
        left_panel.addWidget(self.available_list)
        
        # Load available samples
        self.load_available_samples()
        
        # Buttons
        button_panel = QVBoxLayout()
        button_panel.addStretch()
        
        self.add_button = QPushButton("Add →")
        self.add_button.clicked.connect(self.add_samples)
        button_panel.addWidget(self.add_button)
        
        self.remove_button = QPushButton("← Remove")
        self.remove_button.clicked.connect(self.remove_samples)
        button_panel.addWidget(self.remove_button)
        
        button_panel.addStretch()
        
        # Selected samples list
        right_panel = QVBoxLayout()
        right_panel.addWidget(QLabel("Selected Samples (2-4):"))
        self.selected_list = QListWidget()
        self.selected_list.setSelectionMode(QAbstractItemView.SelectionMode.MultiSelection)
        right_panel.addWidget(self.selected_list)
        
        selection_layout.addLayout(left_panel, 1)
        selection_layout.addLayout(button_panel, 0)
        selection_layout.addLayout(right_panel, 1)
        
        selection_group.setLayout(selection_layout)
        layout.addWidget(selection_group)
        
        # Analysis controls
        controls_layout = QHBoxLayout()
        
        self.compare_button = QPushButton("Run Comparison")
        self.compare_button.clicked.connect(self.run_comparison)
        self.compare_button.setEnabled(False)
        controls_layout.addWidget(self.compare_button)
        
        self.export_button = QPushButton("Export Data")
        self.export_button.clicked.connect(self.export_data)
        self.export_button.setEnabled(False)
        controls_layout.addWidget(self.export_button)
        
        self.report_button = QPushButton("Generate Report")
        self.report_button.clicked.connect(self.generate_report)
        self.report_button.setEnabled(False)
        controls_layout.addWidget(self.report_button)
        
        controls_layout.addStretch()
        layout.addLayout(controls_layout)
        
        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)
        
        # Results tabs
        self.results_tabs = QTabWidget()
        
        # Differential Expression tab
        self.diff_expr_widget = QWidget()
        diff_expr_layout = QVBoxLayout()
        self.diff_expr_table = QTableWidget()
        diff_expr_layout.addWidget(self.diff_expr_table)
        self.diff_expr_widget.setLayout(diff_expr_layout)
        self.results_tabs.addTab(self.diff_expr_widget, "Differential Expression")
        
        # Latent Space tab
        self.latent_widget = QWidget()
        latent_layout = QVBoxLayout()
        self.latent_canvas = FigureCanvas(Figure(figsize=(10, 6)))
        latent_toolbar = NavigationToolbar(self.latent_canvas, self)
        latent_layout.addWidget(latent_toolbar)
        latent_layout.addWidget(self.latent_canvas)
        self.latent_widget.setLayout(latent_layout)
        self.results_tabs.addTab(self.latent_widget, "Latent Space")
        
        # Deposition tab
        self.deposition_widget = QWidget()
        depo_layout = QVBoxLayout()
        self.depo_canvas = FigureCanvas(Figure(figsize=(10, 6)))
        depo_toolbar = NavigationToolbar(self.depo_canvas, self)
        depo_layout.addWidget(depo_toolbar)
        depo_layout.addWidget(self.depo_canvas)
        self.deposition_widget.setLayout(depo_layout)
        self.results_tabs.addTab(self.deposition_widget, "Deposition")
        
        # Cell Composition tab
        self.cell_widget = QWidget()
        cell_layout = QVBoxLayout()
        self.cell_canvas = FigureCanvas(Figure(figsize=(10, 6)))
        cell_toolbar = NavigationToolbar(self.cell_canvas, self)
        cell_layout.addWidget(cell_toolbar)
        cell_layout.addWidget(self.cell_canvas)
        self.cell_widget.setLayout(cell_layout)
        self.results_tabs.addTab(self.cell_widget, "Cell Composition")
        
        layout.addWidget(self.results_tabs, 1)
        
        self.setLayout(layout)
        
        # Connect selection changes
        self.selected_list.itemSelectionChanged.connect(self.update_button_states)
        
    def load_available_samples(self):
        """Load available samples from test data."""
        try:
            test_meta = self.engine.load_test_data()
            sample_ids = test_meta.index.tolist()
            self.available_list.addItems(sample_ids)
        except Exception as e:
            print(f"Error loading samples: {e}")
            
    def add_samples(self):
        """Add selected samples to comparison list."""
        selected_items = self.available_list.selectedItems()
        
        for item in selected_items:
            # Check if already in selected list
            if self.selected_list.findItems(item.text(), Qt.MatchFlag.MatchExactly):
                continue
                
            # Check limit
            if self.selected_list.count() >= 4:
                QMessageBox.warning(self, "Limit Reached", 
                                  "Maximum 4 samples can be compared at once.")
                break
                
            self.selected_list.addItem(item.text())
            
        self.update_button_states()
        
    def remove_samples(self):
        """Remove selected samples from comparison list."""
        selected_items = self.selected_list.selectedItems()
        for item in selected_items:
            self.selected_list.takeItem(self.selected_list.row(item))
            
        self.update_button_states()
        
    def update_button_states(self):
        """Update button enabled states based on selection."""
        n_selected = self.selected_list.count()
        self.compare_button.setEnabled(n_selected >= 2 and n_selected <= 4)
        
    def run_comparison(self):
        """Run comparative analysis."""
        # Get selected sample IDs
        sample_ids = [self.selected_list.item(i).text() 
                     for i in range(self.selected_list.count())]
        
        # Show progress
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 0)  # Indeterminate
        self.compare_button.setEnabled(False)
        
        try:
            # Run analysis
            self.comparison_results = self.analyzer.compare_samples(sample_ids)
            
            # Update displays
            self.update_diff_expr_table()
            self.update_latent_plot()
            self.update_deposition_plot()
            self.update_cell_plot()
            
            # Enable export buttons
            self.export_button.setEnabled(True)
            self.report_button.setEnabled(True)
            
            QMessageBox.information(self, "Success", 
                                  f"Comparison complete for {len(sample_ids)} samples!")
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Comparison failed: {str(e)}")
            
        finally:
            self.progress_bar.setVisible(False)
            self.compare_button.setEnabled(True)
            
    def update_diff_expr_table(self):
        """Update differential expression table."""
        if not self.comparison_results:
            return
            
        diff_expr = self.comparison_results['differential_expression']
        
        # Set up table
        self.diff_expr_table.setRowCount(len(diff_expr))
        self.diff_expr_table.setColumnCount(len(diff_expr.columns))
        self.diff_expr_table.setHorizontalHeaderLabels(diff_expr.columns.tolist())
        
        # Populate table
        for i, row in diff_expr.iterrows():
            for j, col in enumerate(diff_expr.columns):
                value = row[col]
                if isinstance(value, (int, float)):
                    item = QTableWidgetItem(f"{value:.4f}")
                else:
                    item = QTableWidgetItem(str(value))
                self.diff_expr_table.setItem(i, j, item)
                
        self.diff_expr_table.resizeColumnsToContents()
        
    def update_latent_plot(self):
        """Update latent space comparison plot."""
        if not self.comparison_results:
            return
            
        latent_comp = self.comparison_results['latent_comparison']
        sample_ids = self.comparison_results['sample_ids']
        
        fig = self.latent_canvas.figure
        fig.clear()
        
        # Physics latent
        ax1 = fig.add_subplot(2, 1, 1)
        for i, sid in enumerate(sample_ids):
            ax1.plot(range(5), latent_comp['z_phys'][i], marker='o', 
                    label=sid, linewidth=2)
        ax1.set_xlabel('Physics Dimension')
        ax1.set_ylabel('Latent Value')
        ax1.set_title('Physics Latent Space Comparison')
        ax1.legend()
        ax1.grid(alpha=0.3)
        ax1.axhline(0, color='black', linewidth=0.5)
        
        # Biology latent
        ax2 = fig.add_subplot(2, 1, 2)
        for i, sid in enumerate(sample_ids):
            ax2.plot(range(3), latent_comp['z_bio'][i], marker='s', 
                    label=sid, linewidth=2)
        ax2.set_xlabel('Biology Dimension')
        ax2.set_ylabel('Latent Value')
        ax2.set_title('Biology Latent Space Comparison')
        ax2.legend()
        ax2.grid(alpha=0.3)
        ax2.axhline(0, color='black', linewidth=0.5)
        
        fig.tight_layout()
        self.latent_canvas.draw()
        
    def update_deposition_plot(self):
        """Update deposition comparison plot."""
        if not self.comparison_results:
            return
            
        depo_comp = self.comparison_results['deposition_comparison']
        sample_ids = self.comparison_results['sample_ids']
        
        fig = self.depo_canvas.figure
        fig.clear()
        ax = fig.add_subplot(111)
        
        regions = depo_comp['region'].tolist()
        x = np.arange(len(regions))
        width = 0.8 / len(sample_ids)
        
        colors = plt.cm.Set2(range(len(sample_ids)))
        
        for i, sid in enumerate(sample_ids):
            values = depo_comp[f'sample_{sid}'].values
            ax.bar(x + i * width, values, width, label=sid, 
                  color=colors[i], alpha=0.8)
        
        ax.set_xlabel('Region')
        ax.set_ylabel('Deposition (%)')
        ax.set_title('Regional Deposition Comparison')
        ax.set_xticks(x + width * (len(sample_ids) - 1) / 2)
        ax.set_xticklabels(regions, rotation=15, ha='right')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        
        fig.tight_layout()
        self.depo_canvas.draw()
        
    def update_cell_plot(self):
        """Update cell composition comparison plot."""
        if not self.comparison_results:
            return
            
        cell_comp = self.comparison_results['cell_composition']
        sample_ids = self.comparison_results['sample_ids']
        
        fig = self.cell_canvas.figure
        fig.clear()
        ax = fig.add_subplot(111)
        
        cell_types = cell_comp['cell_type'].tolist()
        x = np.arange(len(cell_types))
        width = 0.8 / len(sample_ids)
        
        colors = plt.cm.Set3(range(len(sample_ids)))
        
        for i, sid in enumerate(sample_ids):
            values = cell_comp[f'sample_{sid}'].values
            ax.bar(x + i * width, values, width, label=sid, 
                  color=colors[i], alpha=0.8)
        
        ax.set_xlabel('Cell Type')
        ax.set_ylabel('Proportion')
        ax.set_title('Cell Type Composition Comparison')
        ax.set_xticks(x + width * (len(sample_ids) - 1) / 2)
        ax.set_xticklabels(cell_types, rotation=30, ha='right', fontsize=8)
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        
        fig.tight_layout()
        self.cell_canvas.draw()
        
    def export_data(self):
        """Export comparison data to files."""
        if not self.comparison_results:
            return
            
        # Get output directory
        output_dir = QFileDialog.getExistingDirectory(
            self, "Select Output Directory")
        
        if not output_dir:
            return
            
        try:
            exported_files = DataExporter.export_comparison_data(
                self.comparison_results, output_dir)
            
            file_list = "\n".join([f"- {k}: {os.path.basename(v)}" 
                                  for k, v in exported_files.items()])
            
            QMessageBox.information(self, "Export Complete", 
                                  f"Data exported successfully:\n\n{file_list}")
            
        except Exception as e:
            QMessageBox.critical(self, "Export Error", f"Failed to export data: {str(e)}")
            
    def generate_report(self):
        """Generate PDF report."""
        if not self.comparison_results:
            return
            
        # Get output file
        output_path, _ = QFileDialog.getSaveFileName(
            self, "Save Report", "", "PDF Files (*.pdf)")
        
        if not output_path:
            return
            
        try:
            self.progress_bar.setVisible(True)
            self.progress_bar.setRange(0, 0)
            
            self.report_generator.generate_comparison_report(
                self.comparison_results, output_path)
            
            QMessageBox.information(self, "Report Generated", 
                                  f"Report saved to:\n{output_path}")
            
        except Exception as e:
            QMessageBox.critical(self, "Report Error", 
                               f"Failed to generate report: {str(e)}")
        finally:
            self.progress_bar.setVisible(False)


class SensitivityAnalysisWidget(QWidget):
    """
    Widget for sensitivity analysis.
    """
    
    def __init__(self, engine, analyzer, parent=None):
        super().__init__(parent)
        self.engine = engine
        self.analyzer = analyzer
        self.sensitivity_results = None
        
        self.init_ui()
        
    def init_ui(self):
        """Initialize UI components."""
        layout = QVBoxLayout()
        
        # Controls
        controls_group = QGroupBox("Analysis Settings")
        controls_layout = QHBoxLayout()
        
        controls_layout.addWidget(QLabel("Sample:"))
        self.sample_combo = QComboBox()
        self.load_samples()
        controls_layout.addWidget(self.sample_combo)
        
        controls_layout.addWidget(QLabel("Outcome:"))
        self.outcome_combo = QComboBox()
        self.outcome_combo.addItems(['deposition', 'z_phys', 'z_bio'])
        controls_layout.addWidget(self.outcome_combo)
        
        self.analyze_button = QPushButton("Run Sensitivity Analysis")
        self.analyze_button.clicked.connect(self.run_analysis)
        controls_layout.addWidget(self.analyze_button)
        
        self.export_button = QPushButton("Export Results")
        self.export_button.clicked.connect(self.export_results)
        self.export_button.setEnabled(False)
        controls_layout.addWidget(self.export_button)
        
        controls_layout.addStretch()
        controls_group.setLayout(controls_layout)
        layout.addWidget(controls_group)
        
        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)
        
        # Results
        results_splitter = QSplitter(Qt.Orientation.Vertical)
        
        # Table
        self.results_table = QTableWidget()
        results_splitter.addWidget(self.results_table)
        
        # Plot
        plot_widget = QWidget()
        plot_layout = QVBoxLayout()
        self.results_canvas = FigureCanvas(Figure(figsize=(10, 6)))
        plot_toolbar = NavigationToolbar(self.results_canvas, self)
        plot_layout.addWidget(plot_toolbar)
        plot_layout.addWidget(self.results_canvas)
        plot_widget.setLayout(plot_layout)
        results_splitter.addWidget(plot_widget)
        
        results_splitter.setSizes([300, 400])
        layout.addWidget(results_splitter, 1)
        
        self.setLayout(layout)
        
    def load_samples(self):
        """Load available samples."""
        try:
            test_meta = self.engine.load_test_data()
            sample_ids = test_meta.index.tolist()
            self.sample_combo.addItems(sample_ids)
        except Exception as e:
            print(f"Error loading samples: {e}")
            
    def run_analysis(self):
        """Run sensitivity analysis."""
        sample_id = self.sample_combo.currentText()
        outcome = self.outcome_combo.currentText()
        
        if not sample_id:
            QMessageBox.warning(self, "No Sample", "Please select a sample.")
            return
            
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 0)
        self.analyze_button.setEnabled(False)
        
        try:
            # Run analysis
            self.sensitivity_results = self.analyzer.analyze_gene_sensitivity(
                sample_id, outcome)
            
            # Update displays
            self.update_results_table()
            self.update_results_plot()
            
            self.export_button.setEnabled(True)
            
            QMessageBox.information(self, "Success", 
                                  "Sensitivity analysis complete!")
            
        except Exception as e:
            QMessageBox.critical(self, "Error", 
                               f"Analysis failed: {str(e)}")
        finally:
            self.progress_bar.setVisible(False)
            self.analyze_button.setEnabled(True)
            
    def update_results_table(self):
        """Update results table."""
        if self.sensitivity_results is None:
            return
            
        df = self.sensitivity_results
        
        # Set up table
        self.results_table.setRowCount(len(df))
        self.results_table.setColumnCount(len(df.columns))
        self.results_table.setHorizontalHeaderLabels(df.columns.tolist())
        
        # Populate
        for i, row in df.iterrows():
            for j, col in enumerate(df.columns):
                value = row[col]
                if isinstance(value, (int, float)):
                    item = QTableWidgetItem(f"{value:.6f}")
                else:
                    item = QTableWidgetItem(str(value))
                self.results_table.setItem(i, j, item)
                
        self.results_table.resizeColumnsToContents()
        
    def update_results_plot(self):
        """Update tornado plot."""
        if self.sensitivity_results is None:
            return
            
        df = self.sensitivity_results.head(20)  # Top 20
        
        fig = self.results_canvas.figure
        fig.clear()
        ax = fig.add_subplot(111)
        
        # Tornado plot
        colors = ['red' if x > 0 else 'blue' for x in df['sensitivity']]
        ax.barh(range(len(df)), df['sensitivity'], color=colors, alpha=0.7)
        ax.set_yticks(range(len(df)))
        ax.set_yticklabels(df['gene'], fontsize=9)
        ax.set_xlabel('Sensitivity')
        ax.set_title('Top 20 Genes by Sensitivity (Tornado Plot)')
        ax.axvline(0, color='black', linewidth=0.5)
        ax.grid(axis='x', alpha=0.3)
        
        fig.tight_layout()
        self.results_canvas.draw()
        
    def export_results(self):
        """Export sensitivity results."""
        if self.sensitivity_results is None:
            return
            
        output_path, _ = QFileDialog.getSaveFileName(
            self, "Save Results", "", "CSV Files (*.csv)")
        
        if not output_path:
            return
            
        try:
            self.sensitivity_results.to_csv(output_path, index=False)
            QMessageBox.information(self, "Export Complete", 
                                  f"Results saved to:\n{output_path}")
        except Exception as e:
            QMessageBox.critical(self, "Export Error", 
                               f"Failed to export: {str(e)}")
