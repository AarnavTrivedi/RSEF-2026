"""
3D Visualization Widget - Three.js WebGL Backend
--------------------------------------------------
Production-grade 3D visualization using Three.js embedded in QWebEngineView.
Features PBR materials, professional lighting, and bloom effects.
"""

import os
import json
from PyQt6.QtWidgets import QFrame, QVBoxLayout, QLabel
from PyQt6.QtCore import Qt, QUrl, pyqtSlot, QObject
from PyQt6.QtWebEngineWidgets import QWebEngineView
from PyQt6.QtWebEngineCore import QWebEngineSettings
from PyQt6.QtWebChannel import QWebChannel


class PythonBridge(QObject):
    """Bridge object for Python<->JavaScript communication."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
    @pyqtSlot(str)
    def log(self, message):
        """Receive log messages from JavaScript."""
        print(f"[3D JS] {message}")


class AirwayRenderer(QFrame):
    """
    Production-grade 3D airway visualization using Three.js and WebGL.
    
    Features:
    - PBR materials with metalness and roughness
    - Three-point professional lighting
    - Bloom post-processing for highlights
    - Smooth orbital controls with damping
    - Real-time deposition colormap updates
    """
    
    def __init__(self, parent=None):
        super().__init__(parent)
        print("[3D Viz] AirwayRenderer.__init__ (Three.js WebGL backend)")
        
        self._initialized = False
        
        # Layout
        self._layout = QVBoxLayout(self)
        self._layout.setContentsMargins(0, 0, 0, 0)
        
        try:
            # Create WebEngine view
            self._webview = QWebEngineView()
            self._webview.setMinimumSize(400, 300)
            
            # Configure settings
            settings = self._webview.settings()
            settings.setAttribute(QWebEngineSettings.WebAttribute.LocalContentCanAccessRemoteUrls, True)
            settings.setAttribute(QWebEngineSettings.WebAttribute.LocalContentCanAccessFileUrls, True)
            settings.setAttribute(QWebEngineSettings.WebAttribute.WebGLEnabled, True)
            settings.setAttribute(QWebEngineSettings.WebAttribute.Accelerated2dCanvasEnabled, True)
            
            # Setup web channel for Python <-> JS communication
            self._channel = QWebChannel()
            self._bridge = PythonBridge()
            self._channel.registerObject('python', self._bridge)
            self._webview.page().setWebChannel(self._channel)
            
            # Load the HTML file
            html_path = os.path.join(os.path.dirname(__file__), 'viewer3d', 'index.html')
            if os.path.exists(html_path):
                self._webview.load(QUrl.fromLocalFile(html_path))
                print(f"[3D Viz] Loading viewer from: {html_path}")
            else:
                print(f"[3D Viz] ERROR: HTML file not found at {html_path}")
                self._show_error("Viewer HTML file not found")
                return
            
            self._layout.addWidget(self._webview)
            self._initialized = True
            print("[3D Viz] ✓ Three.js WebGL renderer initialized")
            
        except ImportError as e:
            print(f"[3D Viz] ERROR: WebEngine not available: {e}")
            self._show_error(f"WebEngine not installed: {e}")
        except Exception as e:
            print(f"[3D Viz] ERROR: Initialization failed: {e}")
            self._show_error(str(e))
            
    def _show_error(self, message):
        """Display error message in widget."""
        error_label = QLabel(f"⚠️ 3D Visualization Error\n\n{message}\n\nPlease install: pip install PyQt6-WebEngine")
        error_label.setStyleSheet("""
            QLabel {
                color: #ff5858;
                font-size: 14px;
                padding: 40px;
                background-color: #1a1a2e;
                border: 2px dashed #ff5858;
                border-radius: 10px;
            }
        """)
        error_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._layout.addWidget(error_label)
        
    def update_deposition(self, mmad: float, q_flow: float):
        """Update the 3D visualization with new parameters."""
        if not self._initialized:
            return
            
        print(f"[3D Viz] Updating: MMAD={mmad:.1f}, Flow={q_flow:.0f}")
        
        # Call JavaScript function
        js_code = f"if(window.updateDeposition) window.updateDeposition({mmad}, {q_flow});"
        self._webview.page().runJavaScript(js_code)
        
    @property
    def is_ready(self) -> bool:
        """Check if visualization is ready."""
        return self._initialized
