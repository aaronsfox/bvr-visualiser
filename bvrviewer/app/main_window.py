"""Main window: one tab per trial, plus menus."""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtGui import QAction, QKeySequence
from PySide6.QtWidgets import (
    QHBoxLayout, QInputDialog, QMainWindow, QMessageBox, QPushButton, QTabWidget, QWidget,
)

from .. import __version__
from .tab import TrialTab


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("BVR Motion Viewer")
        self.resize(1500, 920)
        self.tabs = QTabWidget()
        self.tabs.setObjectName("TrialTabs")
        self.tabs.setAttribute(Qt.WA_StyledBackground, True)    # lets the stylesheet paint the tab strip dark
        self.tabs.setTabsClosable(True)
        self.tabs.setMovable(True)
        self.tabs.setDocumentMode(True)
        self.tabs.tabCloseRequested.connect(self.close_tab)
        self.tabs.tabBar().tabBarDoubleClicked.connect(self.rename_tab)
        corner = QWidget()
        corner.setObjectName("TabCorner")
        corner.setAttribute(Qt.WA_StyledBackground, True)
        cl = QHBoxLayout(corner)
        cl.setContentsMargins(0, 4, 8, 4)
        plus = QPushButton("+  New tab")
        plus.setObjectName("Small")
        plus.clicked.connect(self.new_tab)
        cl.addWidget(plus)
        self.tabs.setCornerWidget(corner, Qt.TopRightCorner)
        self.setCentralWidget(self.tabs)
        self._menus()
        self.new_tab()

    def _act(self, menu, text, fn, shortcut=None):
        a = QAction(text, self)
        a.triggered.connect(fn)
        if shortcut:
            a.setShortcut(QKeySequence(shortcut))
        menu.addAction(a)
        return a

    def _menus(self) -> None:
        mb = self.menuBar()
        f = mb.addMenu("&File")
        self._act(f, "New tab", self.new_tab, "Ctrl+T")
        self._act(f, "Add model…", lambda: self.current().add_models(), "Ctrl+O")
        self._act(f, "Open saved session…", self.open_session, "Ctrl+Shift+O")
        self._act(f, "Save tab as session…", lambda: self.current().save_session(), "Ctrl+S")
        f.addSeparator()
        self._act(f, "Save screenshot…", lambda: self.current().screenshot(), "Ctrl+P")
        self._act(f, "Export video…", lambda: self.current().export_video(), "Ctrl+E")
        f.addSeparator()
        self._act(f, "Rename tab…", lambda: self.rename_tab(self.tabs.currentIndex()))
        self._act(f, "Close tab", lambda: self.close_tab(self.tabs.currentIndex()), "Ctrl+W")
        self._act(f, "Quit", self.close, "Ctrl+Q")
        v = mb.addMenu("&View")
        for key, text, sc in [("3d", "Free 3D", "1"), ("xy", "XY plane", "2"), ("yz", "YZ plane", "3"),
                              ("xz", "XZ plane", "4")]:
            self._act(v, text, lambda _=False, k=key: self.current().set_view(k), sc)
        self._act(v, "Fit bones in view", lambda: self.current().scene.zoom_to_bones(), "F")
        r = mb.addMenu("&Radiographs")
        self._act(r, "Add X-ray camera…", lambda: self.current().add_camera())
        self._act(r, "Import cameras from .cfg…", lambda: self.current().import_cfg())
        h = mb.addMenu("&Help")
        self._act(h, "About", self.about)

    def current(self) -> TrialTab:
        return self.tabs.currentWidget()

    def new_tab(self, state=None) -> TrialTab:
        tab = TrialTab(state)
        n = self.tabs.count() + 1
        if state is None:
            tab.state.name = f"Untitled {n}" if n > 1 else "Untitled"
        i = self.tabs.addTab(tab, tab.state.name)
        tab.titleChanged.connect(lambda t, w=tab: self.tabs.setTabText(self.tabs.indexOf(w), t))
        self.tabs.setCurrentIndex(i)
        return tab

    def open_session(self) -> None:
        tab = self.current()
        if tab.state.models or tab.state.cameras:
            tab = self.new_tab()
        tab.open_session()

    def rename_tab(self, i: int) -> None:
        if i < 0:
            return
        tab = self.tabs.widget(i)
        text, ok = QInputDialog.getText(self, "Rename tab", "Name:", text=tab.state.name)
        if ok and text.strip():
            tab.state.name = text.strip()
            self.tabs.setTabText(i, tab.state.name)

    def close_tab(self, i: int) -> None:
        if i < 0:
            return
        tab = self.tabs.widget(i)
        if tab.state.models and QMessageBox.question(
                self, "Close tab", f"Close “{tab.state.name}”? Unsaved settings will be lost.") != QMessageBox.Yes:
            return
        tab.close_tab()
        self.tabs.removeTab(i)
        tab.deleteLater()
        if self.tabs.count() == 0:
            self.new_tab()

    def about(self) -> None:
        QMessageBox.about(self, "BVR Motion Viewer",
                          f"<b>BVR Motion Viewer</b> {__version__}<br><br>Visualise bone motion tracked in "
                          "Autoscoper / SlicerAutoscoperM from biplanar videoradiography.<br><br>"
                          "Prototype. Models: STL/OBJ/PLY/VTK surfaces or TIF partial volumes. "
                          "Tracking: Autoscoper .tra (matrix or xyzypr). Calibration: MayaCam 2.0.")

    def closeEvent(self, e):  # noqa: N802
        for i in range(self.tabs.count()):
            self.tabs.widget(i).close_tab()
        super().closeEvent(e)
