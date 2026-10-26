"""Main window: a Start page plus one tab per open tool (Visualiser, and later the analysis tools)."""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtGui import QAction, QColor, QIcon, QKeySequence, QPainter, QPixmap
from PySide6.QtWidgets import (
    QHBoxLayout, QInputDialog, QMainWindow, QMessageBox, QPushButton, QTabBar, QTabWidget, QWidget,
)

from .. import __version__
from .model_builder import ModelBuilderTab
from .start_page import StartPage
from .tab import VisualiserTab
from .tools import tool


def _tab_icon(color: str) -> QIcon:
    pm = QPixmap(12, 12)
    pm.fill(Qt.transparent)
    p = QPainter(pm)
    p.setRenderHint(QPainter.Antialiasing)
    p.setPen(Qt.NoPen)
    p.setBrush(QColor(color))
    p.drawEllipse(2, 2, 8, 8)
    p.end()
    return QIcon(pm)


class MainWindow(QMainWindow):
    # Tool key -> tab widget class. Tools without an entry show as "Coming soon" on the Start page.
    TAB_CLASSES = {"visualiser": VisualiserTab, "model_builder": ModelBuilderTab}

    def __init__(self):
        super().__init__()
        self.setWindowTitle("BVR Motion Viewer")
        self.resize(1500, 920)
        self._counts: dict[str, int] = {}
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
        plus.setToolTip("Go to the Start page to open a new tool tab")
        plus.clicked.connect(self.show_start)
        cl.addWidget(plus)
        self.tabs.setCornerWidget(corner, Qt.TopRightCorner)
        self.setCentralWidget(self.tabs)
        self.start = StartPage()
        self.start.openTool.connect(self.open_tool)
        self.start.openSession.connect(self.open_session)
        self.tabs.addTab(self.start, "Start")
        self.tabs.tabBar().setTabButton(0, QTabBar.RightSide, None)     # the Start page can't be closed
        self.tabs.tabBar().setTabButton(0, QTabBar.LeftSide, None)
        self._menus()

    def _act(self, menu, text, fn, shortcut=None):
        a = QAction(text, self)
        a.triggered.connect(fn)
        if shortcut:
            a.setShortcut(QKeySequence(shortcut))
        menu.addAction(a)
        return a

    def _vis(self, method: str, *args):
        """Menu action that needs a Visualiser tab: run it on the current one, or explain."""
        def run():
            tab = self.current()
            if isinstance(tab, VisualiserTab):
                getattr(tab, method)(*args)
            else:
                QMessageBox.information(self, "Open a Visualiser tab",
                                        "This works in a Visualiser tab. Open one from the Start page, "
                                        "or with File → New Visualiser tab.")
        return run

    def _menus(self) -> None:
        mb = self.menuBar()
        f = mb.addMenu("&File")
        self._act(f, "Start page", self.show_start, "Ctrl+Shift+T")
        self._act(f, "New Visualiser tab", lambda: self.open_tool("visualiser"), "Ctrl+T")
        f.addSeparator()
        self._act(f, "Add model…", self._vis("add_models"), "Ctrl+O")
        self._act(f, "Open saved session…", self.open_session, "Ctrl+Shift+O")
        self._act(f, "Save tab as session…", self._vis("save_session"), "Ctrl+S")
        f.addSeparator()
        self._act(f, "Save screenshot…", self._vis("screenshot"), "Ctrl+P")
        self._act(f, "Export video…", self._vis("export_video"), "Ctrl+E")
        f.addSeparator()
        self._act(f, "Rename tab…", lambda: self.rename_tab(self.tabs.currentIndex()))
        self._act(f, "Close tab", lambda: self.close_tab(self.tabs.currentIndex()), "Ctrl+W")
        self._act(f, "Quit", self.close, "Ctrl+Q")
        v = mb.addMenu("&View")
        for key, text, sc in [("3d", "Free 3D", "1"), ("xy", "XY plane", "2"), ("yz", "YZ plane", "3"),
                              ("xz", "XZ plane", "4")]:
            self._act(v, text, self._vis("set_view", key), sc)
        self._act(v, "Fit bones in view", lambda: self._vis_scene_fit(), "F")
        r = mb.addMenu("&Radiographs")
        self._act(r, "Add X-ray camera…", self._vis("add_camera"))
        self._act(r, "Import cameras from .cfg…", self._vis("import_cfg"))
        h = mb.addMenu("&Help")
        self._act(h, "About", self.about)

    def _vis_scene_fit(self) -> None:
        tab = self.current()
        if isinstance(tab, VisualiserTab):
            tab.scene.zoom_to_bones()

    # ---------------------------------------------------------------- tabs
    def current(self):
        return self.tabs.currentWidget()

    def show_start(self) -> None:
        self.tabs.setCurrentWidget(self.start)

    def open_tool(self, key: str):
        """Open a new tab for a tool from the Start page."""
        t = tool(key)
        cls = self.TAB_CLASSES.get(key)
        if cls is None or not t.available:
            QMessageBox.information(self, t.title, f"{t.title} isn't available yet.")
            return None
        tab = cls()
        self._counts[key] = self._counts.get(key, 0) + 1
        tab.state.name = f"{t.tab_title} {self._counts[key]}"
        i = self.tabs.addTab(tab, _tab_icon(t.color), tab.state.name)
        self.tabs.setTabToolTip(i, f"{t.title} tab · double-click to rename")
        tab.titleChanged.connect(lambda text, w=tab: self.tabs.setTabText(self.tabs.indexOf(w), text))
        self.tabs.setCurrentIndex(i)
        return tab

    def new_visualiser_tab(self) -> VisualiserTab:
        return self.open_tool("visualiser")

    new_tab = new_visualiser_tab     # older name, used by scripts

    def open_session(self) -> None:
        tab = self.current()
        if not isinstance(tab, VisualiserTab) or tab.state.models or tab.state.cameras:
            tab = self.new_visualiser_tab()
        tab.open_session()

    def rename_tab(self, i: int) -> None:
        tab = self.tabs.widget(i)
        if i < 0 or tab is self.start:
            return
        text, ok = QInputDialog.getText(self, "Rename tab", "Name:", text=tab.state.name)
        if ok and text.strip():
            tab.state.name = text.strip()
            self.tabs.setTabText(i, tab.state.name)

    def close_tab(self, i: int) -> None:
        tab = self.tabs.widget(i)
        if i < 0 or tab is self.start:
            return
        if getattr(tab.state, "models", None) and QMessageBox.question(
                self, "Close tab", f"Close “{tab.state.name}”? Unsaved settings will be lost.") != QMessageBox.Yes:
            return
        tab.close_tab()
        self.tabs.removeTab(i)
        tab.deleteLater()
        if self.tabs.count() == 1:
            self.show_start()

    def about(self) -> None:
        QMessageBox.about(self, "BVR Motion Viewer",
                          f"<b>BVR Motion Viewer</b> {__version__}<br><br>Visualise and analyse bone motion "
                          "tracked in Autoscoper / SlicerAutoscoperM from biplanar videoradiography.<br><br>"
                          "Prototype. Tools: Visualiser (available); Reliability Analysis and Validity Analysis "
                          "(coming soon).")

    def closeEvent(self, e):  # noqa: N802
        for i in range(self.tabs.count()):
            self.tabs.widget(i).close_tab()
        super().closeEvent(e)
