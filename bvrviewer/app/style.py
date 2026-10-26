"""Dark theme matching the design mock-up."""

ACCENT = "#4F9CF9"


def apply_theme(app) -> None:
    """Dark theme everywhere.

    The Fusion style plus a dark palette stops Windows' native style from painting light
    backgrounds behind widgets the stylesheet doesn't cover (such as the strip behind the tabs).
    """
    from PySide6.QtGui import QColor, QPalette

    app.setStyle("Fusion")
    pal = QPalette()
    for role, colour in [
        (QPalette.Window, "#121419"), (QPalette.WindowText, "#E6E8EC"), (QPalette.Base, "#121419"),
        (QPalette.AlternateBase, "#1C1F26"), (QPalette.Text, "#E6E8EC"), (QPalette.Button, "#232731"),
        (QPalette.ButtonText, "#E6E8EC"), (QPalette.Highlight, ACCENT), (QPalette.HighlightedText, "#0B1220"),
        (QPalette.ToolTipBase, "#1C1F26"), (QPalette.ToolTipText, "#E6E8EC"), (QPalette.PlaceholderText, "#7A8291"),
        (QPalette.Link, "#7DB5FB"), (QPalette.BrightText, "#FFFFFF"), (QPalette.Mid, "#343946"),
        (QPalette.Dark, "#0E1014"), (QPalette.Light, "#343946"), (QPalette.Shadow, "#07080B"),
    ]:
        pal.setColor(role, QColor(colour))
    for role in (QPalette.WindowText, QPalette.Text, QPalette.ButtonText):
        pal.setColor(QPalette.Disabled, role, QColor("#5A6170"))
    app.setPalette(pal)
    app.setStyleSheet(QSS + _indicator_qss())


_ICONS = {
    "check": '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 16 16"><path d="M3.5 8.5l3 3 6-7" '
             'fill="none" stroke="#0B1220" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"/></svg>',
    "radio": f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 16 16"><circle cx="8" cy="8" r="4" '
             f'fill="{ACCENT}"/></svg>',
    "close": '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 16 16"><path d="M4.5 4.5l7 7M11.5 4.5l-7 7" '
             'stroke="#A3AAB8" stroke-width="1.6" stroke-linecap="round"/></svg>',
    "close-hover": '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 16 16"><path d="M4.5 4.5l7 7M11.5 4.5l-7 7" '
                   'stroke="#F0A3A3" stroke-width="1.8" stroke-linecap="round"/></svg>',
    "up": '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 10 6"><path d="M1 5l4-4 4 4" fill="none" '
          'stroke="#C3C8D2" stroke-width="1.4" stroke-linecap="round" stroke-linejoin="round"/></svg>',
    "down": '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 10 6"><path d="M1 1l4 4 4-4" fill="none" '
            'stroke="#C3C8D2" stroke-width="1.4" stroke-linecap="round" stroke-linejoin="round"/></svg>',
}


def _icon_dir():
    """Write the small theme icons to a cache folder once (stylesheets need real file paths)."""
    import tempfile
    from pathlib import Path

    d = Path(tempfile.gettempdir()) / "bvrviewer-icons-v1"
    d.mkdir(exist_ok=True)
    for name, svg in _ICONS.items():
        p = d / f"{name}.svg"
        if not p.exists() or p.read_text() != svg:
            p.write_text(svg)
    return d.as_posix()


def _indicator_qss() -> str:
    """Checkbox, radio, spin-box and tab-close styling that stays visible on the dark background."""
    d = _icon_dir()
    return f"""
QCheckBox::indicator {{ width: 15px; height: 15px; border: 1px solid #6B7280; border-radius: 4px; background: #121419; }}
QCheckBox::indicator:hover {{ border-color: {ACCENT}; }}
QCheckBox::indicator:checked {{ background: {ACCENT}; border-color: {ACCENT}; image: url({d}/check.svg); }}
QCheckBox::indicator:disabled {{ border-color: #3A3F4B; background: #1C1F26; }}
QRadioButton::indicator {{ width: 15px; height: 15px; border: 1px solid #6B7280; border-radius: 8px; background: #121419; }}
QRadioButton::indicator:hover {{ border-color: {ACCENT}; }}
QRadioButton::indicator:checked {{ border-color: {ACCENT}; image: url({d}/radio.svg); }}
QSpinBox, QDoubleSpinBox {{ padding-right: 18px; }}
QSpinBox::up-button, QDoubleSpinBox::up-button {{ subcontrol-origin: border; subcontrol-position: top right; width: 16px;
    border-left: 1px solid #343946; border-top-right-radius: 6px; background: #1C1F26; }}
QSpinBox::down-button, QDoubleSpinBox::down-button {{ subcontrol-origin: border; subcontrol-position: bottom right;
    width: 16px; border-left: 1px solid #343946; border-bottom-right-radius: 6px; background: #1C1F26; }}
QSpinBox::up-button:hover, QSpinBox::down-button:hover, QDoubleSpinBox::up-button:hover,
QDoubleSpinBox::down-button:hover {{ background: #2A2F3B; }}
QSpinBox::up-arrow, QDoubleSpinBox::up-arrow {{ image: url({d}/up.svg); width: 8px; height: 5px; }}
QSpinBox::down-arrow, QDoubleSpinBox::down-arrow {{ image: url({d}/down.svg); width: 8px; height: 5px; }}
QComboBox::drop-down {{ border: 0; width: 18px; }}
QComboBox::down-arrow {{ image: url({d}/down.svg); width: 9px; height: 6px; }}
QTabBar::close-button {{ image: url({d}/close.svg); subcontrol-position: right; margin-left: 4px; }}
QTabBar::close-button:hover {{ image: url({d}/close-hover.svg); background: #2A2F3B; border-radius: 4px; }}
"""

QSS = f"""
* {{ font-family: "IBM Plex Sans", "Segoe UI", "Helvetica Neue", Arial, sans-serif; font-size: 13px; color: #E6E8EC; }}
QMainWindow, QDialog, QWidget#Page {{ background: #121419; }}
QWidget#Panel {{ background: #171A20; }}
QFrame#Card {{ background: #1C1F26; border: 1px solid #343946; border-radius: 10px; }}
QFrame#Row {{ border: 1px solid transparent; border-radius: 8px; }}
QFrame#Row[selected="true"] {{ background: #232A36; border: 1px solid #3D5A80; }}
QFrame#Empty {{ background: #0B0D11; }}
QLabel#H2 {{ font-size: 11px; font-weight: 600; letter-spacing: 1px; color: #A3AAB8; }}
QLabel#Muted {{ color: #A3AAB8; font-size: 12px; }}
QLabel#Mono {{ font-family: "IBM Plex Mono", Menlo, Consolas, monospace; font-size: 11px; color: #A3AAB8; }}
QLabel#Ok {{ color: #8FD3A9; font-size: 12px; }}
QLabel#Warn {{ color: #F2C48D; font-size: 12px; }}
QLabel#Error {{ color: #F0A3A3; font-size: 12px; }}
QLabel#Big {{ font-size: 24px; font-weight: 600; }}
QLabel#Step {{ background: {ACCENT}; color: #0B1220; border-radius: 14px; font-weight: 600; }}
QLabel#StepOff {{ border: 1px solid #4A5160; color: #C3C8D2; border-radius: 14px; font-weight: 600; }}
QPushButton {{ background: #232731; border: 1px solid #343946; border-radius: 7px; padding: 6px 12px; color: #E6E8EC; }}
QPushButton:hover {{ background: #2A2F3B; }}
QPushButton:checked {{ background: #2A3A52; border-color: #3D5A80; }}
QPushButton:disabled {{ color: #5A6170; }}
QPushButton#Primary {{ background: {ACCENT}; color: #0B1220; border: 0; font-weight: 600; }}
QPushButton#Primary:hover {{ background: #6AAEFB; }}
QPushButton#Small {{ padding: 3px 8px; font-size: 12px; }}
QPushButton#Attach {{ background: #2A2116; border: 1px solid #6B4A1F; color: #F2C48D; padding: 3px 8px; font-size: 12px; }}
QPushButton#Danger {{ background: transparent; border: 1px solid #5A3434; color: #F0A3A3; padding: 4px 10px; font-size: 12px; }}
QPushButton#Dashed {{ background: transparent; border: 1px dashed #4A5160; color: #C3C8D2; padding: 8px; }}
QPushButton#Swatch {{ border: 1px solid #3A3F4B; border-radius: 6px; padding: 0; }}
QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox {{ background: #121419; border: 1px solid #343946; border-radius: 6px; padding: 4px 6px; }}
QLineEdit#NameEdit {{ background: transparent; border: 1px solid transparent; font-size: 14px; font-weight: 500; }}
QLineEdit#NameEdit:focus {{ background: #121419; border: 1px solid #3D5A80; }}
QComboBox QAbstractItemView {{ background: #1C1F26; selection-background-color: #2A3A52; }}
QSlider::groove:horizontal {{ height: 4px; background: #343946; border-radius: 2px; }}
QSlider::sub-page:horizontal {{ background: {ACCENT}; border-radius: 2px; }}
QSlider::handle:horizontal {{ background: #E6E8EC; width: 14px; margin: -6px 0; border-radius: 7px; }}
QCheckBox, QRadioButton {{ spacing: 8px; }}
QTabWidget#TrialTabs {{ background: #0E1014; }}
QTabWidget#TrialTabs::pane {{ border: 0; border-top: 1px solid #262A33; background: #121419; }}
QTabWidget#TrialTabs::tab-bar {{ left: 8px; }}
QTabWidget#TrialTabs > QTabBar {{ background: #0E1014; }}
QTabWidget#TrialTabs QWidget#TabCorner {{ background: #0E1014; }}
QTabBar {{ background: #0E1014; qproperty-drawBase: 0; }}
QTabBar::tab {{ background: transparent; color: #A3AAB8; padding: 9px 14px; border: 1px solid transparent; border-top-left-radius: 8px; border-top-right-radius: 8px; }}
QTabBar::tab:selected {{ background: #171A20; color: #E6E8EC; border-color: #262A33; border-bottom-color: #171A20; }}
QTabBar::tab:hover:!selected {{ color: #E6E8EC; }}
QWidget#OptionsBar {{ background: #0E1014; border-top: 1px solid #262A33; }}
QWidget#OptionsBar QTabBar {{ background: #0E1014; }}
QWidget#OptionsBar QTabBar::tab {{ border-radius: 0; padding: 7px 14px; border: 0; border-top: 2px solid transparent; color: #A3AAB8; }}
QWidget#OptionsBar QTabBar::tab:selected {{ background: #171A20; color: #E6E8EC; border-top: 2px solid {ACCENT}; }}
QWidget#OptionsBody {{ background: #171A20; }}
QScrollArea {{ border: 0; background: transparent; }}
QScrollBar:vertical {{ background: transparent; width: 10px; }}
QScrollBar::handle:vertical {{ background: #343946; border-radius: 5px; min-height: 30px; }}
QScrollBar::add-line, QScrollBar::sub-line {{ height: 0; }}
QMenuBar {{ background: #0E1014; }}
QMenuBar::item:selected {{ background: #232731; }}
QMenu {{ background: #1C1F26; border: 1px solid #343946; }}
QMenu::item:selected {{ background: #2A3A52; }}
QToolTip {{ background: #1C1F26; border: 1px solid #343946; color: #E6E8EC; }}
QGroupBox {{ border: 0; margin-top: 18px; font-weight: 600; color: #A3AAB8; }}
QProgressDialog QLabel {{ color: #E6E8EC; }}
"""
