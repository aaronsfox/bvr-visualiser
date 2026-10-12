"""Dark theme matching the design mock-up."""

ACCENT = "#4F9CF9"

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
QTabWidget::pane {{ border: 0; border-top: 1px solid #262A33; }}
QTabBar::tab {{ background: transparent; color: #A3AAB8; padding: 9px 14px; border: 1px solid transparent; border-top-left-radius: 8px; border-top-right-radius: 8px; }}
QTabBar::tab:selected {{ background: #171A20; color: #E6E8EC; border-color: #262A33; border-bottom-color: #171A20; }}
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
