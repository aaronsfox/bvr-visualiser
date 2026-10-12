"""Start the BVR Motion Viewer."""

from __future__ import annotations

import os
import sys


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv if argv is None else argv)
    os.environ.setdefault("QT_API", "pyside6")
    from PySide6.QtWidgets import QApplication

    import pyvista as pv

    pv.global_theme.allow_empty_mesh = True
    app = QApplication.instance() or QApplication(argv)
    app.setApplicationName("BVR Motion Viewer")
    from .main_window import MainWindow
    from .style import QSS

    app.setStyleSheet(QSS)
    win = MainWindow()
    win.show()
    if "--selftest" in argv:
        from .selftest import run

        i = argv.index("--selftest")
        run(win, argv[i + 1], argv[i + 2])
        return app.exec()
    # Optional: open a saved session given on the command line.
    for a in argv[1:]:
        if a.endswith(".json") and os.path.exists(a):
            win.current().load_session_file(a)
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
