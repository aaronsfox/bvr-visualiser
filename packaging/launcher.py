"""Entry point for the packaged app."""

import os
import sys

os.environ.setdefault("QT_API", "pyside6")

from bvrviewer.app.main import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
