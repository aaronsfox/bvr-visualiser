"""Colour palettes (General #1): Light is a mapping of the Dark UI colours, switched while the app runs."""

import os
import re

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")

from bvrviewer.app import style as S  # noqa: E402


def test_every_stylesheet_and_icon_colour_has_a_light_value():
    used = {h.upper() for h in re.findall(r"#[0-9A-Fa-f]{6}", S.QSS + S._indicator_qss() + "".join(S._ICONS.values()))}
    assert used <= set(S._LIGHT), used - set(S._LIGHT)


def test_c_maps_ui_colours_only_in_light():
    from PySide6.QtWidgets import QApplication

    app = QApplication.instance() or QApplication([])
    seen = []
    S.theme().changed.connect(seen.append)
    try:
        S.apply_theme(app, "dark")
        assert S.C("#E6E8EC") == "#E6E8EC"
        S.apply_theme(app, "light")
        assert S.palette() == "light" and seen[-1] == "light"
        assert S.C("#E6E8EC") == S.LIGHT["#E6E8EC"]
        assert S.C("#1c1f26CC") == S.LIGHT["#1C1F26"] + "CC"          # keeps alpha; any case
        assert S.C("#F0A04B") == "#F0A04B"                              # data colours pass through
        assert app.palette().window().color().name().upper() == S.LIGHT["#121419"].upper()
        assert "#121419" not in app.styleSheet()
    finally:
        S.apply_theme(app, "dark")
    assert seen[-1] == "dark" and S.C("#E6E8EC") == "#E6E8EC"
