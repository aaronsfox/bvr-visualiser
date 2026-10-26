"""The kinds of tab the app can open.

Each tool is a card on the Start page and a kind of tab. To add a new tool later: add a ToolType
here, write its tab widget (a QWidget with `state.name`, `titleChanged` and `close_tab()`), and
register it in MainWindow.TAB_CLASSES. Until then it shows on the Start page as "Coming soon".
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ToolType:
    key: str            # stable id, also written into session files
    title: str          # card title
    tab_title: str      # default tab name ("Visualiser 1", ...)
    blurb: str          # one-sentence description on the card
    details: tuple      # short bullet points on the card
    color: str          # card accent and tab dot
    available: bool     # False = shown on the Start page as "Coming soon"


TOOLS: list[ToolType] = [
    ToolType(
        key="visualiser",
        title="Visualiser",
        tab_title="Visualiser",
        blurb="Watch tracked bone motion in 3D, with the radiographs if you have them.",
        details=("Bone models (STL or TIF), each with its own tracking file",
                 "Optional X-ray cameras, radiographs behind or through the view",
                 "Screenshots, video export and saved sessions"),
        color="#4F9CF9",
        available=True,
    ),
    ToolType(
        key="model_builder",
        title="Model Builder",
        tab_title="Model",
        blurb="Define joints on your bone models from ISB or published templates, and save the model.",
        details=("Pick landmarks, or paint a surface region and use its centre",
                 "Templates (ISB, published foot models) or your own joints",
                 "Saved models feed joint angles into the other tools"),
        color="#B78AF0",
        available=True,
    ),
    ToolType(
        key="reliability",
        title="Reliability Analysis",
        tab_title="Reliability",
        blurb="Compare repeated tracking of the same trial, by one tester or several.",
        details=("Bone models plus several tracking results per bone",
                 "Within-tester and between-tester comparisons",
                 "Accuracy and reliability metrics"),
        color="#5CC8A8",
        available=False,
    ),
    ToolType(
        key="validity",
        title="Validity Analysis",
        tab_title="Validity",
        blurb="Check tracking from one or more testers against criterion data.",
        details=("Bone models, criterion tracking data and testers' tracking",
                 "Errors against the criterion, per tester and per bone",
                 "Accuracy results"),
        color="#F0A04B",
        available=False,
    ),
]


def tool(key: str) -> ToolType:
    for t in TOOLS:
        if t.key == key:
            return t
    raise KeyError(key)
