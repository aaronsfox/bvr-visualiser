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
        details=("Bone surfaces (STL or TIF), each with its own tracking file",
                 "Optional X-ray cameras, radiographs behind or through the view",
                 "Screenshots, video export and saved sessions"),
        color="#4F9CF9",
        available=True,
    ),
    ToolType(
        key="model_builder",
        title="Model Builder",
        tab_title="Model",
        blurb="Define axes on your bone surfaces, then optional joints and reference poses, and save the model.",
        details=("Axes from automatic fits or landmarks you pick or paint",
                 "Joints from published definitions (e.g. Grood & Suntay knee)",
                 "Reference pose from a neutral trial or the scan"),
        color="#B78AF0",
        available=True,
    ),
    ToolType(
        key="assessor",
        title="Assessor Evaluation",
        tab_title="Assessor",
        blurb="How reliable is each rater's tracking, and how close is it to a criterion?",
        details=("Several raters' tracking of a trial, one or more sessions each",
                 "Inter- and intra-rater reliability; validity against criterion tracking",
                 "Helical and surface errors, violin plots, statistics and CSV export"),
        color="#5CC8A8",
        available=True,
    ),
]


def tool(key: str) -> ToolType:
    for t in TOOLS:
        if t.key == key:
            return t
    raise KeyError(key)
