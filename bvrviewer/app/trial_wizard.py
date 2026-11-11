"""Trial setup dialog (Visualiser #17): import a trial from an Autoscoper .cfg and its tracking files, start it
from another trial, or build it by hand; and change an existing trial (Configure…).

One dialog with four pages: 1 Source, 2 Bones, 3 X-ray cameras, 4 Review. For a new trial it runs as a
wizard (Back / Next / Add trial); for an existing one the pages are tabs (Apply). It edits a
``core.trial.TrialDraft``; the tab builds the trial from it.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QImage, QPainter, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QDialog,
    QFileDialog,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QRadioButton,
    QScrollArea,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from ..core import trial as TR
from ..core.models import PALETTE, BoneModel
from ..core.tracking import load_tracking
from .panels import ColorButton, h2, label

PAGES = ("Source", "Bones", "X-ray cameras", "Review")
NONE, OTHER = "__none__", "__other__"


def _polish(w: QLabel, kind: str) -> None:
    w.setObjectName(kind)
    w.style().unpolish(w)
    w.style().polish(w)


def _path_row(text: str, placeholder: str, browse) -> tuple[QWidget, QLineEdit]:
    w = QWidget()
    h = QHBoxLayout(w)
    h.setContentsMargins(0, 0, 0, 0)
    ed = QLineEdit(text)
    ed.setPlaceholderText(placeholder)
    ed.setReadOnly(True)
    b = QPushButton("Browse…")
    b.clicked.connect(browse)
    h.addWidget(ed, 1)
    h.addWidget(b)
    return w, ed


def _voxel_text(v) -> str:
    return " × ".join(f"{x:.4g}" for x in v) if v else ""


def _parse_voxel(text: str):
    parts = [p for p in text.replace("×", " ").replace(",", " ").split() if p]
    try:
        vals = [float(p) for p in parts]
    except ValueError:
        return None
    if len(vals) == 1:
        vals = vals * 3
    return tuple(vals) if len(vals) == 3 and all(x > 0 for x in vals) else None


class TrialSetupDialog(QDialog):
    """Set up a trial: its source (.cfg, another trial or nothing), bones, X-ray cameras, then a review.

    mode "new": a wizard for a new trial. mode "configure": tabs for a trial already in the tab.
    ``trials`` are the tab's trials (to start from one, and to keep names unique).
    """

    def __init__(self, parent, draft: TR.TrialDraft | None = None, mode: str = "new", trials=(),
                 start_from=None, taken_names=()):
        super().__init__(parent)
        self.mode = mode
        self.draft = draft or TR.TrialDraft(name="")
        self.trials = list(trials)
        self.taken = set(taken_names)
        self._checks: dict = {}            # (path, overrides, volume, offset) -> (frames, tracked, warnings)
        self._cams: dict = {}              # row index -> (calibration, images, error)
        self._meshes: dict = {}            # (path, voxel, flip) -> mesh, for the camera preview
        self.setWindowTitle("Import trial" if mode == "new" else f"Configure trial: {self.draft.name}")
        self.resize(1080, 720)

        outer = QVBoxLayout(self)
        body = QHBoxLayout()
        self.steps = QListWidget()
        self.steps.setObjectName("Steps")
        self.steps.setFixedWidth(170)
        self.steps.setStyleSheet("QListWidget::item { padding: 9px 10px; }")
        for i, t in enumerate(PAGES, 1):
            QListWidgetItem(f"{i}  {t}", self.steps)
        self.steps.currentRowChanged.connect(self._go)
        body.addWidget(self.steps)
        self.stack = QStackedWidget()
        body.addWidget(self.stack, 1)
        outer.addLayout(body, 1)
        self.pages = [self._page_source(), self._page_bones(), self._page_cameras(), self._page_review()]
        for p in self.pages:
            sc = QScrollArea()
            sc.setWidgetResizable(True)
            sc.setFrameShape(QFrame.NoFrame)
            sc.setWidget(p)
            self.stack.addWidget(sc)
        bar = QHBoxLayout()
        self.status = label("", "Muted")
        bar.addWidget(self.status, 1)
        self.back = QPushButton("Back")
        self.back.clicked.connect(lambda: self.steps.setCurrentRow(self.steps.currentRow() - 1))
        self.next = QPushButton("Next")
        self.next.setObjectName("Primary")
        self.next.clicked.connect(lambda: self.steps.setCurrentRow(self.steps.currentRow() + 1))
        self.finish = QPushButton("Add trial" if mode == "new" else "Apply")
        self.finish.setObjectName("Primary")
        self.finish.clicked.connect(self._finish)
        cancel = QPushButton("Cancel")
        cancel.clicked.connect(self.reject)
        for b in (cancel, self.back, self.next, self.finish):
            bar.addWidget(b)
        outer.addLayout(bar)
        if start_from is not None:
            self.start_from_trial(start_from)
        self._fill_bones()
        self._fill_cameras()
        self.steps.setCurrentRow(0)

    # ================================================================ page 1: source
    def _page_source(self) -> QWidget:
        w = QWidget()
        v = QVBoxLayout(w)
        v.addWidget(h2("Trial"))
        g = QGridLayout()
        g.setColumnStretch(1, 1)
        self.name = QLineEdit(self.draft.name)
        self.name.setPlaceholderText("e.g. T0019 jog")
        self._name_edited = self.mode != "new"
        self.name.textEdited.connect(self._name_typed)
        g.addWidget(QLabel("Name"), 0, 0)
        g.addWidget(self.name, 0, 1)
        v.addLayout(g)
        self.start_box = QWidget()
        sv = QVBoxLayout(self.start_box)
        sv.setContentsMargins(0, 8, 0, 0)
        sv.addWidget(h2("Start from"))
        self.start_group = QButtonGroup(self)
        self.r_cfg = QRadioButton("An Autoscoper trial file (.cfg): its bones (volumes) and X-ray cameras, as listed")
        self.r_trial = QRadioButton("Another trial in this tab: the same bones, coordinate systems, joints and "
                                    "camera calibrations, with new tracking and radiographs")
        self.r_none = QRadioButton("Nothing: add the bones and cameras on the next pages")
        for k, r in enumerate((self.r_cfg, self.r_trial, self.r_none)):
            self.start_group.addButton(r, k)
        cfg_row, self.cfg_edit = _path_row(str(self.draft.cfg_path or ""), "Choose the trial's .cfg file",
                                           self._browse_cfg)
        self.trial_combo = QComboBox()
        for t in self.trials:
            self.trial_combo.addItem(t.name, t)
        self.trial_combo.currentIndexChanged.connect(
            lambda i: self.r_trial.isChecked() and i >= 0 and self.start_from_trial(self.trial_combo.itemData(i)))
        self.r_trial.setEnabled(bool(self.trials))
        sv.addWidget(self.r_cfg)
        sv.addWidget(self._indent(cfg_row))
        sv.addWidget(self.r_trial)
        sv.addWidget(self._indent(self.trial_combo))
        sv.addWidget(self.r_none)
        self.r_cfg.setChecked(True)
        self.start_group.idToggled.connect(self._start_changed)
        v.addWidget(self.start_box)
        self.start_box.setVisible(self.mode == "new")
        if self.mode != "new" and self.draft.cfg_path:
            v.addWidget(label(f"Imported from {Path(self.draft.cfg_path).name}", "Muted"))
        v.addWidget(h2("Tracking files"))
        trk_row, self.trk_edit = _path_row(str(self.draft.tracking_dir or ""), "The folder with the .tra files",
                                           self._browse_tracking)
        v.addWidget(trk_row)
        self.trk_note = label("", "Muted")
        v.addWidget(self.trk_note)
        v.addWidget(label("Each bone is given the tracking file whose name matches it (the part all the files share, "
                          "e.g. T0019_jog0001_, is ignored). Check and change the matches on the next page.", "Muted"))
        v.addWidget(h2("Model (optional)"))
        mod_row, self.model_edit = _path_row(str(self.draft.model_path or ""), "A model from the Model Builder",
                                             self._browse_model)
        v.addWidget(mod_row)
        v.addWidget(label("Its bones' coordinate systems and its joints are added for the bones with matching names.",
                          "Muted"))
        clear = QPushButton("No model")
        clear.setObjectName("Small")
        clear.clicked.connect(lambda: (setattr(self.draft, "model_path", None), self.model_edit.setText("")))
        v.addWidget(clear, 0, Qt.AlignLeft)
        v.addStretch(1)
        self._update_tracking_note()
        return w

    def _name_typed(self, t: str) -> None:
        self.draft.name = t.strip()
        self._name_edited = bool(t.strip())
        self._update_review()

    @staticmethod
    def _indent(w: QWidget) -> QWidget:
        box = QWidget()
        h = QHBoxLayout(box)
        h.setContentsMargins(24, 0, 0, 6)
        h.addWidget(w, 1)
        return box

    def _start_changed(self, k: int, on: bool) -> None:
        if not on:
            return
        if k == 1 and self.trial_combo.count():
            self.start_from_trial(self.trial_combo.currentData())
        elif k == 2:
            name = self.draft.name
            self.draft = TR.TrialDraft(name=name, tracking_dir=self.draft.tracking_dir, model_path=self.draft.model_path)
            self.draft.match()
            self._fill_bones()
            self._fill_cameras()

    def _browse_cfg(self) -> None:
        p, _ = QFileDialog.getOpenFileName(self, "Autoscoper trial file", str(self.draft.cfg_path or ""),
                                           "Autoscoper configuration (*.cfg);;All files (*)")
        if p:
            self.set_cfg(p)

    def set_cfg(self, path) -> None:
        """Use a .cfg: its volumes become the bones and its cameras the X-ray cameras (tracking matched again)."""
        try:
            folder = self.draft.tracking_dir or TR.guess_tracking_dir(path)
            name = self.name.text().strip()
            d = TR.draft_from_cfg(path, folder)
        except Exception as e:  # noqa: BLE001 - shown to the user
            self.status.setText(f"Couldn't read {Path(path).name}: {e}")
            _polish(self.status, "Error")
            return
        if self._name_edited and name:
            d.name = name                          # a name the user typed stays
        d.model_path = self.draft.model_path
        self.draft = d
        self.r_cfg.setChecked(True)
        self.cfg_edit.setText(str(path))
        self.trk_edit.setText(str(d.tracking_dir or ""))
        self.name.setText(d.name)
        self.status.setText("")
        self._update_tracking_note()
        self._fill_bones()
        self._fill_cameras()

    def start_from_trial(self, t) -> None:
        """Start from another trial: its bones (with their settings, coordinate systems and joints) and camera
        calibrations, with no tracking files or radiographs yet."""
        d = TR.draft_from_trial(t)
        for b in d.bones:
            b.tracking_path = None
        for c in d.cameras:
            c.folder, c.source, c.source_kind = None, [], "folder"
        d.joints = [j for j in t.joints]
        d.name = self.draft.name if self._name_edited and self.draft.name else self._unique(f"{t.name} (2)")
        d.cfg_path, d.tracking_dir = t.cfg_path, None
        d.model_path = self.draft.model_path
        self.draft = d
        self.r_trial.setChecked(True)
        i = next((k for k in range(self.trial_combo.count()) if self.trial_combo.itemData(k) is t), -1)
        if i >= 0 and self.trial_combo.currentIndex() != i:
            self.trial_combo.blockSignals(True)
            self.trial_combo.setCurrentIndex(i)
            self.trial_combo.blockSignals(False)
        self.name.setText(d.name)
        self.trk_edit.setText("")
        self._update_tracking_note()
        self._fill_bones()
        self._fill_cameras()

    def _unique(self, name: str) -> str:
        if name not in self.taken:
            return name
        k = 2
        base = name.removesuffix(" (2)")
        while f"{base} ({k})" in self.taken:
            k += 1
        return f"{base} ({k})"

    def _browse_tracking(self) -> None:
        start = str(self.draft.tracking_dir or (Path(self.draft.cfg_path).parent if self.draft.cfg_path else ""))
        p = QFileDialog.getExistingDirectory(self, "Folder with the tracking files", start)
        if p:
            self.set_tracking_dir(p)

    def set_tracking_dir(self, folder) -> None:
        name = self.draft.name
        TR.set_tracking_dir(self.draft, folder)
        if self._name_edited and name:
            self.draft.name = name
        self.trk_edit.setText(str(folder))
        self.name.setText(self.draft.name)
        self._update_tracking_note()
        self._fill_bones()

    def _update_tracking_note(self) -> None:
        files = self.draft.tracking_files()
        if self.draft.tracking_dir is None:
            self.trk_note.setText("No folder chosen: bones can still be given tracking files one by one.")
        elif not files:
            self.trk_note.setText("No .tra files in this folder.")
        else:
            self.trk_note.setText(f"{len(files)} tracking files: " + ", ".join(f.name for f in files[:4]) +
                                  (" …" if len(files) > 4 else ""))

    def _browse_model(self) -> None:
        p, _ = QFileDialog.getOpenFileName(self, "Model from the Model Builder", "",
                                           "Model from the Model Builder (*.bvrmodel.json *.json)")
        if p:
            self.draft.model_path = Path(p)
            self.model_edit.setText(p)

    # ================================================================ page 2: bones
    def _page_bones(self) -> QWidget:
        w = QWidget()
        v = QVBoxLayout(w)
        v.addWidget(h2("Bones"))
        v.addWidget(label("Untick a bone to leave it out. Each bone's surface must be the file it was tracked with "
                          "(same model space). Pick the tracking file for any bone left without one.", "Muted"))
        self.bone_grid = QGridLayout()
        self.bone_grid.setHorizontalSpacing(8)
        self.bone_grid.setVerticalSpacing(6)
        v.addLayout(self.bone_grid)
        h = QHBoxLayout()
        add = QPushButton("+ Add bone…")
        add.clicked.connect(self._add_bone)
        h.addWidget(add)
        self.fmt_label = label("", "Muted")
        h.addWidget(self.fmt_label, 1)
        self.fmt_btn = QPushButton("Tracking format…")
        self.fmt_btn.setToolTip("How the tracking files were saved from Autoscoper (applies to every bone)")
        self.fmt_btn.clicked.connect(self._edit_format)
        h.addWidget(self.fmt_btn)
        v.addLayout(h)
        self.unused = label("", "Muted")
        v.addWidget(self.unused)
        v.addStretch(1)
        return w

    def _fill_bones(self) -> None:
        g = self.bone_grid
        while g.count():
            it = g.takeAt(0)
            if it.widget():
                it.widget().hide()              # (deleted later: hide it now so it doesn't linger on the page)
                it.widget().deleteLater()
        self.bone_rows = []
        for c, t in enumerate(("", "Name", "Surface", "Colour", "Tracking file", "Check", "")):
            g.addWidget(h2(t), 0, c)
        files = self.draft.tracking_files()
        for r, b in enumerate(self.draft.bones, 1):
            inc = QCheckBox()
            inc.setChecked(b.include)
            name = QLineEdit(b.name)
            surf = QWidget()
            sh = QHBoxLayout(surf)
            sh.setContentsMargins(0, 0, 0, 0)
            sl = QLabel(Path(b.path).name)
            sl.setToolTip(str(b.path))
            sh.addWidget(sl)
            vox = None
            if b.is_volume:
                vox = QLineEdit(_voxel_text(b.voxel_size))
                vox.setPlaceholderText("voxel size, mm")
                vox.setToolTip("Voxel size in mm (x × y × z), from the .cfg; TIFs don't store it")
                vox.setFixedWidth(170)
                sh.addWidget(vox)
            sh.addStretch(1)
            col = ColorButton(b.color)
            trk = QComboBox()
            trk.addItem("(none)", NONE)
            for f in files:
                trk.addItem(f.name, str(f))
            if b.tracking_path is not None and str(b.tracking_path) not in [trk.itemData(i) for i in range(trk.count())]:
                trk.addItem(Path(b.tracking_path).name, str(b.tracking_path))
            trk.addItem("Other file…", OTHER)
            trk.setCurrentIndex(trk.findData(str(b.tracking_path)) if b.tracking_path else 0)
            trk.setMinimumWidth(220)
            check = QLabel()
            check.setWordWrap(True)
            rm = QPushButton("✕")
            rm.setObjectName("Small")
            rm.setToolTip("Remove this bone from the list")
            row = {"bone": b, "inc": inc, "name": name, "vox": vox, "col": col, "trk": trk, "check": check}
            inc.toggled.connect(lambda on, b=b: (setattr(b, "include", on), self._update_bones()))
            name.textEdited.connect(lambda t, b=b: (setattr(b, "name", t.strip()), self._update_bones()))
            if vox is not None:
                vox.editingFinished.connect(lambda b=b, e=vox: self._set_voxel(b, e))
            col.colorChanged.connect(lambda c, b=b: setattr(b, "color", c))
            trk.activated.connect(lambda _i, b=b, w=trk: self._pick_tracking(b, w))
            rm.clicked.connect(lambda _=False, b=b: self._remove_bone(b))
            for c, wdg in enumerate((inc, name, surf, col, trk, check, rm)):
                g.addWidget(wdg, r, c)
            self.bone_rows.append(row)
        if not self.draft.bones:
            g.addWidget(label("No bones yet: choose a .cfg on the first page, or add bones here.", "Muted"), 1, 1, 1, 5)
        g.setColumnStretch(2, 1)
        g.setColumnStretch(5, 1)
        self._update_bones()

    def _set_voxel(self, b: TR.BoneDraft, ed: QLineEdit) -> None:
        v = _parse_voxel(ed.text())
        b.voxel_size = v
        if v is not None:
            ed.setText(_voxel_text(v))
        self._update_bones()

    def _pick_tracking(self, b: TR.BoneDraft, w: QComboBox) -> None:
        data = w.currentData()
        if data == OTHER:
            start = str(self.draft.tracking_dir or Path(b.path).parent)
            p, _ = QFileDialog.getOpenFileName(self, f"Tracking file for {b.name}", start,
                                               "Autoscoper tracking (*.tra *.csv *.txt);;All files (*)")
            if p:
                b.tracking_path = Path(p)
            self._fill_bones()
            return
        b.tracking_path = None if data == NONE else Path(data)
        self._update_bones()

    def _remove_bone(self, b: TR.BoneDraft) -> None:
        self.draft.bones = [x for x in self.draft.bones if x is not b]
        self._fill_bones()

    def _add_bone(self) -> None:
        start = str(self.draft.cfg_path.parent if self.draft.cfg_path else "")
        ps, _ = QFileDialog.getOpenFileNames(self, "Add bones: their surfaces", start,
                                             "Surfaces and volumes (*.stl *.ply *.obj *.vtk *.vtp *.tif *.tiff)")
        for p in ps:
            self.add_bone_file(p, match=False)
        if ps:
            self.draft.match()
            self._fill_bones()

    def add_bone_file(self, path, match: bool = True) -> TR.BoneDraft:
        p = Path(path)
        k = len(self.draft.bones)
        vox = next((b.voxel_size for b in self.draft.bones if b.voxel_size), None)
        b = TR.BoneDraft(name=p.stem, path=p, color=PALETTE[k % len(PALETTE)],
                         voxel_size=vox if p.suffix.lower() in (".tif", ".tiff") else None)
        self.draft.bones.append(b)
        if match:
            self.draft.match()
            self._fill_bones()
        return b

    def _check(self, b: TR.BoneDraft):
        key = (str(b.tracking_path), tuple(sorted(b.tracking_overrides.items())), b.tracking_volume,
               b.tracking_offset)
        if key not in self._checks:
            try:
                t = load_tracking(b.tracking_path, b.tracking_volume, b.tracking_overrides, b.tracking_offset)
                self._checks[key] = (t.n_frames, int(t.valid.sum()), list(t.warnings), t.fmt)
            except Exception as e:  # noqa: BLE001 - shown in the row
                self._checks[key] = (None, None, [str(e)], None)
        return self._checks[key]

    def _update_bones(self) -> None:
        frames = {}
        fmt = None
        for row in self.bone_rows:
            b = row["bone"]
            if b.include and b.tracking_path is not None and Path(b.tracking_path).exists():
                n, tracked, warns, f = self._check(b)
                if n is not None:
                    frames[b.name] = n
                    fmt = fmt or f
        common = max(set(frames.values()), key=list(frames.values()).count) if frames else None
        for row in self.bone_rows:
            b, lab = row["bone"], row["check"]
            for k in ("name", "vox", "col", "trk"):
                if row[k] is not None:
                    row[k].setEnabled(b.include)
            if not b.include:
                lab.setText("Left out")
                _polish(lab, "Muted")
                continue
            if b.is_volume and not b.voxel_size:
                lab.setText("Needs its voxel size")
                _polish(lab, "Error")
                continue
            if b.tracking_path is None:
                lab.setText("No tracking: stays in its model space")
                _polish(lab, "Warn")
                continue
            if not Path(b.tracking_path).exists():
                lab.setText("Tracking file not found")
                _polish(lab, "Error")
                continue
            n, tracked, warns, _f = self._check(b)
            if n is None:
                lab.setText(warns[0] if warns else "Couldn't read the tracking file")
                _polish(lab, "Error")
                continue
            text = f"{n} frames, " + ("all tracked" if tracked == n else f"{tracked} tracked")
            kind = "Ok"
            also = [x.name for x in self.draft.bones if x is not b and x.include and x.tracking_path is not None
                    and Path(x.tracking_path) == Path(b.tracking_path)]
            if also:
                text = f"Same file as {', '.join(also)} · " + text
                kind = "Warn"
            if common is not None and n != common:
                text += f" (others have {common})"
                kind = "Warn"
            if warns:
                text += " · " + warns[0]
                kind = "Warn"
            lab.setText(text)
            _polish(lab, kind)
        self.fmt_label.setText(f"Tracking format: {fmt.describe()}" if fmt is not None else "")
        self.fmt_btn.setEnabled(fmt is not None)
        unused = self.draft.unused_tracking()
        self.unused.setText(("Tracking files not used: " + ", ".join(f.name for f in unused)) if unused else "")
        self._update_review()

    def _edit_format(self) -> None:
        from ..core.tracking import load_tracking as lt
        from .dialogs import FormatDialog

        b = next((x for x in self.draft.bones if x.include and x.tracking_path and Path(x.tracking_path).exists()),
                 None)
        if b is None:
            return
        detected = lt(b.tracking_path)
        current = lt(b.tracking_path, b.tracking_volume, b.tracking_overrides, b.tracking_offset)
        dlg = FormatDialog(self, current.fmt, b.tracking_volume, path=b.tracking_path, detected=detected.fmt,
                           offset=b.tracking_offset, n_rows=detected.n_rows, bone=b.name)
        if dlg.exec() != QDialog.Accepted:
            return
        (overrides, vol), offset = dlg.values(), dlg.offset
        for x in self.draft.bones:                  # the same format for every bone; volume and start for this one
            x.tracking_overrides = dict(overrides)
        b.tracking_volume, b.tracking_offset = vol, offset
        self._update_bones()

    # ================================================================ page 3: X-ray cameras
    def _page_cameras(self) -> QWidget:
        w = QWidget()
        outer = QHBoxLayout(w)
        left = QVBoxLayout()
        left.addWidget(h2("X-ray cameras"))
        left.addWidget(label("Optional. Untick a camera to leave it out. Preview draws the bones (at their first "
                             "tracked frame) on the first radiograph: if they sit on their shadows, the calibration, "
                             "images and tracking match.", "Muted"))
        self.cam_grid = QGridLayout()
        self.cam_grid.setHorizontalSpacing(8)
        self.cam_grid.setVerticalSpacing(6)
        left.addLayout(self.cam_grid)
        h = QHBoxLayout()
        add = QPushButton("+ Add camera…")
        add.clicked.connect(self._add_camera)
        h.addWidget(add)
        h.addStretch(1)
        left.addLayout(h)
        left.addWidget(h2("Radiographs"))
        self.read_now = QRadioButton("Read them all when the trial is first shown (recommended)")
        self.read_now.setToolTip("Playback is then smooth. A progress bar shows the reading, with Skip")
        self.read_later = QRadioButton("Read each one as its frame is shown")
        self.read_later.setToolTip("Starts sooner, but playback can stall, especially from a network drive")
        (self.read_now if self.draft.read_radiographs else self.read_later).setChecked(True)
        self.read_now.toggled.connect(lambda on: setattr(self.draft, "read_radiographs", on))
        left.addWidget(self.read_now)
        left.addWidget(self.read_later)
        left.addStretch(1)
        outer.addLayout(left, 3)
        right = QVBoxLayout()
        right.setSizeConstraint(QVBoxLayout.SetFixedSize)
        right.addWidget(h2("Preview"))
        self.preview = QLabel("Press Preview on a camera.")
        self.preview.setAlignment(Qt.AlignCenter)
        self.preview.setFixedSize(300, 300)
        self.preview.setObjectName("Muted")
        right.addWidget(self.preview)
        self.preview_note = label("", "Muted")
        self.preview_note.setFixedWidth(300)
        right.addWidget(self.preview_note)
        right.addStretch(1)
        outer.addLayout(right, 0)
        return w

    def _fill_cameras(self) -> None:
        g = self.cam_grid
        while g.count():
            it = g.takeAt(0)
            if it.widget():
                it.widget().hide()              # (deleted later: hide it now so it doesn't linger on the page)
                it.widget().deleteLater()
        self._cams = {}
        self.cam_rows = []
        for c, t in enumerate(("", "Name", "Calibration", "Radiographs", "Marker", "")):
            g.addWidget(h2(t), 0, c)
        for r, c in enumerate(self.draft.cameras, 1):
            inc = QCheckBox()
            inc.setChecked(c.include)
            name = QLineEdit(c.name)
            name.setFixedWidth(110)
            cal = QLabel()
            cal.setWordWrap(True)
            imgs = QWidget()
            ih = QHBoxLayout(imgs)
            ih.setContentsMargins(0, 0, 0, 0)
            img_lab = QLabel()
            img_lab.setWordWrap(True)
            ih.addWidget(img_lab, 1)
            br = QPushButton("Folder…")
            br.setObjectName("Small")
            ih.addWidget(br)
            marker = QCheckBox()
            marker.setChecked(c.show_source)
            marker.setToolTip("Show the X-ray source as a marker in the 3D view")
            prev = QPushButton("Preview")
            prev.setObjectName("Small")
            row = {"cam": c, "inc": inc, "name": name, "cal": cal, "img": img_lab, "marker": marker, "prev": prev,
                   "r": r}
            inc.toggled.connect(lambda on, c=c: (setattr(c, "include", on), self._update_review()))
            name.textEdited.connect(lambda t, c=c: setattr(c, "name", t.strip() or c.name))
            br.clicked.connect(lambda _=False, row=row: self._browse_radiographs(row))
            marker.toggled.connect(lambda on, c=c: setattr(c, "show_source", on))
            prev.clicked.connect(lambda _=False, row=row: self.preview_camera(row))
            for k, wdg in enumerate((inc, name, cal, imgs, marker, prev)):
                g.addWidget(wdg, r, k)
            self.cam_rows.append(row)
            self._describe_camera(row)
        if not self.draft.cameras:
            g.addWidget(label("No X-ray cameras. They're optional; add them here or later from the navigator.",
                              "Muted"), 1, 1, 1, 4)
        g.setColumnStretch(2, 1)
        g.setColumnStretch(3, 1)
        self._update_review()

    def _camera_parts(self, row):
        """(calibration, images, error) for a camera row, read once."""
        from ..core.calibration import load_calibration
        from ..core.images import ImageSequence, find_images

        key = row["r"]
        if key not in self._cams:
            c = row["cam"]
            cal = imgs = None
            err = []
            try:
                cal = load_calibration(c.calibration_path)
            except Exception as e:  # noqa: BLE001
                err.append(f"calibration: {e}")
            try:
                if c.source_kind == "folder":
                    if c.folder is None:
                        raise ValueError("choose the radiograph folder")
                    files, _skipped = find_images(c.folder)
                    imgs = ImageSequence(files=files)
                elif c.source_kind == "multipage":
                    imgs = ImageSequence(multipage=Path(c.source[0]))
                else:
                    imgs = ImageSequence(files=sorted(Path(p) for p in c.source))
            except Exception as e:  # noqa: BLE001
                err.append(f"radiographs: {e}")
            self._cams[key] = (cal, imgs, err)
        return self._cams[key]

    def _describe_camera(self, row) -> None:
        cal, imgs, err = self._camera_parts(row)
        c = row["cam"]
        if cal is not None:
            row["cal"].setText(f"{Path(c.calibration_path).name}\n{cal.summary()}")
            _polish(row["cal"], "Value")
        else:
            row["cal"].setText(f"{Path(c.calibration_path).name}\n" + next((e for e in err if e.startswith("cal")), ""))
            _polish(row["cal"], "Error")
        where = Path(c.folder).name if c.folder else (Path(c.source[0]).name if c.source else "")
        if imgs is not None:
            text = f"{where}\n{len(imgs)} images · {imgs.width} × {imgs.height}"
            kind = "Value"
            if cal is not None and (imgs.width, imgs.height) != (cal.width, cal.height):
                text += f" (calibration expects {cal.width} × {cal.height})"
                kind = "Warn"
            row["img"].setText(text)
            _polish(row["img"], kind)
        else:
            row["img"].setText((where + "\n" if where else "") + next((e for e in err if e.startswith("rad")), ""))
            _polish(row["img"], "Error" if c.include else "Muted")
        row["prev"].setEnabled(cal is not None and imgs is not None)

    def _browse_radiographs(self, row) -> None:
        c = row["cam"]
        start = str(c.folder or (Path(self.draft.cfg_path).parent if self.draft.cfg_path else ""))
        p = QFileDialog.getExistingDirectory(self, f"Radiographs for {c.name}", start)
        if p:
            c.folder, c.source_kind, c.source = Path(p), "folder", [Path(p)]
            self._cams.pop(row["r"], None)
            self._describe_camera(row)
            self._update_review()

    def _add_camera(self) -> None:
        start = str(self.draft.cfg_path.parent if self.draft.cfg_path else "")
        p, _ = QFileDialog.getOpenFileName(self, "Camera calibration", start,
                                           "Calibration (*.txt *.csv *.yaml *.json);;All files (*)")
        if not p:
            return
        self.draft.cameras.append(TR.CameraDraft(name=f"X-ray {len(self.draft.cameras) + 1}", calibration_path=Path(p)))
        self._fill_cameras()

    def _bone_points(self, b: TR.BoneDraft):
        """Points of a bone's surface at its first tracked frame (for the preview), or None."""
        if b.tracking_path is None or not Path(b.tracking_path).exists():
            return None
        key = (str(b.path), b.voxel_size, tuple(b.flip))
        if key not in self._meshes:
            src = b.source
            if src is not None and getattr(src, "mesh", None) is not None and str(src.path) == str(b.path):
                self._meshes[key] = src.mesh
            else:
                m = BoneModel(name=b.name, path=Path(b.path), voxel_size=b.voxel_size, flip=b.flip)
                m.build()
                self._meshes[key] = m.mesh
        mesh = self._meshes[key]
        t = load_tracking(b.tracking_path, b.tracking_volume, b.tracking_overrides, b.tracking_offset)
        valid = np.nonzero(t.valid)[0]
        if not len(valid):
            return None
        T = t.poses[valid[0]]
        pts = np.asarray(mesh.points)[:: max(1, mesh.n_points // 3000)]
        return (np.c_[pts, np.ones(len(pts))] @ T.T)[:, :3], int(valid[0])

    def preview_camera(self, row) -> None:
        cal, imgs, _err = self._camera_parts(row)
        if cal is None or imgs is None:
            return
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            bones, problems, frame = [], [], None
            for b in self.draft.bones:
                if not b.include:
                    continue
                try:
                    r = self._bone_points(b)
                except Exception as e:  # noqa: BLE001
                    problems.append(f"{b.name}: {e}")
                    continue
                if r is not None:
                    bones.append((r[0], b.color))
                    frame = r[1] if frame is None else frame
            i = min(frame or 0, len(imgs) - 1)
            img = np.ascontiguousarray(imgs.get(i))
        finally:
            QApplication.restoreOverrideCursor()
        h, w = img.shape
        qimg = QImage(img.data, w, h, w, QImage.Format_Grayscale8).convertToFormat(QImage.Format_RGB32)
        p = QPainter(qimg)
        p.setRenderHint(QPainter.Antialiasing)
        dot = max(4, w // 110)                  # stays visible once scaled down to the preview
        for pts, color in bones:
            uv = cal.project(pts)
            c = QColor(color)
            c.setAlpha(210)
            p.setPen(QColor(0, 0, 0, 150))       # a dark edge, so pale bones show on bright radiographs
            p.setBrush(c)
            for u, vv in uv[:: max(1, len(uv) // 1500)]:
                if 0 <= u < w and 0 <= vv < h:
                    p.drawEllipse(int(u) - dot // 2, int(vv) - dot // 2, dot, dot)
        p.end()
        self._preview_image = qimg               # (kept for scripts)
        self.preview.setPixmap(QPixmap.fromImage(qimg).scaled(300, 300, Qt.KeepAspectRatio, Qt.SmoothTransformation))
        note = f"{row['cam'].name}, image {i + 1} ({imgs.name(i)})"
        note += f" with {len(bones)} bones at frame {i + 1}" if bones else ": no tracked bones to draw"
        if problems:
            note += "\n" + "\n".join(problems)
        self.preview_note.setText(note)

    # ================================================================ page 4: review
    def _page_review(self) -> QWidget:
        w = QWidget()
        v = QVBoxLayout(w)
        v.addWidget(h2("Review"))
        self.summary = label("")
        self.summary.setTextFormat(Qt.RichText)
        v.addWidget(self.summary)
        self.issues = label("")
        self.issues.setTextFormat(Qt.RichText)
        v.addWidget(self.issues)
        v.addStretch(1)
        return w

    def problems(self) -> list[str]:
        out = list(self.draft.problems())
        if not self.draft.name.strip():
            out.insert(0, "Give the trial a name (page 1).")
        elif self.draft.name.strip() in self.taken:
            out.insert(0, f"There's already a trial called “{self.draft.name.strip()}”: choose another name.")
        for row in getattr(self, "cam_rows", []):
            c = row["cam"]
            if c.include:
                _cal, _imgs, err = self._camera_parts(row)
                out.extend(f"{c.name}: {e}" for e in err)
        return out

    def warnings(self) -> list[str]:
        out = []
        for row in getattr(self, "bone_rows", []):
            b = row["bone"]
            if b.include and row["check"].objectName() == "Warn":
                out.append(f"{b.name}: {row['check'].text()}")
        unused = self.draft.unused_tracking()
        if unused:
            out.append("Tracking files not used: " + ", ".join(f.name for f in unused))
        return out

    def _update_review(self) -> None:
        if not hasattr(self, "summary"):
            return
        d = self.draft
        bones = [b for b in d.bones if b.include]
        cams = [c for c in d.cameras if c.include]
        tracked = [b for b in bones if b.tracking_path]
        lines = [f"<b>{d.name or '(no name)'}</b>",
                 f"Bones: {len(bones)}" + (f" ({len(tracked)} with tracking)" if bones else "") +
                 (": " + ", ".join(b.name for b in bones) if bones else ""),
                 f"X-ray cameras: {len(cams)}" + (": " + ", ".join(c.name for c in cams) if cams else "")]
        if cams:
            lines.append("Radiographs: " + ("read when the trial is first shown" if d.read_radiographs
                                            else "read as each frame is shown"))
        if d.model_path:
            lines.append(f"Model: {Path(d.model_path).name} (coordinate systems and joints for matching bones)")
        if getattr(d, "joints", None):
            lines.append(f"Joints: {len(d.joints)} from the trial it started from")
        self.summary.setText("<br>".join(lines))
        probs, warns = self.problems(), self.warnings()
        html = ""
        if probs:
            html += "<p style='color:#F28B82'><b>To fix before the trial can be added:</b><br>" + \
                    "<br>".join(probs) + "</p>"
        if warns:
            html += "<p style='color:#F2C48D'><b>Check:</b><br>" + "<br>".join(warns) + "</p>"
        if not probs and not warns:
            html = "<p style='color:#8FD3A9'>Everything is ready.</p>"
        self.issues.setText(html)
        last = self.steps.currentRow() == len(PAGES) - 1 if hasattr(self, "steps") else False
        if hasattr(self, "finish"):
            self.finish.setEnabled(not probs and (self.mode != "new" or last))

    # ================================================================ navigation
    def _go(self, i: int) -> None:
        if i < 0:
            return
        self.stack.setCurrentIndex(i)
        new = self.mode == "new"
        self.back.setVisible(new)
        self.next.setVisible(new)
        self.back.setEnabled(i > 0)
        self.next.setEnabled(i < len(PAGES) - 1)
        self.finish.setVisible(not new or i == len(PAGES) - 1)
        self.next.setDefault(new and i < len(PAGES) - 1)
        self._update_review()

    def _finish(self) -> None:
        self.draft.name = self.name.text().strip()
        if self.problems():
            self.steps.setCurrentRow(len(PAGES) - 1)
            return
        self.accept()
