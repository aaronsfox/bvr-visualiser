"""What each axes method needs and what the user does for it (Model Builder #6).

``guide(source, segment_type)`` returns a ``Guide``: one line on what the method needs, the steps
the user takes, and cautions. The Model Builder shows the steps under the Method dropdown and the
full text (with how the method works and its references) from its Instructions… button.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Guide:
    needs: str
    steps: list
    cautions: list = field(default_factory=list)


_AUTO_STEPS = [
    "Check the bone's side (step 2): it decides which way is right, and medial from lateral.",
    "Press Apply (step 5). The fit takes a few seconds the first time.",
    "Tick Show fit and check the fitted shapes sit on the right surfaces (e.g. on both condyles).",
    "If an axis points the wrong way, use Flip anterior/posterior or Flip superior/inferior (step 4).",
]

_TIF = ("Surfaces made from TIF partial volumes work: loose specks of thresholded noise are ignored, and if "
        "the surface is too rough for the fit it is smoothed first (noted under Apply).")

_PARTIAL_FEMUR = ("On a partial (distal) femur, as from BVR, the hip centre isn't available: Y follows the shaft "
                  "and anterior comes from the condyles. Include as much shaft as you can (5 cm or more above "
                  "the condyles).")

_PARTIAL_TIBIA = ("On a partial (proximal) tibia the ankle isn't available: Y follows the shaft and anterior "
                  "comes from the tuberosity region. Include as much shaft as you can (5 cm or more below the "
                  "plateau).")

_UNSTABLE_TIBIA = ("On a partial tibia this method is sensitive to small differences in the surface: on the "
                   "Autoscoper knee sample, lightly smoothing the surface turned its axes by 14–23° about the long "
                   "axis, while GIBOC-Plateau and Kai stayed within about 1°. Prefer GIBOC-Plateau or Kai for "
                   "partial tibias, or compare against landmarks.")

_AUTO: dict[str, Guide] = {
    "staple_femur_cylinder": Guide("A distal femur with both condyles and some shaft; the whole femur is better.",
                                   _AUTO_STEPS, [_PARTIAL_FEMUR, _TIF]),
    "staple_femur_spheres": Guide("A distal femur with both condyles and some shaft; the whole femur is better.",
                                  _AUTO_STEPS, [_PARTIAL_FEMUR, _TIF]),
    "staple_femur_ellipsoids": Guide("A distal femur with both condyles and some shaft; the whole femur is better.",
                                     _AUTO_STEPS, [_PARTIAL_FEMUR, _TIF]),
    "staple_femur_kai": Guide("The whole femur, including the femoral head. It can't be used on BVR partial "
                              "volumes: choose a GIBOC method for those.", _AUTO_STEPS, [_TIF]),
    "staple_tibia_kai": Guide("A proximal tibia with the plateau and some shaft. The most robust tibia method on "
                              "rough or partial surfaces.", _AUTO_STEPS, [_PARTIAL_TIBIA, _TIF]),
    "staple_tibia_ellipse": Guide("A proximal tibia with clean plateau surfaces and some shaft; the whole tibia is "
                                  "better.", _AUTO_STEPS, [_UNSTABLE_TIBIA, _PARTIAL_TIBIA, _TIF]),
    "staple_tibia_centroids": Guide("A proximal tibia with clean plateau surfaces and some shaft; the whole tibia "
                                    "is better.", _AUTO_STEPS, [_UNSTABLE_TIBIA, _PARTIAL_TIBIA, _TIF]),
    "staple_tibia_plateau": Guide("A proximal tibia with the plateau and some shaft; the whole tibia is better.",
                                  _AUTO_STEPS, [_PARTIAL_TIBIA, _TIF]),
    "staple_talus_ankle": Guide("The whole talus, with a clean trochlea (dome).", _AUTO_STEPS, [_TIF]),
    "staple_talus_subtalar": Guide("The whole talus, with clean posterior-facet and navicular surfaces.",
                                   _AUTO_STEPS, [_TIF]),
    "inertial": Guide("Any closed surface of the bone.",
                      ["Press Apply (step 5).",
                       ("Check the axes in the view: axis 1 is the long axis, then 2 and 3. They have no anatomical "
                        "names, and a symmetric bone can swap axes 2 and 3.")],
                      [("The axes depend on how much of the bone the surface includes, so use the same surface "
                        "(the one you track with) for every trial.")]),
}

_LEGACY = Guide("Kept so models saved with it still open.",
                ["Choose one of the STAPLE methods for new work.", "Press Apply (step 5) to keep using this one."],
                [])

_LANDMARK_STEPS = [
    ("For each landmark listed under Define (step 4), press Place and click the point on the surface, or Paint "
     "and brush over the area (its centre, or a sphere fitted to it, is used)."),
    ("Tools on the left of the view: Place point (P), Paint (B), Erase (E), Navigate (N); Ctrl+Z undoes. While "
     "placing or painting, right-drag rotates and the scroll wheel zooms."),
    "When every landmark shows ✓, press Apply (step 5).",
]

_AAFACT = Guide(
    "The AAFACT program (a separate download) and a surface of the whole bone.",
    ["Check the bone's side (step 2).",
     ("Install the compiled AAFACT with the MATLAB Runtime (github.com/Lenz-Lab/AAFACT), then press Set AAFACT "
      "program… (step 4) and choose it."),
     "Press Run AAFACT on this bone, or Import results (.xlsx)… if you ran AAFACT yourself.",
     "If AAFACT offers several coordinate systems for the bone, choose one, then press Apply (step 5)."],
    ["AAFACT aligns the bone to a template, so it needs the whole bone."],
)


def guide(source: str, segment_type) -> Guide:
    """What the user needs and does for this axes method on this kind of bone."""
    if source == "landmarks":
        names = ", ".join(d.name for d in segment_type.landmarks)
        cautions = [segment_type.axes_note] if segment_type.axes_note else []
        return Guide(f"Landmarks placed by hand: {names}." if names else "Landmarks placed by hand.",
                     _LANDMARK_STEPS, cautions)
    if source == "aafact":
        return _AAFACT
    if source in _AUTO:
        return _AUTO[source]
    from .auto_axes import AUTO_METHODS

    m = AUTO_METHODS.get(source)
    if m is not None and m.legacy:
        return _LEGACY
    return Guide("", _AUTO_STEPS, [])
