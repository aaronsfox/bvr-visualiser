"""Python translation of the STAPLE toolbox's bone coordinate-system algorithms.

STAPLE: https://github.com/modenaxe/msk-STAPLE, Copyright (c) 2020-2021 Luca Modenese and
Jean-Baptiste Renault, CC BY-NC 4.0 (LICENSE_STAPLE.txt). NON-COMMERCIAL USE ONLY.
What was translated, what changed, and the adaptations for partial bones are listed in NOTICE.md.

Please cite Modenese & Renault (2021) J Biomech 116:110186, and Renault et al. (2018) J Biomech
80:171-178 for the GIBOC methods or Kai et al. (2014) J Biomech 47:1229-1233 for the Kai methods.
"""

from .common import BoneResult, bodySide2Sign, tri_from_pyvista
from .femur import FEMUR_FIT_METHODS, GIBOC_femur, Kai2014_femur
from .tibia import TIBIA_FIT_METHODS, GIBOC_tibia, Kai2014_tibia
from .tri import StapleError, Tri

__all__ = ["BoneResult", "StapleError", "Tri", "tri_from_pyvista", "bodySide2Sign", "GIBOC_femur", "Kai2014_femur",
           "GIBOC_tibia", "Kai2014_tibia", "FEMUR_FIT_METHODS", "TIBIA_FIT_METHODS"]
