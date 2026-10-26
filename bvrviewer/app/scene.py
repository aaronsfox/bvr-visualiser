"""Everything drawn in one tab's 3D viewport: bones, motion trails, radiograph planes."""

from __future__ import annotations

import numpy as np
import pyvista as pv

from ..core.images import adjust_lut
from ..core.models import BoneModel
from ..core.session import TabState, XrayCamera

BACKGROUNDS = {"dark": "#0B0D11", "light": "#D9DCE1", "white": "#FFFFFF"}


class SceneController:
    def __init__(self, plotter, state: TabState):
        self.p = plotter
        self.state = state
        self.frame = 0
        self.selected: BoneModel | None = None
        self.view_mode = "3d"            # "3d" or "xray:<index>"
        self._bone_actors: dict[int, dict] = {}     # id(model) -> {"surface", "volume", "trail": [..]}
        self._cam_actors: dict[int, dict] = {}      # id(camera) -> {"plane", "source", "beams"}
        self._axes_actors: list = []
        self._detector_depths: dict[int, float] = {}
        self.p.set_background(BACKGROUNDS["dark"])
        self.p.add_axes(color="#C3C8D2", line_width=2)

    # ---------------- frames ----------------
    @property
    def n_frames(self) -> int:
        n = [m.tracking.n_frames for m in self.state.models if m.tracking is not None]
        return max(n) if n else 1

    def _hold_still_inverse(self, frame: int) -> np.ndarray:
        name = self.state.settings.hold_still
        if not name:
            return np.eye(4)
        for m in self.state.models:
            if m.name == name and m.tracking is not None:
                ref = m.tracking.pose(frame, "hold")
                return np.linalg.inv(ref) if ref is not None else np.eye(4)
        return np.eye(4)

    def model_pose(self, m: BoneModel, frame: int) -> np.ndarray | None:
        """World pose of a model at a frame (None = hidden). Untracked models stay in model space."""
        if m.tracking is None:
            pose = np.eye(4)
        else:
            pose = m.tracking.pose(frame, self.state.settings.untracked)
            if pose is None:
                return None
        return self._hold_still_inverse(frame) @ pose

    # ---------------- bones ----------------
    def add_model(self, m: BoneModel) -> None:
        self.remove_model(m)
        entry: dict = {"surface": None, "volume": None, "trail": []}
        entry["surface"] = self.p.add_mesh(m.mesh, color=m.color, opacity=m.opacity, smooth_shading=True,
                                           specular=0.25, specular_power=20, pickable=True, reset_camera=False)
        if m.image_data is not None:
            entry["volume"] = self._make_volume_actor(m)
        self._bone_actors[id(m)] = entry
        self.refresh_model_style(m)

    def _make_volume_actor(self, m: BoneModel):
        data = m.image_data["intensity"]
        nz = data[data > 0]
        lo, hi = (np.percentile(nz, [20, 99]) if nz.size else (0, 1))
        actor = self.p.add_volume(m.image_data, scalars="intensity", cmap="bone", clim=(float(lo), float(hi)),
                                  opacity="sigmoid_6", shade=True, show_scalar_bar=False, reset_camera=False)
        return actor

    def remove_model(self, m: BoneModel) -> None:
        entry = self._bone_actors.pop(id(m), None)
        if not entry:
            return
        for a in [entry["surface"], entry["volume"], *entry["trail"]]:
            if a is not None:
                self.p.remove_actor(a, render=False)

    def refresh_model_style(self, m: BoneModel) -> None:
        e = self._bone_actors.get(id(m))
        if not e:
            return
        through = self.view_mode.startswith("xray")
        s = e["surface"]
        s.prop.color = m.color
        op = m.opacity * (self.state.settings.xray_bone_opacity if through else 1.0)
        if m.tracking is None and not through:
            op *= 0.6
        s.prop.opacity = op
        s.prop.style = "wireframe" if (through and self.state.settings.xray_outline) else "surface"
        show_surface = m.display in ("surface", "both") or e["volume"] is None
        s.SetVisibility(m.visible and show_surface)
        if e["volume"] is not None:
            e["volume"].SetVisibility(m.visible and m.display in ("volume", "both"))
        for a in e["trail"]:
            a.prop.color = m.color

    def rebuild_trail(self) -> None:
        for e in self._bone_actors.values():
            for a in e["trail"]:
                self.p.remove_actor(a, render=False)
            e["trail"] = []
        st = self.state.settings
        m = self.selected
        if not st.trail or m is None or m.tracking is None or id(m) not in self._bone_actors:
            return
        e = self._bone_actors[id(m)]
        for k in range(st.trail_count):
            a = self.p.add_mesh(m.mesh, color=m.color, opacity=0.35 - 0.25 * k / max(st.trail_count, 1),
                                smooth_shading=True, pickable=False, reset_camera=False)
            e["trail"].append(a)

    # ---------------- radiographs ----------------
    def _bones_centroid(self) -> np.ndarray:
        pts = []
        for m in self.state.models:
            if m.tracking is None or m.mesh is None:
                continue
            c = np.r_[np.array(m.mesh.center), 1.0]
            valid = np.flatnonzero(m.tracking.valid)
            for f in valid[:: max(1, len(valid) // 20)]:
                pts.append((m.tracking.poses[f] @ c)[:3])
        if not pts:
            for m in self.state.models:
                if m.mesh is not None:
                    pts.append(np.array(m.mesh.center))
        return np.mean(pts, axis=0) if pts else np.zeros(3)

    def add_camera(self, cam: XrayCamera) -> None:
        self.remove_camera(cam)
        cal = cam.calibration
        depth = float(cal.depth(self._bones_centroid())[0]) + self.state.settings.xray_distance
        self._detector_depths[id(cam)] = depth
        corners = cal.plane_corners(depth)
        # Face the plane towards its X-ray source and cull the back, so the image is seen from the
        # source side (where it makes sense) and is see-through from behind instead of hiding the bones.
        normal = np.cross(corners[1] - corners[0], corners[3] - corners[0])
        faces = [4, 0, 1, 2, 3] if np.dot(normal, cal.source - corners[0]) > 0 else [4, 0, 3, 2, 1]
        quad = pv.PolyData(corners, faces=faces)
        quad.active_texture_coordinates = np.array([[0, 1], [1, 1], [1, 0], [0, 0]], float)
        # One grayscale texture per camera, at display size, whose pixels are overwritten each frame
        # (building a new full-resolution texture every frame was what made playback stall).
        tex, scalars = self._new_texture(self._display_shape(cam))
        plane = self.p.add_mesh(quad, texture=tex, lighting=False, pickable=False, reset_camera=False,
                                culling="back")
        src = self.p.add_mesh(pv.Sphere(radius=8, center=cal.source), color="#F2C48D", pickable=False,
                              reset_camera=False)
        beams_mesh = pv.merge([pv.Line(cal.source, c) for c in corners])
        beams = self.p.add_mesh(beams_mesh, color="#5A6170", line_width=1, pickable=False, reset_camera=False)
        self._cam_actors[id(cam)] = {"plane": plane, "source": src, "beams": beams, "image": None,
                                     "tex": tex, "scalars": scalars}
        # Start decoding radiographs in the background, from the first frame onwards.
        cam.images.prefetch(range(cam.first_image, len(cam.images)))
        self.refresh_camera_style()
        self.update_camera_image(cam)

    @staticmethod
    def _display_shape(cam: XrayCamera) -> tuple[int, int]:
        s = cam.images.display_step
        return -(-cam.images.height // s), -(-cam.images.width // s)

    @staticmethod
    def _new_texture(shape: tuple[int, int]):
        tex = pv.Texture(np.zeros(shape, np.uint8))
        tex.interpolate = True
        return tex, tex.to_image().active_scalars

    def remove_camera(self, cam: XrayCamera) -> None:
        """Remove a camera's actors (its images stay loaded; see forget_camera)."""
        e = self._cam_actors.pop(id(cam), None)
        if e:
            for k in ("plane", "source", "beams"):
                self.p.remove_actor(e[k], render=False)

    def forget_camera(self, cam: XrayCamera) -> None:
        """Remove a camera for good, stopping its background image decoding."""
        self.remove_camera(cam)
        if cam.images is not None:
            cam.images.close()

    def reposition_cameras(self) -> None:
        for cam in self.state.cameras:
            if id(cam) in self._cam_actors:
                self.add_camera(cam)
        self.update_frame(self.frame)

    def _lut(self) -> np.ndarray:
        st = self.state.settings
        key = (st.brightness, st.contrast, st.invert)
        if getattr(self, "_lut_key", None) != key:
            self._lut_key, self._lut_table = key, adjust_lut(*key)
        return self._lut_table

    def update_camera_image(self, cam: XrayCamera) -> None:
        e = self._cam_actors.get(id(cam))
        if not e or not e["plane"].GetVisibility():
            return                     # hidden planes aren't updated (or decoded) at all
        i = cam.image_index(self.frame)
        st = self.state.settings
        key = (i, st.brightness, st.contrast, st.invert)
        if e["image"] == key:
            return
        e["image"] = key
        if i is None:
            pixels = np.zeros(self._display_shape(cam), np.uint8)
        else:
            pixels = self._lut()[cam.images.get_display(i)]
            # keep the background decoder ahead of playback
            cam.images.prefetch(range(i + 1, i + 40))
        if pixels.size != e["scalars"].size:          # an image of a different size: new texture
            e["tex"], e["scalars"] = self._new_texture(pixels.shape)
            e["plane"].texture = e["tex"]
        e["scalars"][:] = pixels[::-1].ravel()
        e["tex"].to_image().Modified()

    def refresh_camera_style(self) -> None:
        st = self.state.settings
        through = self.view_mode.startswith("xray")
        active = int(self.view_mode.split(":")[1]) if through else -1
        hold = self._hold_still_inverse(self.frame)
        for idx, cam in enumerate(self.state.cameras):
            e = self._cam_actors.get(id(cam))
            if not e:
                continue
            plane_on = cam.visible and ((through and idx == active) or (not through and st.xray_in_scene))
            e["plane"].SetVisibility(plane_on)
            e["plane"].prop.opacity = 1.0 if through else st.xray_opacity
            e["source"].SetVisibility(cam.visible and st.xray_sources and not through)
            e["beams"].SetVisibility(cam.visible and st.xray_beams and not through)
            for k in ("plane", "source", "beams"):
                e[k].user_matrix = hold

    # ---------------- per-frame update ----------------
    def update_frame(self, frame: int) -> None:
        self.frame = int(frame)
        st = self.state.settings
        for m in self.state.models:
            e = self._bone_actors.get(id(m))
            if not e:
                continue
            pose = self.model_pose(m, self.frame)
            if pose is None:
                e["surface"].SetVisibility(False)
                if e["volume"] is not None:
                    e["volume"].SetVisibility(False)
            else:
                self.refresh_model_style(m)
                e["surface"].user_matrix = pose
                if e["volume"] is not None:
                    e["volume"].user_matrix = pose
            for k, a in enumerate(e["trail"], start=1):
                p = self.model_pose(m, self.frame - k * st.trail_step)
                a.SetVisibility(m.visible and p is not None and self.frame - k * st.trail_step >= 0)
                if p is not None:
                    a.user_matrix = p
        self.refresh_camera_style()
        for cam in self.state.cameras:
            self.update_camera_image(cam)
        if self.view_mode.startswith("xray"):
            self._apply_xray_camera(int(self.view_mode.split(":")[1]))
        self.p.render()

    # ---------------- scene options ----------------
    def apply_settings(self) -> None:
        st = self.state.settings
        self.p.set_background(BACKGROUNDS.get(st.background, BACKGROUNDS["dark"]))
        for a in self._axes_actors:
            self.p.remove_actor(a, render=False)
        self._axes_actors = []
        if st.show_axes:
            b = self.motion_bounds()
            L = 0.25 * (float(np.linalg.norm(np.subtract(b[1::2], b[::2]))) if b else 100.0)
            for d, col in zip(np.eye(3), ("#E5534B", "#57AB5A", "#539BF5")):
                self._axes_actors.append(self.p.add_mesh(pv.Line((0, 0, 0), d * L), color=col, line_width=3,
                                                         pickable=False, reset_camera=False))
        self.p.remove_bounds_axes()
        if st.show_grid:
            b = self.motion_bounds()
            if b is not None:
                self.p.show_grid(bounds=b, color="#8A92A1" if st.background == "dark" else "#555B66",
                                 xtitle="X (mm)", ytitle="Y (mm)", ztitle="Z (mm)", font_size=9)
        for m in self.state.models:
            self.refresh_model_style(m)
        self.rebuild_trail()
        self.update_frame(self.frame)

    def motion_bounds(self) -> list[float] | None:
        """Box around everywhere the visible bones go during the trial (current hold-still frame)."""
        lo, hi = np.full(3, np.inf), np.full(3, -np.inf)
        for m in self.state.models:
            if m.mesh is None or not m.visible:
                continue
            b = np.array(m.mesh.bounds)
            corners = np.array([[x, y, z, 1.0] for x in b[:2] for y in b[2:4] for z in b[4:]])
            frames = range(0, self.n_frames, max(1, self.n_frames // 30)) if m.tracking is not None else [0]
            for f in frames:
                pose = self.model_pose(m, f)
                if pose is None:
                    continue
                pts = (corners @ pose.T)[:, :3]
                lo, hi = np.minimum(lo, pts.min(0)), np.maximum(hi, pts.max(0))
        if not np.all(np.isfinite(lo)):
            return None
        pad = (hi - lo) * 0.05
        lo, hi = lo - pad, hi + pad
        return [lo[0], hi[0], lo[1], hi[1], lo[2], hi[2]]

    def _set_guides_visible(self, on: bool) -> None:
        for a in self._axes_actors:
            a.SetVisibility(on)
        cube = getattr(self.p.renderer, "cube_axes_actor", None)
        if cube is not None:
            cube.SetVisibility(on)

    def _scene_size(self) -> float:
        b = np.array(self.p.bounds).reshape(3, 2)
        size = float(np.linalg.norm(b[:, 1] - b[:, 0]))
        return size if np.isfinite(size) and size > 0 else 100.0

    # ---------------- views ----------------
    def set_view(self, which: str, negative: bool = False) -> None:
        self.view_mode = "3d"
        self._set_guides_visible(True)
        for m in self.state.models:
            self.refresh_model_style(m)
        self.refresh_camera_style()
        bones = [e["surface"] for e in self._bone_actors.values() if e["surface"].GetVisibility()]
        if which == "xy":
            self.p.view_xy(negative=negative, render=False)
        elif which == "yz":
            self.p.view_yz(negative=negative, render=False)
        elif which == "xz":
            self.p.view_xz(negative=negative, render=False)
        else:
            self.p.view_isometric(negative=negative, render=False)
        self.zoom_to_bones(bones)

    def zoom_to_bones(self, actors=None) -> None:
        actors = actors or [e["surface"] for e in self._bone_actors.values()]
        lo, hi = np.full(3, np.inf), np.full(3, -np.inf)
        for a in actors:
            b = np.array(a.GetBounds())
            if not np.all(np.isfinite(b)) or b[1] < b[0]:
                continue
            lo, hi = np.minimum(lo, b[::2]), np.maximum(hi, b[1::2])
        if np.all(np.isfinite(lo)):
            self.p.reset_camera(bounds=[lo[0], hi[0], lo[1], hi[1], lo[2], hi[2]], render=False)
        else:
            self.p.reset_camera(render=False)
        self.p.render()

    def set_xray_view(self, index: int) -> None:
        self.view_mode = f"xray:{index}"
        self._set_guides_visible(False)
        for m in self.state.models:
            self.refresh_model_style(m)
        self.refresh_camera_style()
        self._apply_xray_camera(index)
        self.p.render()

    def _apply_xray_camera(self, index: int) -> None:
        cam = self.state.cameras[index]
        cal = cam.calibration
        depth = self._detector_depths.get(id(cam), float(cal.depth(self._bones_centroid())[0]))
        hold = self._hold_still_inverse(self.frame)
        to_world = lambda p: (hold @ np.r_[p, 1.0])[:3]  # noqa: E731
        c = self.p.camera
        c.position = to_world(cal.source)
        c.focal_point = to_world(cal.pixel_to_world(cal.width / 2, cal.height / 2, depth))
        c.up = hold[:3, :3] @ -cal.R[1]
        c.view_angle = cal.view_angle()
        self.p.renderer.ResetCameraClippingRange()

    def screenshot(self, path: str, scale: int = 2) -> None:
        self.p.screenshot(path, scale=scale)
