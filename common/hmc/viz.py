"""Visualization helpers: passive viewer, GIF recorder, simple plotting.

Viewer   : interactive MuJoCo window (needs a display).
Recorder : offscreen rendering → animated GIF (needs an OpenGL backend; on headless
           Linux set MUJOCO_GL=osmesa or egl). Uses Pillow (installed with matplotlib).
"""
from __future__ import annotations

import time
from pathlib import Path

import mujoco
import numpy as np


class Viewer:
    """Context manager around mujoco.viewer.launch_passive with real-time pacing."""

    def __init__(self, m, d, enabled: bool = True, realtime: float = 1.0, sync_every: int = 8,
                 cam_distance: float = 3.0, cam_elevation: float = -15, cam_azimuth: float = 150,
                 lookat_body: int | None = None):
        self.m, self.d, self.enabled = m, d, enabled
        self.realtime, self.sync_every = realtime, sync_every
        self.cam = (cam_distance, cam_elevation, cam_azimuth)
        self.lookat_body = lookat_body
        self.v = None
        self._k = 0
        self._t0 = None

    def __enter__(self):
        if self.enabled:
            import mujoco.viewer
            self.v = mujoco.viewer.launch_passive(self.m, self.d)
            self.v.cam.distance, self.v.cam.elevation, self.v.cam.azimuth = self.cam
            self._t0 = time.time()
        return self

    def __exit__(self, *exc):
        if self.v is not None:
            self.v.close()

    @property
    def running(self) -> bool:
        return self.v is None or self.v.is_running()

    def sync(self):
        if self.v is None:
            return
        self._k += 1
        if self._k % self.sync_every:
            return
        if self.lookat_body is not None:
            self.v.cam.lookat[:] = self.d.xpos[self.lookat_body]
        self.v.sync()
        if self.realtime > 0:  # pace to wall clock
            lag = self.d.time / self.realtime - (time.time() - self._t0)
            if lag > 0:
                time.sleep(lag)


class Recorder:
    """Collect frames at a fixed fps and save an animated GIF."""

    def __init__(self, m, path: str | None, fps: int = 15, width: int = 400, height: int = 300,
                 distance: float = 2.8, elevation: float = -12, azimuth: float = 150,
                 track_body: int | None = None):
        self.path = path
        self.fps = fps
        self.frames = []
        self.enabled = path is not None
        self._next_t = 0.0
        if self.enabled:
            try:
                self.r = mujoco.Renderer(m, height, width)
            except Exception as e:  # no OpenGL backend
                print(f"[Recorder] không tạo được renderer ({e}); bỏ qua ghi GIF. "
                      "Trên Linux headless thử: MUJOCO_GL=osmesa")
                self.enabled = False
                return
            self.cam = mujoco.MjvCamera()
            self.cam.distance, self.cam.elevation, self.cam.azimuth = distance, elevation, azimuth
            self.track_body = track_body

    def capture(self, d, overlay_text: str | None = None):
        if not self.enabled or d.time + 1e-9 < self._next_t:
            return
        self._next_t += 1.0 / self.fps
        if self.track_body is not None:
            self.cam.lookat[:] = d.xpos[self.track_body]
            self.cam.lookat[2] = max(0.5, self.cam.lookat[2] - 0.2)
        self.r.update_scene(d, self.cam)
        img = self.r.render().copy()
        if overlay_text:
            img = _draw_text(img, overlay_text)
        self.frames.append(img)

    def save(self):
        if not self.enabled or not self.frames:
            return None
        from PIL import Image
        Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        ims = [Image.fromarray(f) for f in self.frames]
        ims[0].save(self.path, save_all=True, append_images=ims[1:],
                    duration=int(1000 / self.fps), loop=0, optimize=True)
        print(f"[Recorder] đã lưu {self.path} ({len(ims)} khung hình)")
        return self.path


def _ascii(text: str) -> str:
    """Pillow's default bitmap font has no Vietnamese glyphs → strip diacritics for overlays."""
    import unicodedata
    text = text.replace("đ", "d").replace("Đ", "D")
    return "".join(c for c in unicodedata.normalize("NFKD", text) if not unicodedata.combining(c))


def _draw_text(img: np.ndarray, text: str) -> np.ndarray:
    from PIL import Image, ImageDraw
    text = _ascii(text)
    im = Image.fromarray(img)
    dr = ImageDraw.Draw(im)
    dr.rectangle([0, 0, im.width, 18 * (text.count("\n") + 1) + 4], fill=(0, 0, 0))
    dr.text((6, 3), text, fill=(255, 255, 255))
    return np.asarray(im)


