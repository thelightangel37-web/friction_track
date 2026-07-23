"""

gesture_engine.py

=================

Background gesture-detection engine for a Raspberry Pi kiosk.

- Captures frames from the default webcam using OpenCV.

- Uses MediaPipe Hands to track a single hand (supports tilted / downward palms).

- Applies Exponential Moving Average (EMA) smoothing to eliminate jitter.

- Detects CLICK/PINCH and SWIPE_LEFT/SWIPE_RIGHT gestures.

- Broadcasts real-time JSON (cursor + all 21 landmarks) over WebSocket (localhost:8765).

  The overlay.py process reads this to render the hand skeleton over the kiosk UI.

No GUI / cv2.imshow — fully headless.

MediaPipe compatibility

-----------------------

  < 0.10  →  mp.solutions.hands  (legacy API, model_complexity=1 for accuracy)

  >= 0.10 →  mediapipe.tasks     (Tasks API, auto-downloads model on first run)

Dependencies:

    pip install opencv-python mediapipe websockets

Usage:

    python gesture_engine.py

"""

from __future__ import annotations

import asyncio

import json

import logging

import math

import os

import platform

import subprocess

import threading

import time

import sys

import urllib.request

import queue

from collections import deque

from typing import List, Optional, Any

import cv2

cv2.setNumThreads(2)

import numpy as np

# ===========================================================================

# ── MediaPipe API Detection ───────────────────────────────────────────────

# ===========================================================================

try:

    import mediapipe as mp

    _api_label = "Tasks API (0.10+)"

except ImportError as exc:

    raise SystemExit(

        f"mediapipe is not installed: {exc}\n"

        "Install with:  pip install mediapipe"

    ) from exc

import websockets

# ===========================================================================

# ── Logging ───────────────────────────────────────────────────────────────

# ===========================================================================

logging.basicConfig(

    level=logging.INFO,

    format="%(asctime)s [%(levelname)s] %(message)s",

    datefmt="%H:%M:%S",

)

log = logging.getLogger("gesture_engine")

log.info("MediaPipe: %s", _api_label)

# ===========================================================================

# ── pynput — OS mouse control (optional) ─────────────────────────────────

# ===========================================================================

try:

    from pynput.mouse import Button as _Button, Controller as _MouseController

    _PYNPUT_AVAILABLE = True

    log.info("pynput available — OS mouse control enabled.")

except ImportError:

    _PYNPUT_AVAILABLE = False

    _Button          = None   # type: ignore[assignment]

    _MouseController = None   # type: ignore[assignment]

    log.warning(

        "pynput not installed — OS cursor/click injection disabled.\n"

        "  Install with:  pip install pynput"

    )

# ===========================================================================

# ── CONFIGURATION ─────────────────────────────────────────────────────────

# ===========================================================================

# ── Display Orientation / Auto-Detection ──────────────────────────────────────

# Fallback orientation if dynamic OS detection fails.

DISPLAY_ORIENTATION: str = "portrait"

_BASE_SHORT: int = 1080

_BASE_LONG:  int = 1920

# ── Camera Hardware ─────────────────────────────────────────────────────────

CAMERA_INDEX:  int = 0

CAMERA_WIDTH:  int = 640

CAMERA_HEIGHT: int = 480

TARGET_FPS:    int = 60

# ── Camera Ergonomic Margins (Phase 2) ──────────────────────────────────────

# Defines the maximum percentage of the camera's field of view to use for 

# cursor movement (0.0 to 1.0). 

# - A smaller number means a FASTER cursor (less hand movement required).

# - A larger number means a SLOWER, more precise cursor.

# - Do NOT use 1.0, or you will have to reach the extreme edges of the camera 

#   where hand tracking is unreliable. A value of 0.80 leaves a 10% deadzone 

#   on each side.

CAMERA_MARGINS = {

    "landscape": {

        "x": 0.80,  # 80% of width (leaves a safe cushion, prevents clipping)

        "y": 0.60   # 60% of height (comfortable vertical reach)

    },

    "portrait": {

        "x": 0.50,  # 50% of width (slower, highly precise horizontal movement)

        "y": 0.80   # 80% of height (safe cushion for reaching top/bottom)

    }

}

def _detect_linux_geometry() -> tuple:

    """

    Parse ``xrandr --verbose`` to get the connected display's active resolution

    AND rotation.  Returns (width, height, rotation_degrees) on success,

    or (0, 0, 0) on failure.

    xrandr --verbose reports the active mode dimensions ALREADY swapped when

    the display is rotated (e.g. a 1920x1080 panel in "left" rotation appears

    as 1080x1920).  The rotation keyword is on the same "connected" line:

        HDMI-1 connected primary 720x1280+0+0 (0x45) left (...)

    Rotation keyword  → angle

        normal        →   0

        left          →  90   (CCW from panel default → content rotated 90 CW)

        inverted      → 180

        right         → -90   (CW from panel default → content rotated 90 CCW)

    Works on DietPi / Raspberry Pi OS with X11 or XWayland.

    """

    if platform.system() == "Windows":

        return 0, 0, 0

    try:

        env = dict(os.environ)

        env.setdefault("DISPLAY", ":0")

        r = subprocess.run(

            ["xrandr", "--verbose"],

            capture_output=True, text=True, timeout=3, env=env,

        )

        _ROTATION_MAP = {

            "normal":   0,

            "left":    90,

            "inverted": 180,

            "right":   -90,

        }

        for line in r.stdout.splitlines():

            if " connected" not in line:

                continue

            # Parse geometry token: "720x1280+0+0" or "1280x720+0+0"

            w, h = 0, 0

            for token in line.split():

                if "x" in token and "+" in token:

                    res = token.split("+")[0]

                    parts = res.split("x")

                    if len(parts) == 2 and parts[0].isdigit() and parts[1].isdigit():

                        w, h = int(parts[0]), int(parts[1])

                        break

            # Parse rotation keyword on the same line

            rotation = 0

            for kw, angle in _ROTATION_MAP.items():

                # Match whole word only — "normal" must not match "(normal"

                if f" {kw} " in f" {line} ":

                    rotation = angle

                    break

            if w > 0 and h > 0:

                return w, h, rotation

    except Exception:

        pass

    return 0, 0, 0

class _SimpleEMA:

    def __init__(self, alpha: float):

        self.alpha = alpha

        self.val = None

    def update(self, new_val: float) -> float:

        if self.val is None:

            self.val = new_val

        else:

            self.val = self.alpha * new_val + (1.0 - self.alpha) * self.val

        return self.val

class OneEuroFilter:

    def __init__(self, min_cutoff=0.8, beta=1.5, d_cutoff=1.0):

        self.min_cutoff = min_cutoff

        self.beta = beta

        self.d_cutoff = d_cutoff

        self.x_prev = None

        self.dx_prev = None

        self.t_prev = None

    def __call__(self, t: float, x: np.ndarray) -> np.ndarray:

        if self.t_prev is None:

            self.x_prev = x.copy()

            self.dx_prev = np.zeros_like(x)

            self.t_prev = t

            return self.x_prev

        te = t - self.t_prev

        if te <= 0.0:

            return self.x_prev

        # The filtered derivative of the signal.

        ad = self._alpha(te, self.d_cutoff)

        dx = (x - self.x_prev) / te

        dx_hat = ad * dx + (1.0 - ad) * self.dx_prev

        # The filtered signal.

        cutoff = self.min_cutoff + self.beta * np.linalg.norm(dx_hat)

        a = self._alpha(te, cutoff)

        x_hat = a * x + (1.0 - a) * self.x_prev

        self.x_prev = x_hat

        self.dx_prev = dx_hat

        self.t_prev = t

        return x_hat

    def _alpha(self, te, cutoff):

        tau = 1.0 / (2.0 * math.pi * cutoff)

        return te / (te + tau)

class DisplayManager:

    """

    Centralized geometry manager.

    Geometry source of truth (in priority order):

      1. overlay.py pushes Qt-verified geometry via set_external_geometry()

         (authoritative on all platforms).  Rotation is re-detected from xrandr

         on every geometry push so live display-rotation changes are handled.

      2. On Linux, _detect_linux_geometry() queries xrandr at startup for both

         resolution AND rotation so the engine starts correctly.

      3. On Windows, update() polls via GetSystemMetrics for local dev/testing.

         Rotation is always 0 on Windows (dev machines are not rotated).

      4. DISPLAY_ORIENTATION / _BASE_* constants are the last-resort fallback.

    """

    def __init__(self):

        self._lock = threading.Lock()

        self.rotation: int = 0   # degrees: 0, 90, -90, or 180 — auto-detected

        # ── Step 1: static fallback (safe non-zero placeholder) ────────────

        if DISPLAY_ORIENTATION == "portrait":

            self.target_width, self.target_height = _BASE_SHORT, _BASE_LONG

        else:

            self.target_width, self.target_height = _BASE_LONG, _BASE_SHORT

        # ── Step 2: Linux auto-detection via xrandr ───────────────────────────

        # xrandr --verbose returns ALREADY-SWAPPED dimensions when the display

        # is rotated (e.g. "720x1280" for a 1280x720 panel rotated "left").

        # We take those swapped dimensions directly as target_width/height, and

        # store the rotation angle so GestureProcessor can rotate camera coords.

        if platform.system() != "Windows":

            xr_w, xr_h, xr_rot = _detect_linux_geometry()

            if xr_w > 0 and xr_h > 0:

                self.target_width  = xr_w

                self.target_height = xr_h

                self.rotation      = xr_rot

                log.info(

                    "Display detected via xrandr: %dx%d  rotation=%d°",

                    xr_w, xr_h, xr_rot,

                )

            else:

                log.warning(

                    "xrandr detection failed — using static fallback %dx%d. "

                    "overlay.py will push the correct geometry on connect.",

                    self.target_width, self.target_height,

                )

        self.active_x_min: float = 0.0

        self.active_x_max: float = 1.0

        self.active_y_min: float = 0.0

        self.active_y_max: float = 1.0

        self.scale_x: float = 0.0

        self.scale_y: float = 0.0

        self._base_ax_range: float = 1.0

        self._base_ay_range: float = 1.0

        

        self._recompute_margins()

    def _recompute_margins(self) -> None:

        """Shared math: derive symmetric active-region bounds from the

        current target_width/target_height. Called any time geometry changes,

        from whichever source (Windows polling or overlay.py's push)."""

        if self.target_width == 0 or self.target_height == 0:

            return

        orientation = "landscape" if self.target_width > self.target_height else "portrait"

        margins = CAMERA_MARGINS.get(orientation, CAMERA_MARGINS["landscape"])

        _ax_range = margins["x"]

        

        # ── DYNAMIC ASPECT RATIO ADAPTATION ──

        # Computes Y margin to perfectly match the display's aspect ratio.

        # This guarantees 1:1 physical-to-screen movement (isotropic scaling)

        # without stretching, on any display or orientation.

        _ay_range = (_ax_range * CAMERA_WIDTH * self.target_height) / (CAMERA_HEIGHT * self.target_width)

        

        # If the requested Y margin exceeds the camera's FOV (e.g. very tall screen),

        # clamp it and shrink the X margin instead to maintain perfect proportions.

        if _ay_range > 0.95:

            _ay_range = 0.95

            _ax_range = (_ay_range * CAMERA_HEIGHT * self.target_width) / (CAMERA_WIDTH * self.target_height)

        # Bounds are always derived symmetrically from the margin — the

        # margin is a WIDTH around center, never a raw max. Keeping this in

        # one helper prevents the classic bug where a margin value gets used

        # directly as a one-sided max, silently pushing the whole deadzone

        # onto one edge.

        with self._lock:

            self._base_ax_range = _ax_range

            self._base_ay_range = _ay_range

            self.active_x_min = round((1.0 - _ax_range) / 2, 4)

            self.active_x_max = round(1.0 - self.active_x_min, 4)

            self.active_y_min = round((1.0 - _ay_range) / 2, 4)

            self.active_y_max = round(1.0 - self.active_y_min, 4)

            self.scale_x = round(self.target_width  / _ax_range, 2) if _ax_range else 0.0

            self.scale_y = round(self.target_height / _ay_range, 2) if _ay_range else 0.0

        log.info(f"Geometry Update: {orientation.upper()} {self.target_width}x{self.target_height}")

        log.info(f" -> Camera Size: {CAMERA_WIDTH}x{CAMERA_HEIGHT}")

        log.info(f" -> Active Margins: X={_ax_range:.2f}, Y={_ay_range:.2f}")

        log.info(f" -> Active Min/Max: X[{self.active_x_min:.3f}, {self.active_x_max:.3f}], Y[{self.active_y_min:.3f}, {self.active_y_max:.3f}]")

        log.info(f" -> Engine Scales: scale_x={self.scale_x}, scale_y={self.scale_y}")

    def get_snapshot(self) -> dict:

        with self._lock:

            return {

                'active_x_min': self.active_x_min,

                'active_x_max': self.active_x_max,

                'active_y_min': self.active_y_min,

                'active_y_max': self.active_y_max,

                'target_width': self.target_width,

                'target_height': self.target_height,

                'scale_x': self.scale_x,

                'scale_y': self.scale_y

            }

    def set_external_geometry(self, w: int, h: int) -> bool:

        """

        Authoritative geometry push from overlay.py (Qt-verified — correct on

        Linux/X11/Wayland). This is the PRIMARY path on the Raspberry Pi kiosk.

        Also re-queries xrandr for the current rotation so that a live display

        rotation (e.g. user rotates the screen while the kiosk is running) is

        picked up immediately without restarting the engine.

        Returns True if geometry or rotation actually changed.

        """

        if w <= 0 or h <= 0:

            return False

        # Re-detect rotation from the live xrandr state every time the overlay

        # pushes geometry (i.e. on connect and on every screen-geometry change).

        # This is cheap (one subprocess call) and is the single place where

        # rotation can be updated at runtime without a restart.

        new_rotation = self.rotation

        if platform.system() != "Windows":

            _, _, xr_rot = _detect_linux_geometry()

            new_rotation = xr_rot

        # xrandr already reports swapped dimensions when the display is rotated,

        # so w/h from the overlay are the "visual" dims — use them directly.

        changed = False

        with self._lock:

            if w != self.target_width or h != self.target_height or new_rotation != self.rotation:

                changed = True

                self.target_width  = w

                self.target_height = h

                self.rotation      = new_rotation

        if changed:

            if new_rotation != self.rotation:

                log.info("Display rotation changed: %d°", new_rotation)

            self._recompute_margins()

        return changed

    def update(self) -> bool:

        """

        Windows-only convenience path (e.g. local dev/testing off-Pi).

        Returns True if geometry changed. Guaranteed no-op on any non-Windows

        platform — it will NOT overwrite geometry pushed by overlay.py with

        a hardcoded fallback; it simply does nothing.

        """

        if platform.system() != "Windows":

            return False

        with self._lock:

            prev_w, prev_h = self.target_width, self.target_height

        w, h = 0, 0

        try:

            import ctypes

            user32 = ctypes.windll.user32

            # SM_CXSCREEN / SM_CYSCREEN — the PRIMARY monitor only.

            # (Deliberately not SM_CXVIRTUALSCREEN/SM_CYVIRTUALSCREEN: on a

            # multi-monitor dev box those report the bounding box of ALL

            # monitors combined, which inflates target_width and reproduces

            # this exact "can't reach the true right edge" bug on Windows too.)

            w = user32.GetSystemMetrics(0)

            h = user32.GetSystemMetrics(1)

        except Exception:

            pass

        if w == 0 or h == 0:

            # Detection failed this cycle — keep the last known-good geometry

            # rather than resetting to a static, possibly-wrong fallback.

            return False

        with self._lock:

            if w == prev_w and h == prev_h:

                return False

            self.target_width = w

            self.target_height = h

            

        self._recompute_margins()

        return True

display_manager = DisplayManager()

# ── Tracking Filter ───────────────────────────────────────────────────────

# Vectorized Kalman filter parameters.

# KF_Q = Process noise (acceleration variance). Higher = snappier, less lag.

# KF_R = Measurement noise. Lower = trusts raw input more, less smoothing.

KF_Q: float = 100.0

KF_R: float = 0.001

KF_Q_Z: float = 2.0      # Z is noisier, trust the model more (smoother)

KF_R_Z: float = 0.05     # Z is noisier, trust the measurement less

# ── Overlay landmark smoothing ────────────────────────────────────────────

# Gentle EMA applied to each of the 21 landmarks before sending to the

# overlay renderer — prevents dot flickering without adding noticeable lag.

# Lower alpha = more stable dots, Higher alpha = more responsive/accurate.

LM_EMA_ALPHA: float = 0.22     # 0.22 = balanced: stable skeleton structure, still responsive

# ── Dwell-to-Click Detection ──────────────────────────────────────────────

# A click fires when the hand cursor stays within DWELL_RADIUS_PX pixels

# of its starting position for DWELL_DURATION_S seconds.

DWELL_DURATION_S: float  = 1.2    # seconds of stillness required

DWELL_RADIUS_PX: int     = 48     # max pixel drift before resetting the timer (relaxed for natural hover)

DWELL_COOLDOWN_S: float  = 1.5    # seconds to wait after a click before accepting another

# ── Swipe Detection ────────────────────────────────────────────────────────

SWIPE_HISTORY_DURATION_S: float = 0.15   # seconds to keep history

SWIPE_VELOCITY_THRESHOLD: float = 1.0    # normalized units per second

SWIPE_COOLDOWN_S: float         = 0.35   # seconds

MISS_TOLERANCE_S: float         = 0.35   # seconds — bridges 7 frames at 20 FPS

# ── Hand Detection Sensitivity ────────────────────────────────────────────

# IMPORTANT: tracking confidence should be LOWER than (or equal to) detection

# confidence, not higher. MediaPipe only re-runs the (expensive) full-frame

# palm detector when tracking confidence drops below this threshold — set it

# too high and the tracker keeps "losing the lock" on a perfectly good hand,

# forcing a full re-detect every few frames. That's both a CPU/heat spike

# (full-frame search is far costlier than incremental tracking) and the

# direct cause of the hand "jumping" — re-detection finds the hand in a

# slightly different spot than the tracker would have, producing a visible

# snap. Keeping tracking confidence modest keeps the lock stable.

DETECT_CONFIDENCE: float   = 0.60

TRACK_CONFIDENCE: float    = 0.40   # comfortably below detect, per your own note above

PRESENCE_CONFIDENCE: float = 0.40   # Tasks API equivalent of TRACK_CONFIDENCE

# ── WebSocket ──────────────────────────────────────────────────────────────

WS_HOST: str = "localhost"

WS_PORT: int = 8765

# ── Pseudo-3D Physical Mapping ─────────────────────────────────────────────

# Distance-dependent size/velocity correction

REFERENCE_SIZE: float    = 0.15   # baseline palm metric at comfortable distance

REFERENCE_3D_SPAN: float = 0.11   # baseline 3D wrist-to-middle-MCP span (~11cm)

Z_MIN: float             = 0.4    # nearest zoom bound

Z_MAX: float             = 2.5    # furthest zoom bound

SIZE_EMA_ALPHA: float    = 0.05   # slow time constant for depth metric

# ── Model complexity (legacy solutions API only) ───────────────────────────

#   0 = "lite"  — ~3x cheaper than complexity 1, small accuracy trade-off.

#                 Recommended on any Raspberry Pi.

#   1 = "full"  — heavier model, only worth it on desktop-class CPUs.

MODEL_COMPLEXITY: int = 0   # actually lite — matches the Pi recommendation above

# ── System Mouse Control ──────────────────────────────────────────────────

# Set False to keep gesture_engine as a pure WebSocket broadcaster without

# touching the OS cursor (useful when a browser / Electron frontend handles

# its own UI reactions from the WebSocket stream).

ENABLE_SYSTEM_MOUSE: bool = True

# ── Tasks API model ────────────────────────────────────────────────────────

_MODEL_URL  = (

    "https://storage.googleapis.com/mediapipe-models/"

    "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"

)

_MODEL_PATH = os.path.join(

    os.path.dirname(os.path.abspath(__file__)), "hand_landmarker.task"

)

# ===========================================================================

# ── Model download (Tasks API) ────────────────────────────────────────────

# ===========================================================================

def _ensure_model() -> None:

    if os.path.exists(_MODEL_PATH):

        return

    log.info("Downloading hand_landmarker.task (~6 MB) …")

    try:

        urllib.request.urlretrieve(_MODEL_URL, _MODEL_PATH)

        log.info("Model saved to: %s", _MODEL_PATH)

    except Exception as exc:

        raise SystemExit(

            f"Model download failed: {exc}\n"

            f"Download manually from:\n  {_MODEL_URL}\n"

            f"and place it next to gesture_engine.py as 'hand_landmarker.task'."

        ) from exc

# ===========================================================================

# ── Shared State ──────────────────────────────────────────────────────────

# ===========================================================================

class GestureState:

    """Thread-safe gesture state container."""

    def __init__(self) -> None:

        self._lock = threading.RLock()

        self._hand_detected = False

        self._gesture = "NONE"

        self._cursor_state = "MOVE"

        self._landmarks = []

        self._dwell_progress = 0.0

        self._cursor_x = 0

        self._cursor_y = 0

        self._skel_anchor_x = 0.0

        self._skel_anchor_y = 0.0

        self._hand_size = 1.0

        self._depth_z = 1.0

        self._hand_pose_mode = "PALM_FACING_CAMERA"

        self._palm_normal = (0.0, 0.0, 1.0)

    @property

    def hand_detected(self) -> bool:

        with self._lock: return self._hand_detected

    

    @hand_detected.setter

    def hand_detected(self, value: bool) -> None:

        with self._lock: self._hand_detected = value

        

    @property

    def gesture(self) -> str:

        with self._lock: return self._gesture

    

    @gesture.setter

    def gesture(self, value: str) -> None:

        with self._lock: self._gesture = value

    @property

    def state(self) -> str:

        with self._lock: return self._cursor_state

    @state.setter

    def state(self, value: str) -> None:

        with self._lock: self._cursor_state = value

    @property

    def x(self) -> int:

        with self._lock: return self._cursor_x

    @x.setter

    def x(self, value: int) -> None:

        with self._lock: self._cursor_x = value

    @property

    def y(self) -> int:

        with self._lock: return self._cursor_y

    @y.setter

    def y(self, value: int) -> None:

        with self._lock: self._cursor_y = value

    @property

    def hand_pose_mode(self) -> str:

        with self._lock: return self._hand_pose_mode

    @hand_pose_mode.setter

    def hand_pose_mode(self, value: str) -> None:

        with self._lock: self._hand_pose_mode = value

    @property

    def palm_normal(self) -> tuple[float, float, float]:

        with self._lock: return self._palm_normal

    @palm_normal.setter

    def palm_normal(self, value: tuple[float, float, float]) -> None:

        with self._lock: self._palm_normal = value

    @property

    def skel_anchor_x(self) -> float:

        with self._lock: return self._skel_anchor_x

    @skel_anchor_x.setter

    def skel_anchor_x(self, value: float) -> None:

        with self._lock: self._skel_anchor_x = value

    @property

    def skel_anchor_y(self) -> float:

        with self._lock: return self._skel_anchor_y

    @skel_anchor_y.setter

    def skel_anchor_y(self, value: float) -> None:

        with self._lock: self._skel_anchor_y = value

    @property

    def dwell_progress(self) -> float:

        with self._lock: return self._dwell_progress

    @dwell_progress.setter

    def dwell_progress(self, value: float) -> None:

        with self._lock: self._dwell_progress = value

        

    @property

    def landmarks(self) -> List[List[float]]:

        with self._lock: return list(self._landmarks)

        

    @landmarks.setter

    def landmarks(self, value: List[List[float]]) -> None:

        with self._lock: self._landmarks = value

    def snapshot(self) -> dict:

        """Get all state in single atomic operation."""

        with self._lock:

            return {

                'hand_detected': self._hand_detected,

                'gesture': self._gesture,

                'state': self._cursor_state,

                'landmarks': list(self._landmarks),

                'dwell_progress': self._dwell_progress,

                'x': self._cursor_x,

                'y': self._cursor_y,

                'skel_anchor_x': self._skel_anchor_x,

                'skel_anchor_y': self._skel_anchor_y,

                'hand_size': self._hand_size,

                'depth_z': self._depth_z,

                'hand_pose_mode': self._hand_pose_mode,

                'palm_normal': self._palm_normal,

            }

    def update(

        self,

        x: int,

        y: int,

        state: str,

        gesture: str,

        hand_detected: bool = False,

        dwell_progress: float = 0.0,

        landmarks: Optional[List[List[float]]] = None,

        hand_size: float = 1.0,

        depth_z: float = 1.0,

    ) -> None:

        with self._lock:

            self._cursor_x = x

            self._cursor_y = y

            self._cursor_state = state

            self._gesture = gesture

            self._hand_detected = hand_detected

            self._dwell_progress = dwell_progress

            self._landmarks = landmarks if landmarks is not None else []

            self._hand_size = hand_size

            self._depth_z = depth_z

    def to_json(self) -> str:

        with self._lock:

            d = self.snapshot()

            d['dwell_progress'] = round(d['dwell_progress'], 3)

            # Broadcast the complete camera→screen mapping contract so that

            # overlay.py (and any future renderer) never needs to recompute

            # geometry itself.  The overlay consumes scale_x / scale_y directly;

            # the active_* and screen_* fields are included for completeness

            # (calibration UIs, debug viewers, alternate renderers, etc.).

            dm_snap = display_manager.get_snapshot()

            d['active_x_min']  = dm_snap['active_x_min']

            d['active_x_max']  = dm_snap['active_x_max']

            d['active_y_min']  = dm_snap['active_y_min']

            d['active_y_max']  = dm_snap['active_y_max']

            d['screen_width']  = dm_snap['target_width']

            d['screen_height'] = dm_snap['target_height']

            d['scale_x']       = dm_snap['scale_x']

            d['scale_y']       = dm_snap['scale_y']

            d['depth_z']       = self._depth_z

            d['camera_width']  = CAMERA_WIDTH

            d['camera_height'] = CAMERA_HEIGHT

            d['cam_aspect']    = round(CAMERA_WIDTH / CAMERA_HEIGHT, 4) if CAMERA_HEIGHT else 1.333

            # Legacy keys — kept so overlay.py has a safe fallback before the

            # first scale_x/scale_y arrives (uses window size as approximation).

            d['display_w'] = display_manager.target_width

            d['display_h'] = display_manager.target_height

            return json.dumps(d, separators=(",", ":"))

# ===========================================================================

# ── Gesture Processor Helpers ─────────────────────────────────────────────

# ===========================================================================

def compute_palm_normal_3d(world_lm_np: Optional[np.ndarray], handedness_str: str = "Right") -> tuple[tuple[float, float, float], str, bool]:

    """

    Computes 3D unit palm normal vector (Nx, Ny, Nz) using metric world landmarks.

    Inverts cross product for Left hands so palm-down is consistently -Ny.

    Classifies posture mode:

      - FLAT_PALM_DOWN: Ny < -0.50 (palm facing floor / back of hand facing camera)

      - FLAT_PALM_UP:   Ny >  0.50 (palm facing ceiling)

      - SIDEWAYS_EDGE:  |Nx| > 0.70 (hand turned sideways)

      - PALM_FACING_CAMERA: default front-facing pose

    Returns ((Nx, Ny, Nz), pose_mode_str, is_flat_down_mode_bool)

    """

    if world_lm_np is None or len(world_lm_np) < 18:

        return (0.0, 0.0, 1.0), "PALM_FACING_CAMERA", False

    v_index = world_lm_np[5, :3] - world_lm_np[0, :3]

    v_pinky = world_lm_np[17, :3] - world_lm_np[0, :3]

    if "Left" in handedness_str:

        raw_normal = np.cross(v_pinky, v_index)

    else:

        raw_normal = np.cross(v_index, v_pinky)

    norm = np.linalg.norm(raw_normal)

    if norm < 1e-6:

        return (0.0, 0.0, 1.0), "PALM_FACING_CAMERA", False

    n = raw_normal / norm

    nx, ny, nz = float(n[0]), float(n[1]), float(n[2])

    is_flat_down = ny < -0.50 or abs(ny) > 0.65

    if ny < -0.50:

        mode = "FLAT_PALM_DOWN"

    elif ny > 0.50:

        mode = "FLAT_PALM_UP"

    elif abs(nx) > 0.70:

        mode = "SIDEWAYS_EDGE"

    else:

        mode = "PALM_FACING_CAMERA"

    return (nx, ny, nz), mode, is_flat_down

def check_finger_curl(lm_np: np.ndarray, world_lm_np: Optional[np.ndarray] = None, is_flat_mode: bool = False) -> tuple[float, float]:

    """

    Calculates pointing pose confidence score p in [0.0, 1.0] and index_score.

    Evaluates 2D joint angles, with automatic 3D World Landmark fallback when

    2D segments are compressed (< 0.035) or when in FLAT_PALM_DOWN mode.

    Immune to camera Z-axis foreshortening.

    Returns (pointing_score, index_score)

    """

    def cosine_angle(mcp_idx: int, pip_idx: int, tip_idx: int) -> float:

        v1_2d = lm_np[pip_idx, :2] - lm_np[mcp_idx, :2]

        v2_2d = lm_np[tip_idx, :2] - lm_np[pip_idx, :2]

        n1_2d, n2_2d = np.linalg.norm(v1_2d), np.linalg.norm(v2_2d)

        # Edge-on compression fallback to 3D World Space

        if (n1_2d < 0.035 or n2_2d < 0.035 or is_flat_mode) and world_lm_np is not None:

            v1_3d = world_lm_np[pip_idx, :3] - world_lm_np[mcp_idx, :3]

            v2_3d = world_lm_np[tip_idx, :3] - world_lm_np[pip_idx, :3]

            n1_3d, n2_3d = np.linalg.norm(v1_3d), np.linalg.norm(v2_3d)

            if n1_3d < 1e-6 or n2_3d < 1e-6:

                return 1.0

            return float(np.clip(np.dot(v1_3d, v2_3d) / (n1_3d * n2_3d), -1.0, 1.0))

        if n1_2d < 1e-6 or n2_2d < 1e-6:

            return 1.0

        cos_val = np.dot(v1_2d, v2_2d) / (n1_2d * n2_2d)

        return float(np.clip(cos_val, -1.0, 1.0))

    def clamp(val: float, lo: float = 0.0, hi: float = 1.0) -> float:

        return max(lo, min(hi, val))

    index_cos  = cosine_angle(5,  6,  8)

    middle_cos = cosine_angle(9,  10, 12)

    ring_cos   = cosine_angle(13, 14, 16)

    pinky_cos  = cosine_angle(17, 18, 20)

    # Index finger score: 0.65 -> 0.85 maps to 0.0 -> 1.0 (natural pointing angle)
    index_score = clamp((index_cos - 0.65) / (0.85 - 0.65))

    # Other fingers score: max cos 0.80 -> 0.65 maps to 0.0 -> 1.0 (relaxed curled fingers)
    max_other_cos = max(middle_cos, ring_cos, pinky_cos)
    other_curl_score = clamp((0.80 - max_other_cos) / (0.80 - 0.65))

    # Pointing active is primarily driven by index extension so pointing activates with open, relaxed, or closed non-index fingers
    pointing_score = index_score

    return pointing_score, index_score

class GestureFSM:

    def __init__(self):

        self.state = "MOVE"

        self.dwell_start_ts = 0.0

        self.cooldown_until = 0.0

        self.dwell_origin = None

    def tick(self, now: float, is_pointing: bool, screen_x: float, screen_y: float, dwell_radius: float) -> tuple[str, float]:

        """

        Returns (cursor_state, dwell_progress)

        """

        if now < self.cooldown_until:

            self.state = "MOVE"

            self.dwell_origin = None

            return "MOVE", 0.0

        if not is_pointing:

            self.state = "MOVE"

            self.dwell_origin = None

            return "MOVE", 0.0

        # Pointing is active

        if self.dwell_origin is None:

            self.dwell_origin = (screen_x, screen_y)

            self.dwell_start_ts = now

            self.state = "DWELL"

            return "DWELL", 0.0

        drift = math.hypot(screen_x - self.dwell_origin[0], screen_y - self.dwell_origin[1])

        if drift > dwell_radius:

            self.dwell_origin = (screen_x, screen_y)

            self.dwell_start_ts = now

            self.state = "DWELL"

            return "DWELL", 0.0

        elapsed = now - self.dwell_start_ts

        progress = min(1.0, elapsed / DWELL_DURATION_S)

        if progress >= 1.0:

            self.state = "CLICK"

            self.cooldown_until = now + 0.8

            self.dwell_origin = None

            return "CLICK", 1.0

        self.state = "DWELL"

        return "DWELL", progress

class GestureProcessor:

    """

    Stateful per-frame gesture interpreter.

    Accepts landmarks from the Tasks API.

    Click mechanism: dwell-to-click.

    A CLICK fires when the cursor stays within DWELL_RADIUS_PX pixels of its

    starting position for DWELL_DURATION_S seconds. After firing, a cooldown

    prevents accidental repeat clicks.

    """

    def __init__(self, shared_state: GestureState) -> None:

        self._state = shared_state

        self._last_ts: float = 0.0   # timestamp of most-recent detected frame

        self._fsm = GestureFSM()

        self._cursor_filter = OneEuroFilter(min_cutoff=2.5, beta=3.5, d_cutoff=1.0)

        # Velocity extrapolation & Adaptive EMA landmark smoothing state

        self._last_raw_lm:  Optional[np.ndarray] = None

        self._lm_velocity:  Optional[np.ndarray] = None

        self._lm_ema:       Optional[np.ndarray] = None

        self._pointing_active: bool              = False

        # Swipe detection

        self._x_history: deque[tuple[float, float]] = deque()

        self._swipe_cooldown_until: float = 0.0

        # OS mouse controller (pynput) — None when pynput is unavailable or disabled

        self._mouse = (

            _MouseController()

            if (_PYNPUT_AVAILABLE and ENABLE_SYSTEM_MOUSE)

            else None

        )

        self._prev_cursor_state: str = "MOVE"

        # Micro-tremor deadzone state (Fix 6)

        self._last_screen_x: Optional[int] = None

        self._last_screen_y: Optional[int] = None

        

        # evdev for linux native clicks

        self._ui = None

        if sys.platform != "win32":

            try:

                import evdev

                from evdev import UInput, ecodes as e

                self._ui = UInput({e.EV_KEY: [e.BTN_LEFT]}, name="GestureEngine_Mouse")

                log.info("Initialized evdev UInput for native clicks.")

            except Exception as ex:

                log.warning(f"Could not init evdev UInput: {ex}. Falling back to xdotool if needed.")

        self._is_pointing: bool = False

    def process(self, lms: Any, world_lms: Optional[Any] = None) -> None:

        """

        lms — iterable of 21 landmark objects with .x / .y (normalised 0-1).

        world_lms — iterable of 21 landmark objects with .x / .y / .z (meters, wrist-origin).

        """

        now = time.perf_counter()

        # ── 1. Extract and Rotate Landmarks ─────────────────────────────────

        raw_lm_np = np.array([[lm.x, lm.y, getattr(lm, 'z', 0.0)] for lm in lms], dtype=np.float32)

        world_lm_np = None

        if world_lms:

            world_lm_np = np.array([[lm.x, lm.y, getattr(lm, 'z', 0.0)] for lm in world_lms], dtype=np.float32)

        _rotation = display_manager.rotation

        if _rotation == 90:

            raw_lm_np[:, :2] = 1.0 - raw_lm_np[:, [1, 0]]

        elif _rotation == -90:

            raw_lm_np[:, :2] = raw_lm_np[:, [1, 0]]

        elif _rotation == 180:

            raw_lm_np[:, 1] = 1.0 - raw_lm_np[:, 1]

            

        self._run_pipeline(now, raw_lm_np, world_lm_np)

        

    def miss(self) -> None:

        now = time.perf_counter()

        # Use velocity extrapolation for up to MISS_TOLERANCE_S before resetting

        if self._last_raw_lm is not None and (now - self._last_ts) < MISS_TOLERANCE_S:

            self._run_pipeline(now, None, None)

        else:

            self.reset()

            

    def _run_pipeline(self, now: float, raw_lm_np: Optional[np.ndarray], world_lm_np: Optional[np.ndarray]) -> None:

        # ── Landmark resolution & Gross Palm Velocity-Scaled Adaptive EMA ──

        if raw_lm_np is not None:

            # 3D Palm Normal & Posture Mode Classification

            palm_normal, hand_pose_mode, is_flat_down = compute_palm_normal_3d(world_lm_np)

            self._hand_pose_mode = hand_pose_mode

            self._palm_normal = palm_normal

            self._state.hand_pose_mode = hand_pose_mode

            self._state.palm_normal = palm_normal

            if self._last_ts > 0 and self._last_raw_lm is not None:

                dt = now - self._last_ts

                if dt > 0:

                    self._lm_velocity = (raw_lm_np - self._last_raw_lm) / dt

                    # Gross Palm Center velocity (Wrist 0, Index Base 5, Pinky Base 17) to prevent single-joint noise spikes

                    palm_vel = np.mean(self._lm_velocity[[0, 5, 17], :2], axis=0)

                    palm_speed = float(np.linalg.norm(palm_vel))

                    alpha_val = float(np.clip(0.15 + (0.95 - 0.15) * (1.0 - np.exp(-3.0 * palm_speed)), 0.15, 0.95))

                    alphas = np.full((21, 1), alpha_val, dtype=np.float32)

                else:

                    alphas = 0.5

            else:

                self._lm_velocity = np.zeros_like(raw_lm_np)

                alphas = 1.0

            if self._lm_ema is None or self._lm_ema.shape != raw_lm_np.shape:

                self._lm_ema = raw_lm_np.copy()

            else:

                self._lm_ema = alphas * raw_lm_np + (1.0 - alphas) * self._lm_ema

            self._last_raw_lm = raw_lm_np.copy()

            self._last_ts = now

            # Pointing score & Index Extension Primacy Hysteresis Schmitt Trigger
            pointing_score, index_score = check_finger_curl(raw_lm_np, world_lm_np, is_flat_down)

            if index_score >= 0.50:
                self._pointing_active = True
            elif index_score <= 0.35:
                self._pointing_active = False

            smooth_lm_np = self._lm_ema

        else:

            # Missed frame: linearly extrapolate up to MISS_TOLERANCE_S

            if self._last_raw_lm is None or self._lm_velocity is None or self._lm_ema is None:

                self.reset()

                return

            dt_miss = now - self._last_ts

            smooth_lm_np = self._last_raw_lm + self._lm_velocity * dt_miss

            pointing_score = 1.0 if self._pointing_active else 0.0

        def clamp(v: float, lo: float = 0.0, hi: float = 1.0) -> float:

            return max(lo, min(hi, v))

            

        # ── Anchor projection & Weighted Dual-Vector Ray Blend ─────────────────

        tip_x = float(smooth_lm_np[8, 0])

        tip_y = float(smooth_lm_np[8, 1])

        # Screen anchor for skeleton rendering (so it doesn't rubber-band)

        skel_anchor_x = tip_x

        skel_anchor_y = tip_y

        if pointing_score > 0.05:

            # Weighted Dual-Vector ray calculation: 60% Wrist->Base (0->5), 40% Base->Tip (5->8)

            v_05 = smooth_lm_np[5, :2] - smooth_lm_np[0, :2]

            v_58 = smooth_lm_np[8, :2] - smooth_lm_np[5, :2]

            n_58 = float(np.linalg.norm(v_58))

            if is_flat_down or n_58 < 0.035:

                dx, dy = float(v_05[0]), float(v_05[1])

            else:

                v_blend = 0.6 * v_05 + 0.4 * v_58

                dx, dy = float(v_blend[0]), float(v_blend[1])

            if world_lm_np is not None:

                depth_span_3d = float(np.linalg.norm(world_lm_np[9, :3] - world_lm_np[0, :3]))

                ray_length_multiplier = 1.6 * max(0.6, min(2.5, REFERENCE_3D_SPAN / max(depth_span_3d, 0.01)))

            else:

                ray_length_multiplier = 1.6

            projected_x = smooth_lm_np[0, 0] + dx * ray_length_multiplier

            projected_y = smooth_lm_np[0, 1] + dy * ray_length_multiplier

            

            # Continuous projection blend based on pointing score

            p = pointing_score

            raw_x = (1.0 - p) * tip_x + p * float(projected_x)

            raw_y = (1.0 - p) * tip_y + p * float(projected_y)

        else:
            raw_x = tip_x
            raw_y = tip_y

        def clamp(v: float, lo: float = 0.0, hi: float = 1.0) -> float:

            return max(lo, min(hi, v))

            

        # ── Anchor projection ────────────────────────────────────────────────

        tip_x = float(smooth_lm_np[8, 0])

        tip_y = float(smooth_lm_np[8, 1])

        # Screen anchor for skeleton rendering (so it doesn't rubber-band)

        skel_anchor_x = tip_x

        skel_anchor_y = tip_y

        if pointing_score > 0.05:

            # Ray from Wrist (0) to Index Base (5)

            dx = smooth_lm_np[5, 0] - smooth_lm_np[0, 0]

            dy = smooth_lm_np[5, 1] - smooth_lm_np[0, 1]

            if world_lm_np is not None:

                depth_span_3d = float(np.linalg.norm(world_lm_np[9, :3] - world_lm_np[0, :3]))

                ray_length_multiplier = 1.6 * max(0.6, min(2.5, REFERENCE_3D_SPAN / max(depth_span_3d, 0.01)))

            else:

                ray_length_multiplier = 1.6

            projected_x = smooth_lm_np[0, 0] + dx * ray_length_multiplier

            projected_y = smooth_lm_np[0, 1] + dy * ray_length_multiplier

            

            # Continuous projection blend based on pointing score

            p = pointing_score

            raw_x = (1.0 - p) * tip_x + p * float(projected_x)

            raw_y = (1.0 - p) * tip_y + p * float(projected_y)

        else:

            raw_x = tip_x

            raw_y = tip_y

        

        # Apply OneEuro Filter to Cursor X/Y

        cursor_pos = self._cursor_filter(now, np.array([raw_x, raw_y]))

        smooth_x, smooth_y = float(cursor_pos[0]), float(cursor_pos[1])

        # Fix 2: Consume display bounds from DisplayManager (single source of truth)

        dm = display_manager.get_snapshot()

        active_x_min  = dm['active_x_min']

        active_x_max  = dm['active_x_max']

        active_y_min  = dm['active_y_min']

        active_y_max  = dm['active_y_max']

        target_width  = dm['target_width']

        target_height = dm['target_height']

        def map_to_screen(x, y):

            mapped_x = clamp((x - active_x_min) / (active_x_max - active_x_min)) if (active_x_max - active_x_min) != 0 else 0.5

            mapped_y = clamp((y - active_y_min) / (active_y_max - active_y_min)) if (active_y_max - active_y_min) != 0 else 0.5

            def apply_edge_acceleration(val: float, power: float = 1.3) -> float:

                nx = (val - 0.5) * 2.0

                nx = math.copysign(abs(nx) ** power, nx)

                return (nx / 2.0) + 0.5

            mapped_x = apply_edge_acceleration(mapped_x)

            mapped_y = apply_edge_acceleration(mapped_y)

            sx = max(0, min(target_width  - 1, int((1.0 - mapped_x) * target_width)))

            sy = max(0, min(target_height - 1, int(mapped_y * target_height)))

            return sx, sy

            

        screen_x, screen_y = map_to_screen(smooth_x, smooth_y)

        skel_sx, skel_sy   = map_to_screen(skel_anchor_x, skel_anchor_y)

        # Fix 6: Micro-tremor velocity deadzone — freeze pixel coords when hovering

        if self._last_screen_x is not None:

            if math.hypot(screen_x - self._last_screen_x, screen_y - self._last_screen_y) < 2.0:

                screen_x, screen_y = self._last_screen_x, self._last_screen_y

            else:

                self._last_screen_x, self._last_screen_y = screen_x, screen_y

        else:

            self._last_screen_x, self._last_screen_y = screen_x, screen_y

        # ── 4. Dwell-to-click detection (gated by Hysteresis Schmitt Trigger) ─

        # Use skeleton anchor (raw index tip) for dwell origin — it's far more

        # stable than the OneEuro-filtered + ray-projected cursor coordinates.

        cursor_state, dwell_progress = self._fsm.tick(

            now, self._pointing_active, skel_sx, skel_sy, DWELL_RADIUS_PX

        )

        if cursor_state == "CLICK":

            log.info("Dwell click fired at (%d, %d)", screen_x, screen_y)

        # Swipe history uses the smoothed value (same axis as cursor)

        self._x_history.append((now, smooth_x))

        while self._x_history and now - self._x_history[0][0] > SWIPE_HISTORY_DURATION_S:

            self._x_history.popleft()

            

        gesture = "NONE"

        if now >= self._swipe_cooldown_until and len(self._x_history) >= 2:

            dt_swipe = self._x_history[-1][0] - self._x_history[0][0]

            if dt_swipe > 0.05:

                dx       = self._x_history[-1][1] - self._x_history[0][1]

                velocity = dx / dt_swipe

                if abs(velocity) > SWIPE_VELOCITY_THRESHOLD:

                    gesture = "SWIPE_RIGHT" if velocity < 0 else "SWIPE_LEFT"

                    self._swipe_cooldown_until = now + SWIPE_COOLDOWN_S

                    self._fsm.dwell_origin = None

                    log.info("Gesture: %s  (vel=%.4f)", gesture, velocity)

        publish_lm = smooth_lm_np.tolist()

        # ── 7. Publish ────────────────────────────────────────────────────

        # Update shared state with skel_anchor for drawing

        self._state.skel_anchor_x = float(skel_sx)

        self._state.skel_anchor_y = float(skel_sy)

        self._state.update(

            screen_x, screen_y, cursor_state, gesture,

            hand_detected=True,

            dwell_progress=dwell_progress,

            landmarks=publish_lm,

            hand_size=1.0,            # fixed size

            depth_z=1.0,              # fixed size

        )

        # ── 8. OS mouse control ──────────────────────────────────

        if self._mouse is not None:

            self._mouse.position = (screen_x, screen_y)

            

        if cursor_state == "CLICK" and self._prev_cursor_state != "CLICK":

            if self._ui is not None:

                # Evdev native click

                try:

                    import evdev

                    self._ui.write(evdev.ecodes.EV_KEY, evdev.ecodes.BTN_LEFT, 1)

                    self._ui.write(evdev.ecodes.EV_KEY, evdev.ecodes.BTN_LEFT, 0)

                    self._ui.syn()

                    log.info("Evdev click injected at (%d, %d)", screen_x, screen_y)

                except Exception as e:

                    log.error(f"Evdev click failed: {e}")

            else:

                # Fallback to xdotool

                def _inject_click():

                    try:

                        if not hasattr(self, "_cached_screenflex_wins"):

                            r = subprocess.run(["xdotool", "search", "--name", "Screenflex"],

                                               capture_output=True, text=True, timeout=3)

                            self._cached_screenflex_wins = [w for w in r.stdout.strip().split() if w.isdigit()]

                        wins = self._cached_screenflex_wins

                        if wins:

                            for wid in wins:

                                subprocess.run(["xdotool", "click", "--window", wid, "1"], capture_output=True, timeout=3)

                        else:

                            subprocess.run(["xdotool", "click", "1"], capture_output=True, timeout=3)

                        log.info("OS click injected at (%d, %d)", screen_x, screen_y)

                    except Exception as e:

                        log.warning("Click injection failed: %s", e)

                threading.Thread(target=_inject_click, daemon=True).start()

                

        self._prev_cursor_state = cursor_state

    def reset(self) -> None:

        """Clear tracking history."""

        self._last_raw_lm = None

        self._lm_velocity = None

        self._lm_ema = None

        self._last_screen_x = None

        self._last_screen_y = None

        self._pointing_active = False

        self._fsm.dwell_origin = None

        self._x_history.clear()

        self._swipe_cooldown_until = 0.0

        self._prev_cursor_state  = "MOVE"

        

        self._state.skel_anchor_x = float(self._state.x)

        self._state.skel_anchor_y = float(self._state.y)

        self._state.update(

            self._state.x, self._state.y, "MOVE", "NONE",

            hand_detected=False,

            dwell_progress=0.0,

            landmarks=[],

            hand_size=1.0,

            depth_z=1.0,

        )

# ===========================================================================

# ── Camera helpers ────────────────────────────────────────────────────────

# ===========================================================================

class CameraStream:

    """Background thread for grabbing frames from the camera."""

    def __init__(self, camera_index: int, width: int, height: int, fps: int, stop_event: threading.Event) -> None:

        self.camera_index = camera_index

        self.width = width

        self.height = height

        self.fps = fps

        self.stop_event = stop_event

        self.cap = None

        self.frame = None

        self.ret = False

        self.stopped = False

        self.lock = threading.Lock()

    def start(self) -> bool:

        log.info("Opening webcam (index=%d) …", self.camera_index)

        backend = cv2.CAP_DSHOW if sys.platform == "win32" else cv2.CAP_V4L2

        self.cap = cv2.VideoCapture(self.camera_index, backend)

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH,  self.width)

        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)

        self.cap.set(cv2.CAP_PROP_FPS,          self.fps)

        self.cap.set(cv2.CAP_PROP_BUFFERSIZE,   1)

        # Request MJPEG from the camera driver — delivers compressed frames over

        # USB, cutting bandwidth ~10× vs raw YUYV and reducing decode CPU on the Pi.

        # Falls back silently to YUYV if the camera doesn't support MJPEG.

        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))

        if not self.cap.isOpened():

            log.error("Cannot open webcam index=%d.", self.camera_index)

            return False

        log.info(

            "Webcam ready: %dx%d @ %.0f fps",

            self.cap.get(cv2.CAP_PROP_FRAME_WIDTH),

            self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT),

            self.cap.get(cv2.CAP_PROP_FPS),

        )

        

        self.ret, frame = self.cap.read()

        if self.ret:

            self.frame = frame.copy()

        threading.Thread(target=self.update, daemon=True).start()

        return True

    def update(self) -> None:

        frames_read = 0

        while not self.stopped and not self.stop_event.is_set():

            if self.cap is not None:

                ret, frame = self.cap.read()

                with self.lock:

                    self.ret = ret

                    if ret:

                        self.frame = frame.copy()

                        frames_read += 1

                if not ret:

                    time.sleep(0.01)

    def read(self):

        with self.lock:

            if not self.ret or getattr(self, 'frame', None) is None:

                return False, None

            return self.ret, self.frame.copy()

    def release(self) -> None:

        self.stopped = True

        if self.cap is not None:

            self.cap.release()

def _open_camera(stop_event: threading.Event) -> Optional[CameraStream]:

    stream = CameraStream(CAMERA_INDEX, CAMERA_WIDTH, CAMERA_HEIGHT, TARGET_FPS, stop_event)

    if stream.start():

        return stream

    return None

def _rate_limit(frame_interval: float, last_ts: float) -> None:

    """Sleep until the next frame deadline. Pure sleep — no spin-wait."""

    remaining = frame_interval - (time.perf_counter() - last_ts)

    if remaining > 0.001:          # only sleep if there's meaningful time left

        time.sleep(remaining)

def validate_camera_access() -> bool:

    """

    Test camera access at startup.

    Provides helpful error messages for permission issues.

    """

    log.info(f"Checking camera access ({CAMERA_INDEX})...")

    try:

        cap = cv2.VideoCapture(CAMERA_INDEX)

        if not cap.isOpened():

            log.error(

                "❌ Camera not accessible. On Linux, run:\n"

                "   sudo usermod -aG video $USER\n"

                "   # Then log out and back in for changes to take effect"

            )

            return False

        ret, frame = cap.read()

        cap.release()

        if not ret:

            log.error("❌ Camera found but cannot read frames. Check drivers/permissions.")

            return False

        log.info(f"✓ Camera accessible ({frame.shape})")

        return True

    except Exception as e:

        log.error(f"❌ Camera check failed: {e}")

        return False

# ===========================================================================

# ── Camera loop — Tasks API (mediapipe 0.10+) ────────────────────────────

# ===========================================================================

def _camera_loop_tasks(

    shared_state: GestureState, stop_event: threading.Event

) -> None:

    from mediapipe.tasks import python as _mp_python

    from mediapipe.tasks.python import vision as _mp_vision

    _ensure_model()

    base_opts = _mp_python.BaseOptions(

        model_asset_path=_MODEL_PATH,

    )

    processor = GestureProcessor(shared_state)

    result_queue = queue.Queue(maxsize=1)

    in_flight = False
    in_flight_ts = 0.0

    def _result_callback(result, output_image, timestamp_ms: int):

        nonlocal in_flight

        in_flight = False

        try:

            wl = result.hand_world_landmarks[0] if result.hand_world_landmarks else None

            lm = result.hand_landmarks[0] if result.hand_landmarks else None

            # Drain old queue items if any, keep latest

            try:

                while True:

                    result_queue.get_nowait()

            except queue.Empty:

                pass

            result_queue.put_nowait((lm, wl))

        except queue.Full:

            pass

    options   = _mp_vision.HandLandmarkerOptions(

        base_options=base_opts,

        running_mode=_mp_vision.RunningMode.LIVE_STREAM,

        num_hands=1,

        min_hand_detection_confidence=DETECT_CONFIDENCE,

        min_hand_presence_confidence=PRESENCE_CONFIDENCE,

        min_tracking_confidence=TRACK_CONFIDENCE,

        result_callback=_result_callback,

    )

    with _mp_vision.HandLandmarker.create_from_options(options) as landmarker:

        cap = None

        reconnect_attempt = 0

        max_reconnect_attempts = 5

        reconnect_delay = 2.0

        frame_interval = 1.0 / TARGET_FPS

        last_ts  = 0.0

        last_ts_ms = 0

        geometry_poll_ctr = 0

        read_fail_count = 0     # tolerate transient USB frame drops

        last_hand_seen_ts = 0.0  # grace-period anchor for thermal throttle

        while not stop_event.is_set():

            try:

                if cap is None:

                    log.info(f"Opening camera {CAMERA_INDEX}...")

                    cap = _open_camera(stop_event)

                    if cap is None:

                        raise RuntimeError(f"Camera {CAMERA_INDEX} not available")

                    log.info("Camera opened successfully")

                    reconnect_attempt = 0

                    read_fail_count = 0

                    

                    with shared_state._lock:

                        shared_state._hand_detected = False

                        shared_state._gesture = "NONE"

                # Grace period: keep 20 FPS for 1.5s after losing the hand,
                # preventing motion blur amplification and death spirals.
                if shared_state.hand_detected:

                    last_hand_seen_ts = time.perf_counter()

                grace_active = (time.perf_counter() - last_hand_seen_ts) < 1.5

                target_fps = 20.0 if (shared_state.hand_detected or grace_active) else 12.0

                frame_interval = 1.0 / target_fps

                _rate_limit(frame_interval, last_ts)

                ret, frame = cap.read()

                if not ret or frame is None:

                    read_fail_count += 1

                    if read_fail_count > 10:

                        raise RuntimeError(

                            f"Camera returned {read_fail_count} consecutive "

                            "bad frames — likely disconnected"

                        )

                    continue

                read_fail_count = 0

                last_ts   = time.perf_counter()

                # Windows-only dev convenience: cheap poll

                geometry_poll_ctr += 1

                if geometry_poll_ctr >= TARGET_FPS * 2:  # ~every 2 seconds

                    geometry_poll_ctr = 0

                    display_manager.update()

                # Fast downscale for MediaPipe (320x240 for ultra-fast CPU inference)

                small_frame = cv2.resize(frame, (320, 240), interpolation=cv2.INTER_LINEAR)

                rgb = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

                ts_ms    = int(time.perf_counter() * 1000)

                if ts_ms <= last_ts_ms:

                    ts_ms = last_ts_ms + 1

                last_ts_ms = ts_ms

                if in_flight and (time.perf_counter() - in_flight_ts > 0.2):
                    log.warning("MediaPipe callback timeout. Releasing in_flight lock.")
                    in_flight = False

                if not in_flight:

                    in_flight = True

                    in_flight_ts = time.perf_counter()

                    try:
                        landmarker.detect_async(mp_image, ts_ms)
                    except Exception as e:
                        log.warning("MediaPipe inference submission failed: %s", e)
                        in_flight = False

                

                try:

                    while True:

                        lm, wl = result_queue.get_nowait()

                        if lm:

                            processor.process(lm, wl)

                        else:

                            processor.miss()

                except queue.Empty:

                    pass

            except Exception as e:

                log.error(f"Camera error: {e}")

                if cap is not None:

                    try:

                        cap.release()

                    except Exception:

                        pass

                    cap = None

                    

                reconnect_attempt += 1

                if reconnect_attempt > max_reconnect_attempts:

                    log.critical(f"Camera reconnection failed after {max_reconnect_attempts} attempts.")

                    log.critical("Giving up. Check camera hardware/permissions.")

                    stop_event.set()

                    break

                    

                wait_time = reconnect_delay * (2 ** min(reconnect_attempt - 1, 3))

                log.info(f"Retrying camera in {wait_time:.1f}s (attempt {reconnect_attempt})")

                time.sleep(wait_time)

    if cap is not None:

        cap.release()

    log.info("Camera loop stopped.")

# ===========================================================================

# ── Camera thread dispatcher ──────────────────────────────────────────────

# ===========================================================================

def camera_loop(

    shared_state: GestureState, stop_event: threading.Event

) -> None:

    _camera_loop_tasks(shared_state, stop_event)

# ===========================================================================

# ── WebSocket server ──────────────────────────────────────────────────────

# ===========================================================================

_connected_clients: set = set()

async def ws_handler(websocket) -> None:

    log.info("Client connected: %s", websocket.remote_address)

    _connected_clients.add(websocket)

    try:

        # overlay.py pushes {"type": "geometry", "width": W, "height": H}

        # on connect and whenever the OS reports a rotation/resolution

        # change. This is the authoritative geometry path — see

        # DisplayManager.set_external_geometry() for why.

        async for raw in websocket:

            try:

                msg = json.loads(raw)

            except json.JSONDecodeError:

                continue

            if msg.get("type") == "geometry":

                try:

                    w = int(msg["width"])

                    h = int(msg["height"])

                except (KeyError, TypeError, ValueError):

                    continue

                if display_manager.set_external_geometry(w, h):

                    log.info("Geometry received from overlay.py: %dx%d", w, h)

    finally:

        _connected_clients.discard(websocket)

        log.info("Client disconnected: %s", websocket.remote_address)

async def broadcast_loop(

    shared_state: GestureState, stop_event: threading.Event

) -> None:

    interval = 1.0 / TARGET_FPS

    loop = asyncio.get_running_loop()

    while not stop_event.is_set():

        start = loop.time()

        if _connected_clients:

            # Only serialise when at least one client is connected

            payload = shared_state.to_json()

            await asyncio.gather(

                *[_send_safe(ws, payload) for ws in list(_connected_clients)],

                return_exceptions=True,

            )

        elapsed = loop.time() - start

        await asyncio.sleep(max(0.0, interval - elapsed))

async def _send_safe(ws, payload: str) -> None:

    try:

        await ws.send(payload)

    except Exception:

        pass

async def run_server(

    shared_state: GestureState, stop_event: threading.Event

) -> None:

    log.info("WebSocket server starting on ws://%s:%d", WS_HOST, WS_PORT)

    async with websockets.serve(ws_handler, WS_HOST, WS_PORT):

        log.info("WebSocket server ready.")

        await broadcast_loop(shared_state, stop_event)

# ===========================================================================

# ── Entry Point ───────────────────────────────────────────────────────────

# ===========================================================================

def main() -> None:

    log.info("Starting engine. Camera and display will initialize shortly.")

    # Validate camera BEFORE spinning up threads

    if not validate_camera_access():

        raise SystemExit("Camera initialization failed")

    shared_state = GestureState()

    stop_event   = threading.Event()

    # ── Camera thread ──────────────────────────────────────────────────

    cam_thread = threading.Thread(

        target=camera_loop,

        args=(shared_state, stop_event),

        name="CameraThread",

        daemon=True,

    )

    cam_thread.start()

    # ── GUI & Asyncio integration ──────────────────────────────────────

    try:

        from PyQt5.QtWidgets import QApplication

        import qasync

        from overlay import OverlayWindow

    except ImportError as e:

        log.critical(f"Missing UI dependency: {e}. Please install PyQt5 and qasync.")

        stop_event.set()

        cam_thread.join()

        sys.exit(1)

    if sys.platform != "win32":

        os.environ.setdefault("QT_XCB_NATIVE_PAINTING", "1")

        os.environ.setdefault("LIBGL_ALWAYS_SOFTWARE", "0")

        os.environ.setdefault("QT_LOGGING_RULES", "qt.qpa.*=false")

    app = QApplication(sys.argv)

    app.setApplicationName("GestureEngine")

    # Set qasync as the asyncio event loop

    loop = qasync.QEventLoop(app)

    asyncio.set_event_loop(loop)

    # Initialize the OverlayWindow (connects it directly to the shared_state)

    def geometry_changed(w: int, h: int) -> None:

        display_manager.set_external_geometry(w, h)

        log.info(f"Display resolution updated to {w}x{h}")

        

    window = OverlayWindow(shared_state, geometry_callback=geometry_changed)

    window.show()

    # Start the lazy WS server on the qasync loop

    loop.create_task(run_server(shared_state, stop_event))

    log.info("Running with unified qasync event loop. Press Ctrl+C in terminal to stop.")

    try:

        with loop:

            loop.run_forever()

    except KeyboardInterrupt:

        log.info("Shutdown requested (Ctrl+C).")

    finally:

        stop_event.set()

        cam_thread.join(timeout=3.0)

        log.info("Engine stopped cleanly.")

if __name__ == "__main__":

    main()
