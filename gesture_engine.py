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
import queue
import logging
import math
import os
import platform
import subprocess
import threading
import time
import sys
import shutil
import urllib.request
from collections import deque
from typing import List, Optional

import cv2
cv2.setNumThreads(2)
import numpy as np

import depth_metrics

# ===========================================================================
# ── MediaPipe API Detection ───────────────────────────────────────────────
# ===========================================================================

try:
    import mediapipe as mp

    if hasattr(mp, "solutions") and hasattr(mp.solutions, "hands"):
        _USE_TASKS_API = False
        _hands_mod = mp.solutions.hands
        _api_label = "legacy solutions API"
    else:
        raise AttributeError("solutions namespace missing")

except AttributeError:
    try:
        import mediapipe as mp
        _USE_TASKS_API = True
        _api_label = "Tasks API (0.10+)"
    except ImportError as exc:
        raise SystemExit(
            f"mediapipe import failed: {exc}\n"
            "Install with:  pip install mediapipe"
        ) from exc

except ModuleNotFoundError as exc:
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

class AsyncMouse:
    def __init__(self):
        self.x = -1
        self.y = -1
        try:
            from pynput.mouse import Controller
            self._mouse = Controller()
        except ImportError:
            self._mouse = None
            log.warning("pynput not installed. OS mouse control disabled.")

    def set_position(self, x: int, y: int) -> None:
        if self._mouse is None:
            return
        self.x, self.y = x, y
        try:
            self._mouse.position = (x, y)
        except Exception:
            pass

async_mouse = AsyncMouse()

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
TARGET_FPS:    int = 30
CAMERA_ROTATION: int = 0

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
        "x": 0.60,  # 60% of width (requires less physical horizontal sweep)
        "y": 0.45   # 45% of height (comfortable vertical reach)
    },
    "portrait": {
        "x": 0.40,  # 40% of width (lighter movement horizontally)
        "y": 0.60   # 60% of height (safe cushion for reaching top/bottom)
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
    if not shutil.which("xrandr"):
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
            self.rotation = 90
        else:
            self.target_width, self.target_height = _BASE_LONG, _BASE_SHORT
            self.rotation = 0

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
        
        self._bound_emas = {
            'xmin': _SimpleEMA(0.25),
            'xmax': _SimpleEMA(0.25),
            'ymin': _SimpleEMA(0.25),
            'ymax': _SimpleEMA(0.25),
        }
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
                'scale_y': self.scale_y,
                'rotation': self.rotation
            }

    def get_depth_adjusted_bounds(self, hand_size: float, ref_size: float) -> dict:
        """
        Dynamically adjusts the active tracking bounds based on hand depth.
        Window always centered at (0.5, 0.5) — no recentering drift.
        Bounds are smoothed via EMA to prevent jitter.
        """
        with self._lock:
            # Flip the z-ratio: close (large size) = large z, far (small size) = small z
            z = hand_size / max(0.001, ref_size)
            
            # Lock tracking window size (do not scale with depth 'z').
            # Prevents rubber-banding and erratic sensitivity when hand closes (z drops falsely).
            ax = self._base_ax_range
            ay = self._base_ay_range
            
            # Fixed center at (0.5, 0.5) — no rubber-band recentering
            raw_xmin = 0.5 - ax / 2.0
            raw_xmax = 0.5 + ax / 2.0
            raw_ymin = 0.5 - ay / 2.0
            raw_ymax = 0.5 + ay / 2.0
            
            # Bounds Smoothing / Hysteresis
            return {
                'active_x_min': self._bound_emas['xmin'].update(raw_xmin),
                'active_x_max': self._bound_emas['xmax'].update(raw_xmax),
                'active_y_min': self._bound_emas['ymin'].update(raw_ymin),
                'active_y_max': self._bound_emas['ymax'].update(raw_ymax),
                'target_width': self.target_width,
                'target_height': self.target_height
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
            
            # Wayland (e.g. Raspberry Pi OS Bookworm) often reports rotation=0 
            # via xrandr even when physically rotated. If the overlay pushes a 
            # portrait geometry but xrandr says 0, infer 90 degrees.
            if new_rotation == 0 and h > w:
                new_rotation = 90
            
        changed = False
        with self._lock:
            if w != self.target_width or h != self.target_height or new_rotation != self.rotation:
                changed = True
                
                if new_rotation != self.rotation:
                    log.info("Display rotation changed: %d°", new_rotation)
                
                self.target_width  = w
                self.target_height = h
                self.rotation      = new_rotation

        if changed:
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

        new_rotation = 90 if h > w else 0

        with self._lock:
            if w == prev_w and h == prev_h and new_rotation == self.rotation:
                return False

            self.target_width = w
            self.target_height = h
            self.rotation = new_rotation
        self._recompute_margins()
        return True

display_manager = DisplayManager()


# ── Cursor Smoothing — One Euro Filter ────────────────────────────────────────────
# Adaptive low-pass filter: still positions get heavy smoothing, fast
# movements pass through with minimal lag.
# Ref: Casiez et al., "1€ Filter", CHI 2012.
#
# Tuned for Index Finger Tip (LM8).
#
#   OEF_MIN_CUTOFF  — Hz. Lower → smoother at rest, less micro-jitter.
#   OEF_BETA        — speed coeff. Higher → snappier tracking on fast moves.
#
OEF_MIN_CUTOFF: float = 0.4   # lower = highly stable at rest (great for dwell aiming)
OEF_BETA:       float = 5.0    # higher = snaps instantly to fast hand movements
OEF_D_CUTOFF:   float = 1.0    # Hz — derivative filter (rarely needs changing)

# ── Pointing Mode Stabilization ───────────────────────────────────────────
# When the index finger is extended and other fingers are curled, the cursor
# tip (LM 8) is the noisiest landmark due to angular amplification from the
# palm→finger chain. Two mechanisms combine to fix this:
#
# 1. GEOMETRIC ANCHOR BLEND: cursor reads from a weighted point between the
#    fingertip (LM 8) and the index MCP knuckle (LM 5). This is a small
#    bias correction — NOT the primary stabilizer. It targets specifically
#    joint-angle jitter, where LM 8 moves relative to LM 5. Whole-hand
#    translation (both points move together) passes through at full gain.
#    NOTE: this intentionally reduces fingertip aiming precision by ~12%
#    in exchange for eliminating ~80% of rest-state jitter.
#    alpha=1.0 → pure tip (no change), alpha=0.0 → pure knuckle.
#
# 2. OEF RETUNE: the primary stabilization. Pointing-specific min_cutoff
#    and beta are used instead of the default values, applied with a slow
#    EMA gate to prevent the filter params themselves from oscillating.
#
POINTING_ANCHOR_ALPHA: float   = 0.88   # small geometric bias: tip-weighted, not knuckle-heavy
POINTING_OEF_MIN_CUTOFF: float = 0.35   # lower rest cutoff → more aggressive jitter suppression (tuned up slightly for responsiveness)
POINTING_OEF_BETA: float       = 4.5    # lower speed coeff → calmer, less rubber-band on fast moves (tuned up for snappiness)

# ── Pointing-Mode Hysteresis ──────────────────────────────────────────────
# Hard boolean mode-switches per-frame are worse than the problem they solve:
# if the hand is near the curl threshold, is_pointing can flicker frame-to-
# frame and simultaneously flip the anchor blend, OEF params, and flow mask.
# These constants enforce a minimum dwell before switching:
#   POINTING_ENTER_FRAMES — consecutive frames of pointing pose required to
#                           enter pointing mode. ~5 frames at 30fps = ~170ms.
#   POINTING_EXIT_FRAMES  — consecutive frames of non-pointing pose required
#                           to leave pointing mode. Longer = hysteresis.
#   POINTING_BLEND_FRAMES — number of frames to linearly interpolate the
#                           anchor alpha and OEF params at mode transitions,
#                           preventing hard parameter jumps.
POINTING_ENTER_FRAMES: int = 5
POINTING_EXIT_FRAMES:  int = 8
POINTING_BLEND_FRAMES: int = 10

# ── Overlay landmark smoothing ────────────────────────────────────────────
# Gentle EMA applied to each of the 21 landmarks before sending to the
# overlay renderer — prevents dot flickering without adding noticeable lag.
# Lower alpha = more stable dots, Higher alpha = more responsive/accurate.
LM_EMA_ALPHA: float = 0.75     # 0.75 = highly responsive: follows fast movements immediately
# ── Dwell-to-Click Detection ──────────────────────────────────────────────
# A click fires when the hand cursor stays within DWELL_RADIUS_PX pixels
# of its starting position for DWELL_DURATION_S seconds.
DWELL_DURATION_S: float  = 1.2    # seconds of stillness required
DWELL_RADIUS_PX: int     = 30     # max pixel drift before resetting the timer
DWELL_COOLDOWN_S: float  = 1.5    # seconds to wait after a click before accepting another

# ── Swipe Detection ────────────────────────────────────────────────────────
SWIPE_HISTORY_LEN: int          = 10
SWIPE_VELOCITY_THRESHOLD: float = 0.018
SWIPE_COOLDOWN_FRAMES: int      = 20
MISS_TOLERANCE_FRAMES: int      = 10     # ~330ms at 30fps grace period for continuity

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
DETECT_CONFIDENCE: float   = 0.50
TRACK_CONFIDENCE: float    = 0.55   # increased to drop bad tracking (fists, etc.) faster
PRESENCE_CONFIDENCE: float = 0.55   # Tasks API equivalent of TRACK_CONFIDENCE

# ── WebSocket ──────────────────────────────────────────────────────────────
WS_HOST: str = "localhost"
WS_PORT: int = 8765

# ── Pseudo-3D Physical Mapping ─────────────────────────────────────────────
# Distance-dependent size/velocity correction
REFERENCE_SIZE: float = 0.15   # baseline palm metric at comfortable distance
Z_MIN: float          = 0.4    # nearest zoom bound
Z_MAX: float          = 2.5    # furthest zoom bound
SIZE_EMA_ALPHA: float = 0.05   # slow time constant for depth metric




# ── Model complexity (legacy solutions API only) ───────────────────────────
#   0 = "lite"  — ~3x cheaper than complexity 1, small accuracy trade-off.
#                 Recommended on any Raspberry Pi.
#   1 = "full"  — heavier model, only worth it on desktop-class CPUs.
MODEL_COMPLEXITY: int = 1   # full model for better closed-fist detection

# ── Motion-Adaptive Frame Skipping ─────────────────────────────────────────
# Run MediaPipe dynamically based on motion to save CPU/thermal budget.
# When hand moves > MOTION_THRESHOLD, we use ACTIVE_SKIP_TARGET.
# When hand is relatively still, we use DWELL_SKIP_TARGET.
MOTION_THRESHOLD: float   = 3.5
ACTIVE_SKIP_TARGET: int   = 4
DWELL_SKIP_TARGET: int    = 8

# ── Optical Flow Drift Prevention ──────────────────────────────────────────
# Force MediaPipe re-detect every N frames to prevent drift accumulation.
# At 30fps this means re-detection every ~1 second.
FLOW_MAX_AGE_FRAMES: int  = 15
# Max pixel drift any single landmark is allowed from its last MediaPipe
# position before we force a re-detect. Prevents skeleton stretching from
# accumulated per-landmark optical flow tracking errors.
FLOW_DRIFT_CLAMP_PX: float = 10.0

# ── Tier 1: Presence Gate & Idle Mode ──────────────────────────────────────
# Radically reduce CPU when no one is using the kiosk.
IDLE_FPS: int = 8
GATE_RESOLUTION = (160, 120)
GATE_DIFF_THRESH: int = 25
GATE_PIXEL_THRESH: int = 500
IDLE_TIMEOUT_S: float = 1.0
SAFETY_NET_INTERVAL: float = 2.0

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
        self._hand_size = 1.0
        self._depth_z = 1.0
        self._is_pointing = False

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
                'hand_size': self._hand_size,
                'depth_z': self._depth_z,
                'is_pointing': self._is_pointing,
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
        is_pointing: bool = False,
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
            self._is_pointing = is_pointing

    def to_json(self) -> str:
        with self._lock:
            d = {
                "x": self._cursor_x,
                "y": self._cursor_y,
                "state": self._cursor_state,
                "gesture": self._gesture,
                "hand_detected": self._hand_detected,
                "dwell_progress": round(self._dwell_progress, 3),
                "hand_size": self._hand_size,
                "depth_z": self._depth_z,
                "is_pointing": self._is_pointing,
            }
            if self._hand_detected and self._landmarks:
                d["landmarks"] = list(self._landmarks)
                
        dm_snap = display_manager.get_snapshot()
        d['active_x_min']  = dm_snap['active_x_min']
        d['active_x_max']  = dm_snap['active_x_max']
        d['active_y_min']  = dm_snap['active_y_min']
        d['active_y_max']  = dm_snap['active_y_max']
        d['screen_width']  = dm_snap['target_width']
        d['screen_height'] = dm_snap['target_height']
        d['scale_x']       = dm_snap['scale_x']
        d['scale_y']       = dm_snap['scale_y']
        d['camera_width']  = CAMERA_WIDTH
        d['camera_height'] = CAMERA_HEIGHT
        d['cam_aspect']    = round(CAMERA_WIDTH / CAMERA_HEIGHT, 4) if CAMERA_HEIGHT else 1.333
        d['rotation']      = 0
        d['display_w'] = dm_snap['target_width']
        d['display_h'] = dm_snap['target_height']
        
        return json.dumps(d, separators=(",", ":"))


# ===========================================================================
# ── One Euro Filter ───────────────────────────────────────────────────────
# ===========================================================================

class _OneEuroFilter:
    """
    Adaptive low-pass filter for smooth pointer / cursor input.

    Slow or stationary input is heavily smoothed (removes jitter).
    Fast input passes through with minimal added lag.

    Parameters
    ----------
    freq        Nominal input sample rate in Hz (e.g. TARGET_FPS).
    min_cutoff  Minimum cutoff frequency (Hz).  Lower → smoother at rest.
    beta        Speed coefficient.  Higher → quicker response to fast motion.
    d_cutoff    Cutoff for the derivative estimate (usually 1.0 Hz).

    Reference: Géry Casiez et al., "1€ Filter: A Simple Speed-based Low-pass
               Filter for Noisy Input in Interactive Systems", CHI 2012.
    """

    def __init__(
        self,
        freq:       float,
        min_cutoff: float = 1.0,
        beta:       float = 0.007,
        d_cutoff:   float = 1.0,
    ) -> None:
        self._freq       = freq
        self._min_cutoff = min_cutoff
        self._beta       = beta
        self._d_cutoff   = d_cutoff
        self._x:  Optional[float] = None   # last filtered value
        self._dx: float           = 0.0    # last filtered derivative

    @staticmethod
    def _alpha(cutoff: float, freq: float) -> float:
        """First-order low-pass coefficient for the given cutoff frequency."""
        tau = 1.0 / (2.0 * math.pi * cutoff)
        te  = 1.0 / freq
        return 1.0 / (1.0 + tau / te)

    def filter(self, x: float, dt: Optional[float] = None) -> float:
        """
        Filter a new sample.

        Parameters
        ----------
        x   Raw (noisy) sample.
        dt  Seconds since the last sample.  None → use constructor freq.
        """
        freq = (1.0 / dt) if (dt and dt > 0) else self._freq

        if self._x is None:          # bootstrap
            self._x = x
            return x

        # Filtered derivative
        dx_raw   = (x - self._x) * freq
        alpha_d  = self._alpha(self._d_cutoff, freq)
        self._dx = self._dx + alpha_d * (dx_raw - self._dx)

        # Adaptive cutoff: faster motion → higher cutoff → less lag
        cutoff = self._min_cutoff + self._beta * abs(self._dx)

        # Filtered signal
        alpha   = self._alpha(cutoff, freq)
        self._x = self._x + alpha * (x - self._x)
        return self._x

    def reset(self) -> None:
        """Reset to uninitialised state (call when tracking is lost)."""
        self._x  = None
        self._dx = 0.0


# ===========================================================================
# ── Gesture Processor ─────────────────────────────────────────────────────
# ===========================================================================

class GestureProcessor:
    """Stateful per-frame gesture interpreter."""
    
    def __init__(self, state: GestureState) -> None:
        self._state = state
        self._last_ts = 0.0
        
        # Adaptive cursor smoothing
        self._filter_x = _OneEuroFilter(TARGET_FPS)
        self._filter_y = _OneEuroFilter(TARGET_FPS)
        
        self._prev_screen_x: Optional[int] = None
        self._prev_screen_y: Optional[int] = None

        self._lm_filters = None  # populated on first frame

        # ── Fix A: Pre-allocated numpy buffers ────────────────────────────────
        # Avoids ~120 heap allocations/second from np.array() and np.zeros_like()
        # in the per-frame hot path.
        self._raw_lm_buf      = np.zeros((21, 2), dtype=np.float32)
        self._filtered_lm_buf = np.zeros((21, 2), dtype=np.float32)

        # Depth size metric filter
        self._size_filter = depth_metrics.EMAFilter(alpha=SIZE_EMA_ALPHA)

        # Dwell-to-click state
        self._dwell_origin_x: Optional[int] = None
        self._dwell_origin_y: Optional[int] = None
        self._dwell_start_ts: float         = 0.0
        self._dwell_cooldown_until: float   = 0.0

        # Swipe detection
        self._x_history: deque[float] = deque(maxlen=SWIPE_HISTORY_LEN)
        self._swipe_cooldown: int     = 0

        self._mouse = async_mouse
        self._prev_cursor_state: str = "MOVE"
        
        # 5. Auto-Calibration for REFERENCE_SIZE
        self._ref_size: float = REFERENCE_SIZE
        self._calibrating: bool = True
        self._calibration_start_ts: float = 0.0
        self._calibration_samples: List[float] = []
        
        # Dedicated depth_z smoother — separate from the size metric EMA
        # used for bounds calculation. This one is specifically for the
        # skeleton scale factor sent to the overlay.
        self._depth_z_ema = _SimpleEMA(alpha=0.08)  # very smooth skeleton scaling

        # ── Pointing-Mode State Machine ───────────────────────────────────────
        self.is_pointing: bool = False
        self._point_enter_ctr: int = 0
        self._point_exit_ctr:  int = 0
        self._point_blend: float = 0.0

        # Separate slow EMA for OEF parameter tuning only.
        self._oef_depth_ema = _SimpleEMA(alpha=0.03)

        # ── Fix C: Cached OEF blend state ─────────────────────────────────────
        # Skip re-applying filter params when neither _point_blend nor depth has
        # changed meaningfully, saving 6 float assignments per stable frame.
        self._last_oef_depth_z: float   = -1.0
        self._last_point_blend: float   = -1.0
        self._cached_lm_min_cutoff: float = OEF_MIN_CUTOFF
        self._cached_lm_beta: float       = OEF_BETA

    @staticmethod
    def _is_pointing_raw(raw_lm_np: np.ndarray) -> bool:
        """
        Rotation-invariant pointing detection using joint-distance ratios.

        Y-only comparisons (tip.y < pip.y) break when the hand is tilted
        sideways or downward relative to the camera. Instead, we compare the
        distance each fingertip is from the wrist vs. its MCP knuckle's distance
        from the wrist — a ratio that is stable under any in-plane rotation.

        A finger is EXTENDED when its tip is farther from the wrist than its
        MCP knuckle (ratio > 1.0 with a tolerance margin).
        A finger is CURLED  when its tip is closer to the wrist than its MCP.

        MediaPipe landmark indices:
          Wrist: 0
          Index:  MCP=5,  PIP=6,  DIP=7,  Tip=8
          Middle: MCP=9,  PIP=10, DIP=11, Tip=12
          Ring:   MCP=13, PIP=14, DIP=15, Tip=16
          Pinky:  MCP=17, PIP=18, DIP=19, Tip=20
        """
        wrist = raw_lm_np[0]

        def tip_mcp_ratio(tip_idx: int, mcp_idx: int) -> float:
            """tip-to-wrist distance / mcp-to-wrist distance."""
            d_tip = float(np.linalg.norm(raw_lm_np[tip_idx] - wrist))
            d_mcp = float(np.linalg.norm(raw_lm_np[mcp_idx] - wrist))
            return d_tip / max(d_mcp, 1e-6)

        # Index extended: tip farther from wrist than MCP (ratio > 1.15 headroom)
        index_ext   = tip_mcp_ratio(8,  5) > 1.15

        # Other fingers curled: tip closer to wrist than MCP (ratio < 0.95)
        middle_curl = tip_mcp_ratio(12, 9)  < 0.95
        ring_curl   = tip_mcp_ratio(16, 13) < 0.95
        pinky_curl  = tip_mcp_ratio(20, 17) < 0.95
        curled_count = sum([middle_curl, ring_curl, pinky_curl])

        # Require index extended AND at least 2 other fingers curled
        return index_ext and curled_count >= 2

    def process(self, lms) -> None:
        """
        lms — iterable of 21 landmark objects with .x / .y (normalised 0-1).
              Works with both NormalizedLandmarkList.landmark (legacy)
              and list[NormalizedLandmark] (Tasks API).
        """
        now = time.perf_counter()

        # ── 1. Extract and Rotate Landmarks ─────────────────────────────────
        # Fix A: Fill the pre-allocated buffer in-place rather than allocating
        # a new array every frame (avoids ~60 numpy allocs/sec).
        raw_lm_np = self._raw_lm_buf
        for _i, _lm in enumerate(lms):
            raw_lm_np[_i, 0] = _lm.x
            raw_lm_np[_i, 1] = _lm.y

        # Camera axes align with the physical world — no rotation needed.
        # raw_lm_np is already in the correct frame.

        # ── Pointing-Mode Detection + Hysteresis ─────────────────────────────
        # Classify each frame independently, then require N consecutive frames
        # before committing to a mode change. Prevents threshold-edge flickering
        # from simultaneously flipping the anchor blend, OEF params, and the
        # optical-flow landmark mask used by the camera loop.
        is_raw = self._is_pointing_raw(raw_lm_np)
        if is_raw:
            self._point_enter_ctr += 1
            self._point_exit_ctr   = 0
        else:
            self._point_exit_ctr  += 1
            self._point_enter_ctr  = 0

        if not self.is_pointing and self._point_enter_ctr >= POINTING_ENTER_FRAMES:
            self.is_pointing = True
            log.debug("Pointing mode: ENTER")
        elif self.is_pointing and self._point_exit_ctr >= POINTING_EXIT_FRAMES:
            self.is_pointing = False
            log.debug("Pointing mode: EXIT")

        # Advance blend scalar toward target (0=normal, 1=pointing) by 1/BLEND_FRAMES
        # per frame — smoothly interpolates anchor alpha and OEF params across
        # every mode transition instead of hard-switching them.
        blend_target = 1.0 if self.is_pointing else 0.0
        blend_step   = 1.0 / max(1, POINTING_BLEND_FRAMES)
        if self._point_blend < blend_target:
            self._point_blend = min(blend_target, self._point_blend + blend_step)
        elif self._point_blend > blend_target:
            self._point_blend = max(blend_target, self._point_blend - blend_step)

        # ── Cursor Anchor ───────────────────────────────────────────────────
        # In pointing mode, blend LM 8 (tip) toward LM 5 (index MCP knuckle).
        # Targets joint-angle jitter specifically — where LM 8 moves relative
        # to LM 5 due to micro-oscillations. Whole-hand translation (both
        # points moving together) is unaffected and passes through at full gain.
        # INTENTIONAL TRADE-OFF: reduces ~12% of fine fingertip aiming precision
        # in exchange for eliminating the dominant jitter source at rest.
        # The overlay skeleton still renders at true LM 8 (cursor != skeleton).
        eff_alpha = POINTING_ANCHOR_ALPHA * self._point_blend + 1.0 * (1.0 - self._point_blend)
        raw_x = float(eff_alpha * raw_lm_np[8, 0] + (1.0 - eff_alpha) * raw_lm_np[5, 0])
        raw_y = float(eff_alpha * raw_lm_np[8, 1] + (1.0 - eff_alpha) * raw_lm_np[5, 1])


        def clamp(v: float, lo: float = 0.0, hi: float = 1.0) -> float:
            return max(lo, min(hi, v))
        s_raw = depth_metrics.get_rigid_palm_metric(raw_lm_np)
        s = max(self._size_filter.filter(s_raw), 0.001)
        
        if self._calibrating:
            if self._calibration_start_ts == 0.0:
                self._calibration_start_ts = now
                log.info("Starting hand depth calibration...")
            
            self._calibration_samples.append(s)
            
            if now - self._calibration_start_ts > 1.5:  # 1.5s calibration window
                self._ref_size = sum(self._calibration_samples) / len(self._calibration_samples)
                self._calibrating = False
                log.info(f"Calibration complete. New REFERENCE_SIZE: {self._ref_size:.4f}")
            
            # During calibration, lock the cursor by not emitting mouse events
            # But we still update the filters so it doesn't jump wildly after calibration
            # We'll just exit early after updating EMAs.
        
        # Compute the depth ratio for skeleton scaling in overlay
        raw_depth_z = s / max(0.001, self._ref_size)
        # Clamp to a sane visual range, then smooth heavily to prevent
        # the skeleton from jittering/reshaping on every frame
        depth_z = max(0.5, min(2.0, self._depth_z_ema.update(raw_depth_z)))
        
        # ── Depth-adaptive One Euro Filter tuning ─────────────────────────
        # OEF params are driven by a SEPARATE slow EMA (alpha=0.03, ~33-frame
        # time constant) rather than directly from depth_z. This decouples the
        # filter's own parameters from fast depth fluctuations — the root cause
        # of the "rubber band" effect where min_cutoff and beta oscillate each
        # frame, making the filter itself a noise source.
        oef_depth_z  = max(0.5, min(2.0, self._oef_depth_ema.update(raw_depth_z)))
        depth_factor = max(1.0, math.sqrt(oef_depth_z))

        # Blend OEF params between default and pointing-specific using _point_blend.
        target_min_cutoff = (POINTING_OEF_MIN_CUTOFF * self._point_blend
                             + OEF_MIN_CUTOFF         * (1.0 - self._point_blend))
        target_beta       = (POINTING_OEF_BETA * self._point_blend
                             + OEF_BETA        * (1.0 - self._point_blend))

        # Fix C: Only re-apply filter params when depth or blend has changed
        # meaningfully. Skips 6 float assignments on stable frames.
        _depth_changed = abs(oef_depth_z - self._last_oef_depth_z) > 0.01
        _blend_changed = abs(self._point_blend - self._last_point_blend) > 0.005
        if _depth_changed or _blend_changed:
            self._filter_x._min_cutoff = target_min_cutoff * depth_factor
            self._filter_y._min_cutoff = target_min_cutoff * depth_factor
            self._filter_x._beta       = target_beta / depth_factor
            self._filter_y._beta       = target_beta / depth_factor
            self._cached_lm_min_cutoff = target_min_cutoff * depth_factor
            self._cached_lm_beta       = target_beta / depth_factor
            self._last_oef_depth_z     = oef_depth_z
            self._last_point_blend     = self._point_blend


        # ── 2. One Euro Filter — adaptive smooth cursor ─────────────────────
        dt      = (now - self._last_ts) if self._last_ts > 0 else None
        self._last_ts = now
        smooth_x = self._filter_x.filter(raw_x, dt)
        smooth_y = self._filter_y.filter(raw_y, dt)
        
        if self._calibrating:
            # Feed the display manager EMAs even while calibrating to pre-warm the center
            display_manager.get_depth_adjusted_bounds(s, REFERENCE_SIZE)
            return

        # ── 3. Normalised → screen pixels (mirror X for natural movement) ─
        # Get dynamic tracking bounds based on current depth
        dm_snap = display_manager.get_depth_adjusted_bounds(s, self._ref_size)
        active_x_min = dm_snap['active_x_min']
        active_x_max = dm_snap['active_x_max']
        active_y_min = dm_snap['active_y_min']
        active_y_max = dm_snap['active_y_max']
        target_width = dm_snap['target_width']
        target_height = dm_snap['target_height']

        mapped_x = clamp((smooth_x - active_x_min) / (active_x_max - active_x_min)) if (active_x_max - active_x_min) != 0 else 0.5
        mapped_y = clamp((smooth_y - active_y_min) / (active_y_max - active_y_min)) if (active_y_max - active_y_min) != 0 else 0.5

        def apply_edge_acceleration(val: float, power: float = 1.0) -> float:
            nx = (val - 0.5) * 2.0
            nx = math.copysign(abs(nx) ** power, nx)
            return (nx / 2.0) + 0.5

        mapped_x = apply_edge_acceleration(mapped_x)
        mapped_y = apply_edge_acceleration(mapped_y)

        screen_x = max(0, min(target_width  - 1, int((1.0 - mapped_x) * target_width)))
        screen_y = max(0, min(target_height - 1, int(mapped_y * target_height)))

        # ── Cursor Dead-Zone (Stabilization) ──────────────────────────────
        if self._prev_screen_x is not None and self._prev_screen_y is not None:
            dist = math.hypot(screen_x - self._prev_screen_x, screen_y - self._prev_screen_y)
            DEAD_ZONE_PX = 4.0   # ignore sub-pixel drift while "aiming"
            if dist < DEAD_ZONE_PX:
                screen_x = self._prev_screen_x
                screen_y = self._prev_screen_y

        self._prev_screen_x = screen_x
        self._prev_screen_y = screen_y

        # ── 4. Dwell-to-click detection ───────────────────────────────────
        in_cooldown = now < self._dwell_cooldown_until

        if in_cooldown:
            # Suppress input during cooldown; keep progress at 0
            dwell_progress = 0.0
            cursor_state   = "MOVE"
            # Reset dwell origin so dwell restarts fresh after cooldown
            self._dwell_origin_x = screen_x
            self._dwell_origin_y = screen_y
            self._dwell_start_ts = now
        else:
            # Initialise origin on first frame after appearance / cooldown end
            if self._dwell_origin_x is None:
                self._dwell_origin_x = screen_x
                self._dwell_origin_y = screen_y
                self._dwell_start_ts = now
            else:
                # Check whether hand has drifted out of the dwell zone
                drift = math.hypot(
                    screen_x - self._dwell_origin_x,
                    screen_y - self._dwell_origin_y,
                )
                edge_factor = max(abs(mapped_x - 0.5), abs(mapped_y - 0.5)) * 2.0
                dynamic_radius = DWELL_RADIUS_PX + (DWELL_RADIUS_PX * 1.5 * edge_factor)
                
                if drift > dynamic_radius:
                    # Moved too much — restart the timer at the new position
                    self._dwell_origin_x = screen_x
                    self._dwell_origin_y = screen_y
                    self._dwell_start_ts = now

            elapsed        = now - self._dwell_start_ts
            dwell_progress = min(1.0, elapsed / DWELL_DURATION_S)

            if dwell_progress >= 1.0:
                cursor_state             = "CLICK"
                self._dwell_cooldown_until = now + DWELL_COOLDOWN_S
                # Reset origin — next dwell starts from scratch after cooldown
                self._dwell_origin_x = None
                log.info("Dwell click fired at (%d, %d)", screen_x, screen_y)
            else:
                cursor_state = "MOVE"

        # Swipe history uses the One-Euro-filtered value (same axis as cursor)
        self._x_history.append(smooth_x)
        gesture = "NONE"

        if self._swipe_cooldown > 0:
            self._swipe_cooldown -= 1
        elif len(self._x_history) == SWIPE_HISTORY_LEN:
            dx       = self._x_history[-1] - self._x_history[0]
            velocity = dx / SWIPE_HISTORY_LEN
            if abs(velocity) > SWIPE_VELOCITY_THRESHOLD:
                gesture = "SWIPE_RIGHT" if velocity < 0 else "SWIPE_LEFT"
                self._swipe_cooldown = SWIPE_COOLDOWN_FRAMES
                self._dwell_origin_x = None
                log.info("Gesture: %s  (vel=%.4f)", gesture, velocity)

        # ── 6. Per-landmark Uniform EMA Filter for smooth overlay dots ────────────────
        if self._lm_filters is None:
            self._lm_filters = []
            for i in range(21):
                self._lm_filters.append((
                    _SimpleEMA(alpha=LM_EMA_ALPHA),
                    _SimpleEMA(alpha=LM_EMA_ALPHA)
                ))

        filtered_lm_np = self._filtered_lm_buf
        for i in range(21):
            filtered_lm_np[i, 0] = self._lm_filters[i][0].update(float(raw_lm_np[i, 0]))
            filtered_lm_np[i, 1] = self._lm_filters[i][1].update(float(raw_lm_np[i, 1]))

        # We purposely do NOT clamp the landmarks to [0.0, 1.0] here.
        # Clamping at the edges causes the hand skeleton to compress against 
        # the boundary and stretch like rubber when the hand moves partially off-screen.

        publish_lm = filtered_lm_np.tolist()

        # ── 7. Publish ────────────────────────────────────────────────────
        self._state.update(
            screen_x, screen_y, cursor_state, gesture,
            hand_detected=True,
            dwell_progress=dwell_progress,
            landmarks=publish_lm,        # smoothed dots sent to overlay (true LM 8)
            hand_size=s,                 # smoothed depth proxy
            depth_z=depth_z,             # depth ratio
            is_pointing=self.is_pointing, # persistent mode flag (separate from gesture enum)
        )

        # ── 8. OS mouse control (xdotool) ──────────────────────────────────
        if self._mouse is not None:
            # Move the system cursor asynchronously so it never blocks the camera loop
            self._mouse.set_position(screen_x, screen_y)
            # Fire a single left-click only on the MOVE → CLICK transition
            if cursor_state == "CLICK" and self._prev_cursor_state != "CLICK":
                def _inject_click():
                    try:
                        from pynput.mouse import Controller, Button
                        mouse = Controller()
                        mouse.click(Button.left, 1)
                        log.info("OS click injected at (%d, %d)", screen_x, screen_y)
                    except ImportError:
                        log.warning("pynput not installed. Click skipped.")
                    except Exception as e:
                        log.warning("Click injection failed: %s", e)
                
                # Run the click injection asynchronously so it doesn't freeze the camera loop
                threading.Thread(target=_inject_click, daemon=True).start()
                
        self._prev_cursor_state = cursor_state

    def reset(self) -> None:
        """Called when no hand is detected — clears transient state."""
        self._filter_x.reset()
        self._filter_y.reset()
        self._last_ts        = 0.0
        self._lm_filters         = None      # cleared so it re-seeds on next detection
        self._prev_screen_x      = None
        self._prev_screen_y      = None
        self._dwell_origin_x = None
        self._dwell_origin_y = None
        self._x_history.clear()
        self._swipe_cooldown     = 0
        self._prev_cursor_state  = "MOVE"
        # Reset pointing-mode state so re-detection starts from a clean slate
        self.is_pointing      = False
        self._point_enter_ctr = 0
        self._point_exit_ctr  = 0
        self._point_blend     = 0.0
        self._state.update(
            self._state.x, self._state.y, "MOVE", "NONE",
            hand_detected=False,
            dwell_progress=0.0,
            landmarks=[],
            hand_size=1.0,
            depth_z=1.0,
            is_pointing=False,
        )



# ===========================================================================
# ── Camera helpers ────────────────────────────────────────────────────────
# ===========================================================================

LK_PARAMS = dict(
    winSize=(45, 45),  # Expanded window to catch fast swipes
    maxLevel=2,
    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
)

class FlowLandmark:
    __slots__ = ['x', 'y']
    def __init__(self, x: float, y: float):
        self.x = x
        self.y = y

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
            self.stop_event.set()
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
        while not self.stopped and not self.stop_event.is_set():
            if self.cap is not None:
                ret, frame = self.cap.read()
                with self.lock:
                    self.ret = ret
                    if ret:
                        self.frame = frame.copy()
                if not ret:
                    time.sleep(0.01)
                else:
                    time.sleep(0.001)

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
# ── Camera loop — Legacy solutions API ───────────────────────────────────
# ===========================================================================

def _camera_loop_legacy(
    shared_state: GestureState, stop_event: threading.Event
) -> None:
    processor = GestureProcessor(shared_state)

    with _hands_mod.Hands(
        static_image_mode=False,
        max_num_hands=4,
        model_complexity=MODEL_COMPLEXITY,
        min_detection_confidence=DETECT_CONFIDENCE,
        min_tracking_confidence=TRACK_CONFIDENCE,
    ) as hands:

        cap = None
        reconnect_attempt = 0
        max_reconnect_attempts = 5
        reconnect_delay = 2.0
        frame_interval = 1.0 / TARGET_FPS
        last_ts  = 0.0
        skip_ctr = 0
        flow_age = 0
        miss_streak = 0
        prev_gray = None
        prev_pts = None
        _last_mp_pts = None
        geometry_poll_ctr = 0
        read_fail_count = 0
        
        request_q = queue.Queue(maxsize=1)
        result_q = queue.Queue()
        
        def mp_worker():
            while not stop_event.is_set():
                try:
                    req = request_q.get(timeout=0.1)
                except queue.Empty:
                    continue
                if req is None: break
                req_seq, req_rgb, req_roi_info = req
                
                try:
                    res = hands.process(req_rgb)
                except Exception as e:
                    log.error(f"Worker Error: {e}")
                    res = None
                    
                while True:
                    try: result_q.get_nowait()
                    except queue.Empty: break
                    
                result_q.put((req_seq, res, req_roi_info))
                
        worker_thread = threading.Thread(target=mp_worker, daemon=True)
        worker_thread.start()
        
        seq = 0
        history_pts = {}
        pipeline_mode = "ACTIVE"
        prev_gate_gray = None
        last_hand_seen_time = time.perf_counter()
        last_mp_time = 0.0

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

                current_fps = IDLE_FPS if pipeline_mode == "IDLE" else TARGET_FPS
                frame_interval = 1.0 / current_fps
                _rate_limit(frame_interval, last_ts)

                ret, frame = cap.read()
                if not ret or frame is None:
                    read_fail_count += 1
                    if read_fail_count > 10:
                        raise RuntimeError("Camera returned consecutive bad frames")
                    continue
                read_fail_count = 0

                geometry_poll_ctr += 1
                if geometry_poll_ctr >= TARGET_FPS * 2:
                    geometry_poll_ctr = 0
                    display_manager.update()

                now = time.perf_counter()
                last_ts = now
                seq += 1
                
                curr_gray_full = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                
                # --- Tier 1: Presence Gate ---
                small_gray = cv2.resize(curr_gray_full, GATE_RESOLUTION)
                gate_active = True
                if prev_gate_gray is not None:
                    diff = cv2.absdiff(prev_gate_gray, small_gray)
                    _, thresh = cv2.threshold(diff, GATE_DIFF_THRESH, 255, cv2.THRESH_BINARY)
                    gate_active = (cv2.countNonZero(thresh) > GATE_PIXEL_THRESH)
                prev_gate_gray = small_gray
                
                # --- Check Worker Results ---
                try:
                    res_seq, results, roi_info = result_q.get_nowait()
                    if results and results.multi_hand_landmarks:
                        last_hand_seen_time = now
                        if pipeline_mode == "IDLE":
                            log.info("Hand detected! Waking up to ACTIVE mode (%.0f fps).", TARGET_FPS)
                            pipeline_mode = "ACTIVE"
                        
                        miss_streak = 0
                        best_size = -1
                        best_hand_lm = None
                        for hlms in results.multi_hand_landmarks:
                            lms = hlms.landmark
                            _xy = np.array([[lm.x, lm.y] for lm in lms], dtype=np.float32)
                            _mn = _xy.min(axis=0)
                            _mx = _xy.max(axis=0)
                            size = (_mx[0] - _mn[0]) * (_mx[1] - _mn[1])
                            if size > best_size:
                                best_size = size
                                best_hand_lm = lms
                                
                        if best_hand_lm:
                            roi_x0, roi_y0, roi_h, roi_w, w, h = roi_info
                            pts = np.zeros((21, 1, 2), dtype=np.float32)
                            full_lms = []
                            for i, lm in enumerate(best_hand_lm):
                                full_x = (lm.x * roi_w) + roi_x0
                                full_y = (lm.y * roi_h) + roi_y0
                                pts[i, 0, 0] = full_x
                                pts[i, 0, 1] = full_y
                                full_lms.append(FlowLandmark(full_x / w, full_y / h))
                                
                            # Late fusion offset
                            if res_seq in history_pts and prev_pts is not None:
                                delta = prev_pts - history_pts[res_seq]
                                prev_pts = pts + delta
                            else:
                                prev_pts = pts
                                
                            _last_mp_pts = prev_pts.copy()
                            prev_gray = curr_gray_full
                            
                            adjusted_lms = []
                            for pt in prev_pts:
                                adjusted_lms.append(FlowLandmark(float(pt[0][0])/w, float(pt[0][1])/h))
                                
                            processor.process(adjusted_lms)
                            flow_age = 0
                    else:
                        miss_streak += 1
                        if miss_streak >= MISS_TOLERANCE_FRAMES:
                            processor.reset()
                            prev_pts = None
                            prev_gray = None
                except queue.Empty:
                    pass
                
                if pipeline_mode == "IDLE":
                    if not gate_active and (now - last_mp_time) <= SAFETY_NET_INTERVAL:
                        continue
                    skip_ctr = 0
                
                current_skip_target = DWELL_SKIP_TARGET
                skip_ctr += 1
                flow_age += 1
                
                flow_successful = False
                if prev_gray is not None and prev_pts is not None:
                    if flow_age <= FLOW_MAX_AGE_FRAMES:
                        curr_pts, status, err = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray_full, prev_pts, None, **LK_PARAMS)
                        valid_points = status is not None and np.sum(status) >= 15
                        if valid_points and skip_ctr <= DWELL_SKIP_TARGET:
                            _CRITICAL_LMS = [0, 5, 8, 17]
                            _critical_ok = all(status[ci][0] == 1 for ci in _CRITICAL_LMS)
                            if not _critical_ok:
                                flow_age = FLOW_MAX_AGE_FRAMES
                            else:
                                status_mask = (status == 1).flatten()
                                deltas = curr_pts[status_mask] - prev_pts[status_mask]
                                if len(deltas) > 0:
                                    dx, dy = np.median(deltas, axis=0)[0]
                                    motion = math.hypot(dx, dy)
                                    if motion > MOTION_THRESHOLD:
                                        current_skip_target = ACTIVE_SKIP_TARGET
                                    if skip_ctr < current_skip_target:
                                        rigid_pts = prev_pts + np.array([[[dx, dy]]], dtype=np.float32)
                                        _drift_ok = True
                                        if _last_mp_pts is not None:
                                            max_drift = float(np.max(np.abs(rigid_pts - _last_mp_pts)))
                                            if max_drift > FLOW_DRIFT_CLAMP_PX:
                                                flow_age = FLOW_MAX_AGE_FRAMES
                                                _drift_ok = False
                                        if _drift_ok:
                                            prev_gray = curr_gray_full
                                            prev_pts = rigid_pts
                                            h, w = frame.shape[:2]
                                            flow_lms = [FlowLandmark(float(pt[0][0])/w, float(pt[0][1])/h) for pt in rigid_pts]
                                            processor.process(flow_lms)
                                            miss_streak = 0
                                            flow_successful = True
                                            history_pts[seq] = prev_pts.copy()
                                            
                # Cleanup history
                history_pts.pop(seq - 30, None)
                
                if flow_successful:
                    last_hand_seen_time = now
                    if skip_ctr < current_skip_target:
                        continue
                        
                # Dispatch to worker
                skip_ctr = 0
                frame.flags.writeable = False
                h, w = frame.shape[:2]
                roi_x0, roi_y0, roi_x1, roi_y1 = 0, 0, w, h
                
                if prev_pts is not None:
                    palm_pts = prev_pts[[0, 1, 5, 9, 13, 17], 0, :]
                    min_x, max_x = np.min(palm_pts[:, 0]), np.max(palm_pts[:, 0])
                    min_y, max_y = np.min(palm_pts[:, 1]), np.max(palm_pts[:, 1])
                    margin_x = max((max_x - min_x) * 1.5, 60.0)
                    margin_y = max((max_y - min_y) * 1.5, 60.0)
                    roi_x0 = max(0, int(min_x - margin_x))
                    roi_y0 = max(0, int(min_y - margin_y))
                    roi_x1 = min(w, int(max_x + margin_x))
                    roi_y1 = min(h, int(max_y + margin_y))
                    if (roi_x1 - roi_x0) < 50 or (roi_y1 - roi_y0) < 50:
                        roi_x0, roi_y0, roi_x1, roi_y1 = 0, 0, w, h
                        
                roi_frame = frame[roi_y0:roi_y1, roi_x0:roi_x1]
                roi_h, roi_w = roi_frame.shape[:2]
                scale = 480.0 / max(roi_h, roi_w)
                if scale < 1.0:
                    small_frame = cv2.resize(roi_frame, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
                    rgb = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
                else:
                    rgb = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2RGB)
                    
                try:
                    request_q.put_nowait((seq, rgb, (roi_x0, roi_y0, roi_h, roi_w, w, h)))
                    last_mp_time = now
                except queue.Full:
                    pass
                    
                frame.flags.writeable = True
                
                if not flow_successful:
                    if (now - last_hand_seen_time) >= IDLE_TIMEOUT_S and not gate_active:
                        if pipeline_mode == "ACTIVE":
                            log.info("No hand/motion detected for %.1fs. Dropping to IDLE mode (%d fps).", IDLE_TIMEOUT_S, IDLE_FPS)
                            pipeline_mode = "IDLE"
                            
            except Exception as e:
                log.error(f"Camera error: {e}")
                if cap is not None:
                    try: cap.release()
                    except: pass
                    cap = None
                reconnect_attempt += 1
                if reconnect_attempt > max_reconnect_attempts:
                    log.critical("Camera reconnection failed.")
                    stop_event.set()
                    break
                wait_time = reconnect_delay * (2 ** min(reconnect_attempt - 1, 3))
                time.sleep(wait_time)

    request_q.put(None) # kill worker
    if cap is not None: cap.release()
    log.info("Camera loop stopped.")

def _camera_loop_tasks(
    shared_state: GestureState, stop_event: threading.Event
) -> None:
    from mediapipe.tasks import python as _mp_python
    from mediapipe.tasks.python import vision as _mp_vision

    _ensure_model()

    base_opts = _mp_python.BaseOptions(
        model_asset_path=_MODEL_PATH
    )
    options   = _mp_vision.HandLandmarkerOptions(
        base_options=base_opts,
        running_mode=_mp_vision.RunningMode.VIDEO,
        num_hands=4,
        min_hand_detection_confidence=DETECT_CONFIDENCE,
        min_hand_presence_confidence=PRESENCE_CONFIDENCE,
        min_tracking_confidence=TRACK_CONFIDENCE,
    )

    processor = GestureProcessor(shared_state)

    with _mp_vision.HandLandmarker.create_from_options(options) as landmarker:
        cap = None
        reconnect_attempt = 0
        max_reconnect_attempts = 5
        reconnect_delay = 2.0
        frame_interval = 1.0 / TARGET_FPS
        last_ts  = 0.0
        skip_ctr = 0
        flow_age = 0
        miss_streak = 0
        prev_gray = None
        prev_pts = None
        _last_mp_pts = None
        geometry_poll_ctr = 0
        read_fail_count = 0
        
        request_q = queue.Queue(maxsize=1)
        result_q = queue.Queue()
        
        def mp_worker():
            import mediapipe as mp
            while not stop_event.is_set():
                try:
                    req = request_q.get(timeout=0.1)
                except queue.Empty:
                    continue
                if req is None: break
                req_seq, req_rgb, req_roi_info, ts_ms = req
                
                try:
                    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=req_rgb)
                    res = landmarker.detect_for_video(mp_image, ts_ms)
                except Exception as e:
                    log.error(f"Worker Error: {e}")
                    res = None
                    
                while True:
                    try: result_q.get_nowait()
                    except queue.Empty: break
                    
                result_q.put((req_seq, res, req_roi_info))
                
        worker_thread = threading.Thread(target=mp_worker, daemon=True)
        worker_thread.start()
        
        seq = 0
        history_pts = {}
        pipeline_mode = "ACTIVE"
        prev_gate_gray = None
        last_hand_seen_time = time.perf_counter()
        last_mp_time = 0.0

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

                current_fps = IDLE_FPS if pipeline_mode == "IDLE" else TARGET_FPS
                frame_interval = 1.0 / current_fps
                _rate_limit(frame_interval, last_ts)

                ret, frame = cap.read()
                if not ret or frame is None:
                    read_fail_count += 1
                    if read_fail_count > 10:
                        raise RuntimeError("Camera returned consecutive bad frames")
                    continue
                read_fail_count = 0

                geometry_poll_ctr += 1
                if geometry_poll_ctr >= TARGET_FPS * 2:
                    geometry_poll_ctr = 0
                    display_manager.update()

                now = time.perf_counter()
                last_ts = now
                seq += 1
                
                curr_gray_full = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                
                small_gray = cv2.resize(curr_gray_full, GATE_RESOLUTION)
                gate_active = True
                if prev_gate_gray is not None:
                    diff = cv2.absdiff(prev_gate_gray, small_gray)
                    _, thresh = cv2.threshold(diff, GATE_DIFF_THRESH, 255, cv2.THRESH_BINARY)
                    gate_active = (cv2.countNonZero(thresh) > GATE_PIXEL_THRESH)
                prev_gate_gray = small_gray
                
                try:
                    res_seq, results, roi_info = result_q.get_nowait()
                    if results and results.hand_landmarks:
                        last_hand_seen_time = now
                        if pipeline_mode == "IDLE":
                            log.info("Hand detected! Waking up to ACTIVE mode (%.0f fps).", TARGET_FPS)
                            pipeline_mode = "ACTIVE"
                        
                        miss_streak = 0
                        best_size = -1
                        best_hand_lm = None
                        for lms in results.hand_landmarks:
                            _xy = np.array([[lm.x, lm.y] for lm in lms], dtype=np.float32)
                            _mn = _xy.min(axis=0)
                            _mx = _xy.max(axis=0)
                            size = (_mx[0] - _mn[0]) * (_mx[1] - _mn[1])
                            if size > best_size:
                                best_size = size
                                best_hand_lm = lms
                                
                        if best_hand_lm:
                            roi_x0, roi_y0, roi_h, roi_w, w, h = roi_info
                            pts = np.zeros((21, 1, 2), dtype=np.float32)
                            full_lms = []
                            for i, lm in enumerate(best_hand_lm):
                                full_x = (lm.x * roi_w) + roi_x0
                                full_y = (lm.y * roi_h) + roi_y0
                                pts[i, 0, 0] = full_x
                                pts[i, 0, 1] = full_y
                                full_lms.append(FlowLandmark(full_x / w, full_y / h))
                                
                            if res_seq in history_pts and prev_pts is not None:
                                delta = prev_pts - history_pts[res_seq]
                                prev_pts = pts + delta
                            else:
                                prev_pts = pts
                                
                            _last_mp_pts = prev_pts.copy()
                            prev_gray = curr_gray_full
                            
                            adjusted_lms = []
                            for pt in prev_pts:
                                adjusted_lms.append(FlowLandmark(float(pt[0][0])/w, float(pt[0][1])/h))
                                
                            processor.process(adjusted_lms)
                            flow_age = 0
                    else:
                        miss_streak += 1
                        if miss_streak >= MISS_TOLERANCE_FRAMES:
                            processor.reset()
                            prev_pts = None
                            prev_gray = None
                except queue.Empty:
                    pass
                
                if pipeline_mode == "IDLE":
                    if not gate_active and (now - last_mp_time) <= SAFETY_NET_INTERVAL:
                        continue
                    skip_ctr = 0
                
                current_skip_target = DWELL_SKIP_TARGET
                skip_ctr += 1
                flow_age += 1
                
                flow_successful = False
                if prev_gray is not None and prev_pts is not None:
                    if flow_age <= FLOW_MAX_AGE_FRAMES:
                        curr_pts, status, err = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray_full, prev_pts, None, **LK_PARAMS)
                        valid_points = status is not None and np.sum(status) >= 15
                        if valid_points and skip_ctr <= DWELL_SKIP_TARGET:
                            _CRITICAL_LMS = [0, 5, 8, 17]
                            _critical_ok = all(status[ci][0] == 1 for ci in _CRITICAL_LMS)
                            if not _critical_ok:
                                flow_age = FLOW_MAX_AGE_FRAMES
                            else:
                                status_mask = (status == 1).flatten()
                                deltas = curr_pts[status_mask] - prev_pts[status_mask]
                                if len(deltas) > 0:
                                    dx, dy = np.median(deltas, axis=0)[0]
                                    motion = math.hypot(dx, dy)
                                    if motion > MOTION_THRESHOLD:
                                        current_skip_target = ACTIVE_SKIP_TARGET
                                    if skip_ctr < current_skip_target:
                                        rigid_pts = prev_pts + np.array([[[dx, dy]]], dtype=np.float32)
                                        _drift_ok = True
                                        if _last_mp_pts is not None:
                                            max_drift = float(np.max(np.abs(rigid_pts - _last_mp_pts)))
                                            if max_drift > FLOW_DRIFT_CLAMP_PX:
                                                flow_age = FLOW_MAX_AGE_FRAMES
                                                _drift_ok = False
                                        if _drift_ok:
                                            prev_gray = curr_gray_full
                                            prev_pts = rigid_pts
                                            h, w = frame.shape[:2]
                                            flow_lms = [FlowLandmark(float(pt[0][0])/w, float(pt[0][1])/h) for pt in rigid_pts]
                                            processor.process(flow_lms)
                                            miss_streak = 0
                                            flow_successful = True
                                            history_pts[seq] = prev_pts.copy()
                                            
                history_pts.pop(seq - 30, None)
                
                if flow_successful:
                    last_hand_seen_time = now
                    if skip_ctr < current_skip_target:
                        continue
                        
                skip_ctr = 0
                frame.flags.writeable = False
                h, w = frame.shape[:2]
                roi_x0, roi_y0, roi_x1, roi_y1 = 0, 0, w, h
                
                if prev_pts is not None:
                    palm_pts = prev_pts[[0, 1, 5, 9, 13, 17], 0, :]
                    min_x, max_x = np.min(palm_pts[:, 0]), np.max(palm_pts[:, 0])
                    min_y, max_y = np.min(palm_pts[:, 1]), np.max(palm_pts[:, 1])
                    margin_x = max((max_x - min_x) * 1.5, 60.0)
                    margin_y = max((max_y - min_y) * 1.5, 60.0)
                    roi_x0 = max(0, int(min_x - margin_x))
                    roi_y0 = max(0, int(min_y - margin_y))
                    roi_x1 = min(w, int(max_x + margin_x))
                    roi_y1 = min(h, int(max_y + margin_y))
                    if (roi_x1 - roi_x0) < 50 or (roi_y1 - roi_y0) < 50:
                        roi_x0, roi_y0, roi_x1, roi_y1 = 0, 0, w, h
                        
                roi_frame = frame[roi_y0:roi_y1, roi_x0:roi_x1]
                roi_h, roi_w = roi_frame.shape[:2]
                scale = 480.0 / max(roi_h, roi_w)
                if scale < 1.0:
                    small_frame = cv2.resize(roi_frame, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
                    rgb = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
                else:
                    rgb = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2RGB)
                    
                try:
                    ts_ms = int(time.time() * 1000)
                    request_q.put_nowait((seq, rgb, (roi_x0, roi_y0, roi_h, roi_w, w, h), ts_ms))
                    last_mp_time = now
                except queue.Full:
                    pass
                    
                frame.flags.writeable = True
                
                if not flow_successful:
                    if (now - last_hand_seen_time) >= IDLE_TIMEOUT_S and not gate_active:
                        if pipeline_mode == "ACTIVE":
                            log.info("No hand/motion detected for %.1fs. Dropping to IDLE mode (%d fps).", IDLE_TIMEOUT_S, IDLE_FPS)
                            pipeline_mode = "IDLE"
                            
            except Exception as e:
                log.error(f"Camera error: {e}")
                import traceback
                traceback.print_exc()
                if cap is not None:
                    try: cap.release()
                    except: pass
                    cap = None
                reconnect_attempt += 1
                if reconnect_attempt > max_reconnect_attempts:
                    log.critical("Camera reconnection failed.")
                    stop_event.set()
                    break
                wait_time = reconnect_delay * (2 ** min(reconnect_attempt - 1, 3))
                time.sleep(wait_time)

    request_q.put(None) # kill worker
    if cap is not None: cap.release()
    log.info("Camera loop stopped.")


class _WebSocketServer:
    def __init__(self, shared_state):
        self._state = shared_state
        self._stop = threading.Event()
        self._loop = None
        self._thread = None
        
    def start(self):
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        
    def stop(self):
        self._stop.set()
        if self._loop is not None:
            self._loop.call_soon_threadsafe(self._loop.stop)
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            
    async def _handler(self, websocket):
        while not self._stop.is_set():
            try:
                payload = self._state.to_json()
                await websocket.send(payload)
                await asyncio.sleep(1.0 / TARGET_FPS)
            except websockets.ConnectionClosed:
                break
            except Exception as e:
                log.error("WebSocket server error: %s", e)
                break
                
    async def _serve(self):
        try:
            async with websockets.serve(self._handler, WS_HOST, WS_PORT):
                while not self._stop.is_set():
                    await asyncio.sleep(0.5)
        except Exception as e:
            log.error("WebSocket serve error: %s", e)
            
    def _run(self):
        import asyncio
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        try:
            self._loop.run_until_complete(self._serve())
        except Exception as e:
            log.error("WebSocket loop error: %s", e)
        finally:
            self._loop.close()

def start_engine() -> None:
    # ── Set Process Priority to HIGH ──
    try:
        import psutil
        p = psutil.Process(os.getpid())
        if hasattr(psutil, "HIGH_PRIORITY_CLASS"):
            p.nice(psutil.HIGH_PRIORITY_CLASS)
        else:
            p.nice(-10)
    except Exception as e:
        log.warning("Could not set process priority: %s", e)

    shared_state = GestureState()
    stop_event = threading.Event()

    server = _WebSocketServer(shared_state)
    server.start()

    log.info("Starting engine...")
    try:
        if _USE_TASKS_API:
            _camera_loop_tasks(shared_state, stop_event)
        else:
            _camera_loop_legacy(shared_state, stop_event)
    except KeyboardInterrupt:
        log.info("KeyboardInterrupt received. Shutting down gracefully...")
    finally:
        stop_event.set()
        server.stop()
        log.info("Engine shutdown complete.")

if __name__ == "__main__":
    start_engine()
