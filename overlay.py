"""
overlay.py
==========
Transparent fullscreen overlay for a Raspberry Pi kiosk.

Renders the 21 MediaPipe hand landmarks on top of the kiosk application as a
colour-coded skeleton, mirroring the camera so movement feels natural.

Connects to gesture_engine.py via WebSocket (ws://localhost:8765) and
auto-reconnects if the engine is not running yet.

Requirements
------------
    pip install websockets PyQt5
    # OR on Raspberry Pi OS:
    sudo apt install python3-pyqt5 python3-pyqt5.qtwidgets

Compositor note (X11)
---------------------
True window transparency on X11 requires a compositor.
  Raspberry Pi OS Bullseye:   sudo apt install picom
  Raspberry Pi OS Bookworm:   Wayfire (built-in compositor) — works out of the box.

Usage
-----
    # Run alongside gesture_engine.py:
    DISPLAY=:0 python overlay.py

    # To launch on boot (add to /etc/rc.local or systemd service):
    DISPLAY=:0 python /path/to/overlay.py &
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import threading
import math
import time
from typing import List, Optional, Tuple

# ── Ensure Qt finds the display on Raspberry Pi ─────────────────────────────
# When launched from systemd (no shell), these vars may be absent.
# Set them before any Qt import so the platform plugin can initialise.
if sys.platform != "win32":
    # XDG_RUNTIME_DIR: required by Wayland/XWayland; root's dir is /run/user/0
    if "XDG_RUNTIME_DIR" not in os.environ:
        uid = os.getuid()
        os.environ["XDG_RUNTIME_DIR"] = f"/run/user/{uid}"
    # Target XWayland (xcb) — compatible with both pure-X11 and Wayfire/Wayland
    os.environ.setdefault("QT_QPA_PLATFORM", "xcb")
    os.environ.setdefault("DISPLAY", ":0")

from PyQt5.QtCore import Qt, QTimer, pyqtSignal, QThread, QPointF
from PyQt5.QtGui import (
    QColor, QPainter, QPen, QBrush, QFont, QRadialGradient,
)
from PyQt5.QtWidgets import QApplication, QWidget

import websockets

# ===========================================================================
# ── Configuration ─────────────────────────────────────────────────────────
# ===========================================================================

WS_URI        = "ws://localhost:8765"
RECONNECT_SEC = 1.5       # seconds between reconnect attempts
REFRESH_MS    = 16        # ~60 fps repaint timer

# Landmark fade speed (0-1): higher = snappier appear/disappear
FADE_SPEED = 0.12

# ===========================================================================
# ── Hand skeleton topology (mirrors MediaPipe HAND_CONNECTIONS) ───────────
# ===========================================================================

HAND_CONNECTIONS: List[Tuple[int, int]] = [
    # Thumb
    (0, 1), (1, 2), (2, 3), (3, 4),
    # Index
    (0, 5), (5, 6), (6, 7), (7, 8),
    # Middle
    (0, 9), (9, 10), (10, 11), (11, 12),
    # Ring
    (0, 13), (13, 14), (14, 15), (15, 16),
    # Pinky
    (0, 17), (17, 18), (18, 19), (19, 20),
    # Palm knuckle band
    (5, 9), (9, 13), (13, 17),
]

# Fixed high-contrast skeleton palette.
# Near-white core with thick black glove outline ensures visibility on any background.
# No screen-grab or dynamic luminance sampling required.

_DWELL_GREEN_BASE   = QColor( 57, 255,  20)   # vivid neon green
_DWELL_TRACK        = QColor( 57, 255,  20,  45)  # faint track ring
_DWELL_ARC_RADIUS   = 44      # radius of the progress circle (px)
_DWELL_PEN_WIDTH    = 10.0    # bold stroke width for charging arc (px)
_DWELL_BORDER_WIDTH = 13.0    # stroke width of black contrast border (px)

# Finger tip indices — drawn 2 px larger, same colour
_FINGERTIPS = {4, 8, 12, 16, 20}


# ===========================================================================
# ── WebSocket client thread ───────────────────────────────────────────────
# ===========================================================================




# ===========================================================================
# ── Overlay window ────────────────────────────────────────────────────────
# ===========================================================================

class OverlayWindow(QWidget):
    """
    Fullscreen transparent window that paints the hand skeleton.

    The window sits above the kiosk application and passes all mouse/touch
    events through so it never blocks interaction with the underlying app.
    """

    def __init__(self, shared_state=None, geometry_callback=None) -> None:
        super().__init__()
        self._shared_state = shared_state
        self._geometry_callback = geometry_callback

        # ── Window setup ──────────────────────────────────────────────────
        flags = (
            Qt.FramelessWindowHint |
            Qt.WindowStaysOnTopHint |
            Qt.Tool |
            Qt.WindowTransparentForInput
        )
        import sys
        if sys.platform != "win32":
            flags |= Qt.X11BypassWindowManagerHint
            
        self.setWindowFlags(flags)
        
        self.setAttribute(Qt.WA_TranslucentBackground,    True)
        self.setAttribute(Qt.WA_NoSystemBackground,       True)
        # Belt-and-suspenders: Qt's own event passthrough (covers XWayland edge cases)
        self.setAttribute(Qt.WA_TransparentForMouseEvents, True)
        self.showFullScreen()
        # Use the actual primary screen geometry — works on any resolution/Pi display.
        # This (Qt/XCB) is the RELIABLE cross-platform geometry source — unlike
        # gesture_engine.py's ctypes.windll path, which only works on Windows.
        self._screen = QApplication.primaryScreen()
        screen_geo = self._screen.geometry()
        self.setGeometry(screen_geo)
        self.raise_()
        # QScreen.geometryChanged fires when the physical display resolution or
        # rotation changes — NOT when this window is moved/resized.  Connecting
        # to QWidget.geometryChanged instead would create a feedback loop:
        # setGeometry() → geometryChanged → setGeometry() → …
        self._screen.geometryChanged.connect(self._on_screen_geometry_changed)
        # Debounce: rapid geometry signals (e.g. during rotation animation) are
        # collapsed into one update sent 250 ms after the last signal fires.
        self._geo_debounce = QTimer(self)
        self._geo_debounce.setSingleShot(True)
        self._geo_debounce.timeout.connect(self._flush_geometry_update)
        self._pending_geo = None   # (w, h) to send once debounce fires

        # ── State ─────────────────────────────────────────────────────
        self._landmarks:      List[List[float]] = []
        self._hand_detected:  bool              = False
        self._gesture:        str               = "NONE"
        self._cursor_state:   str               = "MOVE"
        self._cursor_x:       int               = 0
        self._cursor_y:       int               = 0
        self._skel_anchor_x:  float             = 0.0
        self._skel_anchor_y:  float             = 0.0
        self._connected:      bool              = True
        self._dwell_progress: float             = 0.0   # 0.0 → 1.0 charging arc
        self._depth_z:        float             = 1.0   # physical depth scale factor

        # Engine-broadcast pixel-per-normalised-unit scales.
        # These incorporate both display resolution AND the active camera crop,
        # so the overlay never has to compute any geometry itself.
        # Fallback = window dimensions (no crop assumed) until first WS frame.
        self._engine_scale_x: float = float(self.width())
        self._engine_scale_y: float = float(self.height())
        # Raw display dims kept for legacy fallback initialisation only.
        self._engine_w: int = self.width()
        self._engine_h: int = self.height()

        # Animation
        self._fade_alpha:         float = 0.0   # hand skeleton opacity
        self._click_pulse:        float = 0.0   # burst ring when click fires
        self._ghost_protect:      int   = 0     # frames to ignore hand_detected=False after click
        self._new_data:           bool  = False # track if new data arrived



        # ── WebSocket ─────────────────────────────────────────────────────
        # WSThread removed. State is read directly in _tick().
        if self._geometry_callback:
            geo = self._screen.geometry()
            self._geometry_callback(geo.width(), geo.height())

        # ── Repaint timer ─────────────────────────────────────────────────
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._tick)
        self._timer.start(REFRESH_MS)

        # NOTE: z-order/raise timers were removed. By using X11BypassWindowManagerHint,
        # the window naturally sits at the absolute top of the X server stack without
        # needing to aggressively call xdotool (which was breaking browser hover detection).

    # ------------------------------------------------------------------
    # Slots
    # ------------------------------------------------------------------

    def _on_connect(self, connected: bool) -> None:
        self._connected = connected
        if connected and self._geometry_callback:
            geo = self._screen.geometry()
            self._geometry_callback(geo.width(), geo.height())

    def _on_screen_geometry_changed(self, geo) -> None:
        """Fires on QScreen rotation / resolution change (not on window resize).
        Debounced 250 ms so rapid intermediate states during a rotation
        animation are not forwarded to the engine.
        """
        self._pending_geo = (geo.width(), geo.height())
        self._geo_debounce.start(250)   # restart timer; fires 250 ms after last signal

    def _flush_geometry_update(self) -> None:
        """Called once, 250 ms after the last geometryChanged signal."""
        if self._pending_geo is None:
            return
        w, h = self._pending_geo
        self._pending_geo = None
        geo = self._screen.geometry()   # re-read for final stable value
        self.setGeometry(geo)
        if self._geometry_callback:
            self._geometry_callback(geo.width(), geo.height())

    def _on_data(self, data: dict) -> None:
        prev_state            = self._cursor_state
        new_hand_detected     = data.get("hand_detected", False)
        new_state             = data.get("state", "MOVE")
        new_x                 = data.get("x", 0)
        new_y                 = data.get("y", 0)
        new_skel_x            = data.get("skel_anchor_x", 0)
        new_skel_y            = data.get("skel_anchor_y", 0)
        new_progress          = data.get("dwell_progress", 0.0)

        if (new_hand_detected != self._hand_detected or
            new_state != self._cursor_state or
            new_x != self._cursor_x or new_y != self._cursor_y or
            new_skel_x != self._skel_anchor_x or new_skel_y != self._skel_anchor_y or
            abs(new_progress - self._dwell_progress) > 0.001):
            self._new_data = True

        self._hand_detected   = new_hand_detected
        self._landmarks       = data.get("landmarks", [])
        self._gesture         = data.get("gesture", "NONE")
        self._cursor_state    = new_state
        self._cursor_x        = new_x
        self._cursor_y        = new_y
        self._skel_anchor_x   = new_skel_x
        self._skel_anchor_y   = new_skel_y
        self._dwell_progress  = new_progress
        self._depth_z         = data.get("depth_z", 1.0)

        self._engine_w = self.width()
        self._engine_h = self.height()
        self._cam_aspect = 640.0 / 480.0

        # Trigger burst animation the moment a click fires
        if self._cursor_state == "CLICK" and prev_state != "CLICK":
            self._click_pulse    = 1.0
            self._ghost_protect  = 20  # ~320 ms guard — covers xdotool disruption window

    # ------------------------------------------------------------------
    # Animation tick
    # ------------------------------------------------------------------

    def _tick(self) -> None:
        if self._shared_state:
            snap = self._shared_state.snapshot()
            self._on_data(snap)
            self._connected = True
        else:
            self._connected = False
            

        # Ghost-protect: ignore hand_detected=False for a few frames after a click.
        # The xdotool click can briefly disrupt MediaPipe, causing a spurious reset()
        # that would fade the skeleton out even though the hand is still present.
        if self._ghost_protect > 0:
            self._ghost_protect -= 1
            # Force detected=True during guard window so alpha stays up
            effective_detected = True
        else:
            effective_detected = self._hand_detected

        target = 1.0 if effective_detected else 0.0

        animating = False
        if abs(self._fade_alpha - target) > 0.001:
            self._fade_alpha += (target - self._fade_alpha) * FADE_SPEED
            animating = True
        else:
            self._fade_alpha = target

        if self._click_pulse > 0.0:
            self._click_pulse = max(0.0, self._click_pulse - 0.08)
            animating = True

        if animating or self._new_data:
            self.update()  # schedule repaint
            self._new_data = False

    # ------------------------------------------------------------------
    # Painting
    # ------------------------------------------------------------------

    def paintEvent(self, _event) -> None:  # noqa: N802
        if self._fade_alpha < 0.01:
            # Nothing visible — skip all drawing
            return

        w = self.width()
        h = self.height()

        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        if self._fade_alpha > 0.01 and len(self._landmarks) == 21:
            self._draw_skeleton(painter, w, h)

        painter.end()

    def _lm_to_screen(
        self, norm_x: float, norm_y: float, w: int, h: int
    ) -> QPointF:
        """
        Convert normalised landmark coords to screen pixels.
        Mirrors X so the visual matches screen direction.
        """
        return QPointF((1.0 - norm_x) * w, norm_y * h)


    # ------------------------------------------------------------------
    # Skeleton drawing
    # ------------------------------------------------------------------

    def _draw_skeleton(self, painter: QPainter, w: int, h: int) -> None:
        alpha = self._fade_alpha
        lms   = self._landmarks

        # Fixed palette — white bones, black glove outline.
        # Alpha is pre-multiplied by the fade so the skeleton appears/disappears smoothly.
        WHITE_CORE = QColor(240, 245, 255, int(235 * alpha))
        BLACK_OUT  = QColor(  0,   0,   0, int(210 * alpha))
        WHITE_GLOW = QColor(200, 215, 255, int( 55 * alpha))

        if len(lms) < 21:
            return

        # ── Proportional Skeleton Mapping ──────────────────────────────────
        # Anchor at Index Fingertip (LM 8) physical mapped position, not the active cursor!
        anchor_norm_x, anchor_norm_y = lms[8][0], lms[8][1]
        
        # skel_anchor is the mapped pixel coordinate of the actual fingertip, before pointing projection.
        anchor_screen_x = float(getattr(self, '_skel_anchor_x', self._cursor_x))
        anchor_screen_y = float(getattr(self, '_skel_anchor_y', self._cursor_y))
        
        def clamp(v: float, lo: float, hi: float) -> float:
            return max(lo, min(hi, v))

        TARGET_SKEL_SIZE = 1
        raw_scale = TARGET_SKEL_SIZE / (max(0.6, self._depth_z) ** 0.8)
        SKELETON_SCALE = clamp(raw_scale, 0.85, 1.15)

        base_size = min(self._engine_w, self._engine_h)
        skel_scale_x = base_size
        skel_scale_y = base_size / getattr(self, "_cam_aspect", 1.333)

        MAX_BONE_PX = base_size * 0.5

        px_lms = []
        for lm in lms:
            nx, ny = lm[0], lm[1]
            dx_norm = nx - anchor_norm_x
            dy_norm = ny - anchor_norm_y
            
            dx_px = clamp(dx_norm * skel_scale_x * SKELETON_SCALE, -MAX_BONE_PX, MAX_BONE_PX)
            dy_px = clamp(dy_norm * skel_scale_y * SKELETON_SCALE, -MAX_BONE_PX, MAX_BONE_PX)

            sx = anchor_screen_x - dx_px
            sy = anchor_screen_y + dy_px
            
            px_lms.append(QPointF(sx, sy))
            

        # ── Connection lines — two-pass "glove" rendering ─────────────────
        # Pass 1: thick black stroke  →  the exterior glove surface
        pen_blk = QPen(BLACK_OUT, 8.0, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin)
        painter.setPen(pen_blk)
        for a, b in HAND_CONNECTIONS:
            if a < len(px_lms) and b < len(px_lms):
                painter.drawLine(px_lms[a], px_lms[b])

        # Pass 2: thin white stroke  →  the white bone inside the glove
        pen_wht = QPen(WHITE_CORE, 2.5, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin)
        painter.setPen(pen_wht)
        for a, b in HAND_CONNECTIONS:
            if a < len(px_lms) and b < len(px_lms):
                painter.drawLine(px_lms[a], px_lms[b])

        # ── Landmark dots — black outline ring then white core ────────────
        painter.setPen(Qt.NoPen)

        for i, pos in enumerate(px_lms):
            radius = 9 if i in _FINGERTIPS else 6

            # Soft outer glow halo
            grad = QRadialGradient(pos, radius * 2)
            grad.setColorAt(0.0, WHITE_GLOW)
            grad.setColorAt(1.0, QColor(0, 0, 0, 0))
            painter.setBrush(QBrush(grad))
            painter.drawEllipse(pos, radius * 2, radius * 2)

            # Black glove ring (slightly larger than white core)
            painter.setBrush(QBrush(BLACK_OUT))
            painter.drawEllipse(pos, radius + 3, radius + 3)

            # White bone core
            painter.setBrush(QBrush(WHITE_CORE))
            painter.drawEllipse(pos, radius, radius)

        # ── Dwell charging arc on index fingertip (LM 8) ─────────────────
        if len(lms) >= 9:
            tip_pos = QPointF(anchor_screen_x, anchor_screen_y)
            arc_r   = _DWELL_ARC_RADIUS
            rect_x  = int(tip_pos.x() - arc_r)
            rect_y  = int(tip_pos.y() - arc_r)
            rect_d  = arc_r * 2
            pen_w   = _DWELL_PEN_WIDTH

            # Track ring — clean green circle showing the dwell path
            track_alpha = int(120 * alpha)
            if track_alpha > 0:
                track_col = QColor(_DWELL_TRACK)
                track_col.setAlpha(track_alpha)
                painter.setPen(QPen(track_col, pen_w, Qt.SolidLine, Qt.RoundCap))
                painter.setBrush(Qt.NoBrush)
                painter.drawEllipse(rect_x, rect_y, rect_d, rect_d)

            # Pulsing glow ring when dwell is actively charging
            if self._dwell_progress > 0.01:
                pulse = 0.5 + 0.5 * math.sin(time.time() * 6.0)  # ~1Hz breathing
                glow_alpha = int((90 + 50 * pulse) * alpha)
                glow_col = QColor(57, 255, 20, glow_alpha)
                glow_pen = QPen(glow_col, pen_w, Qt.SolidLine, Qt.RoundCap)
                painter.setPen(glow_pen)
                painter.setBrush(Qt.NoBrush)
                painter.drawEllipse(rect_x, rect_y, rect_d, rect_d)

            # Progress arc — vivid lime-green, sweeps clockwise from 12 o'clock
            if self._dwell_progress > 0.001:
                span_deg  = self._dwell_progress * 360.0
                red_val   = int(40  + 40  * self._dwell_progress)
                blue_val  = int(10  + 30  * self._dwell_progress)
                arc_alpha = int((0.70 + 0.30 * self._dwell_progress) * 255 * alpha)
                arc_col   = QColor(red_val, 255, blue_val, arc_alpha)
                
                start_angle = 90 * 16
                span_angle  = -int(span_deg * 16)
                
                # Draw black border behind the arc for contrast
                border_pen_w = _DWELL_BORDER_WIDTH
                border_col = QColor(0, 0, 0, int(210 * alpha))
                painter.setPen(QPen(border_col, border_pen_w, Qt.SolidLine, Qt.RoundCap))
                painter.setBrush(Qt.NoBrush)
                painter.drawArc(rect_x, rect_y, rect_d, rect_d, start_angle, span_angle)

                painter.setPen(QPen(arc_col, pen_w, Qt.SolidLine, Qt.RoundCap))
                painter.setBrush(Qt.NoBrush)
                painter.drawArc(rect_x, rect_y, rect_d, rect_d, start_angle, span_angle)

        # ── CLICK burst ring ──────────────────────────────────────────
        if self._click_pulse > 0.01 and len(lms) >= 9:
            tip_pos = QPointF(anchor_screen_x, anchor_screen_y)
            pulse_r     = 38 + (1.0 - self._click_pulse) * 50
            ring_alpha  = int(self._click_pulse * 220 * alpha)
            # Click burst ring matches the dwell green for visual coherence
            ring_color  = QColor(57, 255, 20, ring_alpha)
            painter.setPen(QPen(ring_color, 7.0))
            painter.setBrush(Qt.NoBrush)
            painter.drawEllipse(tip_pos, pulse_r, pulse_r)



    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def closeEvent(self, event) -> None:  # noqa: N802
        super().closeEvent(event)


# ===========================================================================
# ── Entry point ───────────────────────────────────────────────────────────
# ===========================================================================

def main() -> None:
    # ── Systemd / headless-boot environment guards ────────────────────────────
    # These are no-ops when running from a normal desktop terminal, but are
    # essential when systemd launches the process with a bare environment.
    if sys.platform != "win32":
        # Software rendering fallback — avoids crashes when the GPU driver
        # isn't fully initialised at the point the service starts.
        os.environ.setdefault("QT_XCB_NATIVE_PAINTING", "1")
        # libGL software fallback (Mesa llvmpipe) — prevents a blank window
        # on Pi if the vc4/v3d DRM driver hasn't claimed the display yet.
        os.environ.setdefault("LIBGL_ALWAYS_SOFTWARE", "0")
        # Suppress Qt's "could not connect to display" noise before xcb init
        os.environ.setdefault("QT_LOGGING_RULES", "qt.qpa.*=false")

    app = QApplication(sys.argv)
    app.setApplicationName("GestureOverlay")

    window = OverlayWindow()
    # NOTE: We do NOT set Qt.BlankCursor here anymore.
    # Allowing the hardware cursor to show means the user can see it naturally
    # change into a pointer/hand when they hover over a button in the browser.
    window.show()

    sys.exit(app.exec_())


if __name__ == "__main__":
    main()