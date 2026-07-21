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
    QColor, QPainter, QPen, QBrush, QFont, QRadialGradient, QImage,
)
from PyQt5.QtWidgets import QApplication, QWidget

import websockets

# ===========================================================================
# ── Configuration ─────────────────────────────────────────────────────────
# ===========================================================================

WS_URI        = "ws://127.0.0.1:8765"
RECONNECT_SEC = 0.5       # seconds between reconnect attempts
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

# Skeleton colours are computed dynamically based on background brightness
# (see _get_adaptive_skeleton_colors). These fallbacks are used before the
# first background sample is available.
_DOT_COLOR_DARK  = QColor(240, 245, 255, 240)   # near-white core for dark BG
_GLOW_COLOR_DARK = QColor(200, 215, 255,  55)   # outer glow for dark BG
_LINE_COLOR_DARK = QColor(210, 225, 255,  85)   # skeleton lines for dark BG

_DOT_COLOR_LIGHT  = QColor( 30,  30,  50, 230)  # near-black core for light BG
_GLOW_COLOR_LIGHT = QColor( 20,  20,  60,  70)  # outer glow for light BG
_LINE_COLOR_LIGHT = QColor( 10,  10,  40,  90)  # skeleton lines for light BG

# ── Vivid lime-green for the dwell charging arc ────────────────────────────
# #39FF14 / hue 107°  — highest attention-grab in the green family
_DWELL_GREEN_BASE = QColor( 57, 255,  20)   # vivid neon green
_DWELL_TRACK      = QColor( 57, 255,  20,  45)  # faint track ring

# Finger tip indices — drawn 2 px larger, same colour
_FINGERTIPS = {4, 8, 12, 16, 20}


# ===========================================================================
# ── WebSocket client thread ───────────────────────────────────────────────
# ===========================================================================

class _WSThread(QThread):
    """
    Runs an asyncio WebSocket client on a background QThread.
    Emits data_received(dict) on every valid message.
    Emits connection_changed(bool) when the connection state changes.
    """

    data_received       = pyqtSignal(dict)
    connection_changed  = pyqtSignal(bool)

    def __init__(self) -> None:
        super().__init__()
        self._stop = threading.Event()
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._ws = None  # currently-active connection, if any

    def stop(self) -> None:
        self._stop.set()

    def send_geometry(self, w: int, h: int) -> None:
        """
        Push the Qt-verified screen geometry to gesture_engine.py.

        This is the authoritative geometry path: the engine's own OS-level
        detection (ctypes.windll) only works on Windows and is a no-op on
        this Raspberry Pi/Linux deployment, so it relies entirely on us for
        correct width/height. Safe to call from the Qt (main) thread — hands
        off to the background asyncio loop via run_coroutine_threadsafe.
        """
        if self._loop is None:
            return
        payload = json.dumps({"type": "geometry", "width": int(w), "height": int(h)})
        try:
            asyncio.run_coroutine_threadsafe(self._send(payload), self._loop)
        except RuntimeError:
            pass  # loop not running yet / already closed

    async def _send(self, payload: str) -> None:
        if self._ws is not None:
            try:
                await self._ws.send(payload)
            except Exception:
                pass

    def run(self) -> None:
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        try:
            self._loop.run_until_complete(self._main())
        finally:
            self._loop.close()

    async def _main(self) -> None:
        while not self._stop.is_set():
            try:
                async with websockets.connect(
                    WS_URI,
                    open_timeout=3,
                    ping_interval=5,
                    ping_timeout=10,
                ) as ws:
                    self._ws = ws
                    self.connection_changed.emit(True)
                    async for raw in ws:
                        if self._stop.is_set():
                            return
                        try:
                            self.data_received.emit(json.loads(raw))
                        except json.JSONDecodeError:
                            pass
            except Exception:
                pass  # reconnect silently
            finally:
                self._ws = None

            self.connection_changed.emit(False)

            if not self._stop.is_set():
                await asyncio.sleep(RECONNECT_SEC)


# ===========================================================================
# ── Overlay window ────────────────────────────────────────────────────────
# ===========================================================================

class OverlayWindow(QWidget):
    """
    Fullscreen transparent window that paints the hand skeleton.

    The window sits above the kiosk application and passes all mouse/touch
    events through so it never blocks interaction with the underlying app.
    """

    def __init__(self) -> None:
        super().__init__()

        # ── Window setup ──────────────────────────────────────────────────
        self.setWindowFlags(
            Qt.FramelessWindowHint          |
            Qt.WindowStaysOnTopHint         |
            Qt.X11BypassWindowManagerHint   |  # Bypasses Xorg WM entirely so it never steals focus
            Qt.Tool                         |  # required for WindowTransparentForInput
            Qt.WindowTransparentForInput       # X11 input region set to empty
        )
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
        self._connected:      bool              = False
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

        # Skeleton stabilization — anchor smoothing + bone-length consistency
        self._prev_anchor_x:  float = -1.0
        self._prev_anchor_y:  float = -1.0
        self._prev_bone_lengths: dict = {}  # (a,b) -> length in pixels


        # ── WebSocket ─────────────────────────────────────────────────────
        self._ws = _WSThread()
        self._ws.data_received.connect(self._on_data)
        self._ws.connection_changed.connect(self._on_connect)
        self._ws.start()

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
        if connected:
            # The engine starts with a hardcoded placeholder and cannot
            # reliably detect the real screen itself on this platform — send
            # it the true geometry the instant we (re)connect.
            geo = self._screen.geometry()
            self._ws.send_geometry(geo.width(), geo.height())

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
        self._ws.send_geometry(geo.width(), geo.height())

    def _on_data(self, data: dict) -> None:
        prev_state            = self._cursor_state
        
        new_lms = data.get("landmarks", [])
        new_x = data.get("x", 0)
        new_y = data.get("y", 0)
        new_dwell = data.get("dwell_progress", 0.0)
        new_detect = data.get("hand_detected", False)
        
        if (self._landmarks != new_lms or 
            self._cursor_x != new_x or 
            self._cursor_y != new_y or 
            self._dwell_progress != new_dwell or
            self._hand_detected != new_detect):
            self._new_data = True
            
        self._hand_detected   = new_detect
        self._landmarks       = new_lms
        self._gesture         = data.get("gesture", "NONE")
        self._cursor_state    = data.get("state", "MOVE")
        self._cursor_x        = new_x
        self._cursor_y        = new_y
        self._dwell_progress  = new_dwell
        self._depth_z         = data.get("depth_z", 1.0)
        # Receive the complete mapping contract from the engine.
        # scale_x / scale_y already embed both display resolution and active crop,
        # so the overlay applies them directly without any geometry computation.
        self._engine_scale_x  = data.get("scale_x",   self._engine_scale_x)
        self._engine_scale_y  = data.get("scale_y",   self._engine_scale_y)
        self._engine_rotation = data.get("rotation",  getattr(self, "_engine_rotation", 0))
        self._cam_aspect      = data.get("cam_aspect", getattr(self, "_cam_aspect", 1.333))
        # These now just echo back the geometry WE already pushed to the engine,
        # but with one critical exception: XWayland lies about screen dimensions
        # on rotated Raspberry Pi displays, reporting landscape (e.g. 1920x1080).
        # The engine detects this and forces portrait (1080x1920). If the engine
        # corrects our geometry to portrait, we must force-resize our window so 
        # the transparent overlay isn't cropped.
        self._engine_w = data.get("screen_width", data.get("display_w", self._engine_w))
        self._engine_h = data.get("screen_height", data.get("display_h", self._engine_h))
        
        if abs(self._engine_rotation) == 90 and self.width() > self.height():
            # Force window to portrait if it was erroneously created as landscape
            self.setGeometry(0, 0, min(self.width(), self.height()), max(self.width(), self.height()))

        # Trigger burst animation the moment a click fires
        if self._cursor_state == "CLICK" and prev_state != "CLICK":
            self._click_pulse    = 1.0
            self._ghost_protect  = 20  # ~320 ms guard — covers xdotool disruption window

        pass

    # ------------------------------------------------------------------
    # Animation tick
    # ------------------------------------------------------------------

    def _tick(self) -> None:

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

        # Only update if we're animating, or if we have new data AND we're actually visible
        if animating or (self._new_data and self._fade_alpha >= 0.01):
            self.update()  # schedule repaint
            
        self._new_data = False

    # ------------------------------------------------------------------
    # Painting
    # ------------------------------------------------------------------

    def paintEvent(self, _event) -> None:  # noqa: N802
        if self._fade_alpha < 0.01 and self._connected:
            # Nothing visible and engine is connected — skip all drawing
            return

        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        if self._fade_alpha >= 0.01 and len(self._landmarks) == 21:
            self._draw_skeleton(painter, self.width(), self.height())

        if not self._connected:
            self._draw_status(painter)

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
        # Anchor at Index Fingertip (LM 8)
        anchor_norm_x, anchor_norm_y = lms[8][0], lms[8][1]
        
        # Lock the drawing anchor exactly to the OS cursor coordinates received from gesture_engine.
        # This prevents the drawing and the real cursor from drifting apart if there is a mismatch
        # between the hardcoded resolution in gesture_engine and the actual Qt window size.
        anchor_screen_x = float(self._cursor_x)
        anchor_screen_y = float(self._cursor_y)

        # Smooth the anchor point to prevent micro-jitter from propagating
        # to every landmark in the skeleton (all are rendered relative to anchor).
        if self._prev_anchor_x >= 0:
            anchor_screen_x = 0.85 * anchor_screen_x + 0.15 * self._prev_anchor_x
            anchor_screen_y = 0.85 * anchor_screen_y + 0.15 * self._prev_anchor_y
        self._prev_anchor_x = anchor_screen_x
        self._prev_anchor_y = anchor_screen_y
        
        # Auto-scale skeleton: bigger when far, smaller when close.
        # Power > 1.0 over-compensates the natural landmark compression at distance,
        # making the skeleton actively grow when the hand moves away from the camera.
        # depth_z=0.5 (far) → scale ≈ 2.5x  (bigger for visibility)
        # depth_z=1.0 (cal) → scale = 1.0x   (baseline)
        # depth_z=2.0 (close)→ scale ≈ 0.4x  (smaller, less intrusive)
        TARGET_SKEL_SIZE = 0.8   # Tune this to control overall hand size
        SKELETON_SCALE = TARGET_SKEL_SIZE  # Disable dynamic depth scaling to prevent rubber-banding

        def clamp(v: float, lo: float, hi: float) -> float:
            return max(lo, min(hi, v))

        # Presentation scale — NOT the cursor's interaction scale (scale_x/scale_y).
        # Deliberately separate: this controls how big the hand LOOKS, independent
        # of how sensitive the cursor is. Do not merge these two scales.
        base_size = min(self._engine_w, self._engine_h)
        cam_aspect = getattr(self, "_cam_aspect", 1.333)
        rotation = getattr(self, "_engine_rotation", 0)
        
        if abs(rotation) == 90:
            skel_scale_x = base_size / cam_aspect
            skel_scale_y = base_size
        else:
            skel_scale_x = base_size
            skel_scale_y = base_size / cam_aspect

        MAX_BONE_PX = base_size * 0.5  # defensive cap against a single bad frame

        px_lms = []
        for nx, ny in lms:
            dx_norm = nx - anchor_norm_x
            dy_norm = ny - anchor_norm_y
            
            dx_px = clamp(dx_norm * skel_scale_x * SKELETON_SCALE, -MAX_BONE_PX, MAX_BONE_PX)
            dy_px = clamp(dy_norm * skel_scale_y * SKELETON_SCALE, -MAX_BONE_PX, MAX_BONE_PX)

            sx = anchor_screen_x - dx_px
            sy = anchor_screen_y + dy_px
            
            px_lms.append(QPointF(sx, sy))

        # ── Bone-length consistency clamp ─────────────────────────────────
        # If any bone stretches beyond 1.5× its previous length in a single
        # frame, interpolate the child endpoint back toward the parent to
        # prevent visual stretching from drifted landmark data.
        if self._prev_bone_lengths:
            for a, b in HAND_CONNECTIONS:
                if a < len(px_lms) and b < len(px_lms):
                    bone_len = ((px_lms[b].x() - px_lms[a].x()) ** 2 +
                                (px_lms[b].y() - px_lms[a].y()) ** 2) ** 0.5
                    key = (a, b)
                    prev_len = self._prev_bone_lengths.get(key, bone_len)
                    if prev_len > 1.0 and bone_len > prev_len * 1.5:
                        # Clamp: move endpoint b toward a so bone = prev_len * 1.5
                        ratio = (prev_len * 1.5) / max(bone_len, 0.001)
                        new_x = px_lms[a].x() + (px_lms[b].x() - px_lms[a].x()) * ratio
                        new_y = px_lms[a].y() + (px_lms[b].y() - px_lms[a].y()) * ratio
                        px_lms[b] = QPointF(new_x, new_y)

        # Update bone-length cache for next frame
        new_bone_lengths = {}
        for a, b in HAND_CONNECTIONS:
            if a < len(px_lms) and b < len(px_lms):
                bl = ((px_lms[b].x() - px_lms[a].x()) ** 2 +
                      (px_lms[b].y() - px_lms[a].y()) ** 2) ** 0.5
                new_bone_lengths[(a, b)] = bl
        self._prev_bone_lengths = new_bone_lengths

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

            # Soft outer glow halo (Solid circle now)
            painter.setBrush(QBrush(WHITE_GLOW))
            painter.drawEllipse(pos, radius * 2, radius * 2)

            # Black glove ring (slightly larger than white core)
            painter.setBrush(QBrush(BLACK_OUT))
            painter.drawEllipse(pos, radius + 3, radius + 3)

            # White bone core
            painter.setBrush(QBrush(WHITE_CORE))
            painter.drawEllipse(pos, radius, radius)

        # ── Dwell charging arc on index fingertip (LM 8) ─────────────────
        if len(lms) >= 9:
            tip_pos = QPointF(self._cursor_x, self._cursor_y)
            arc_r   = 38                     # radius of the progress ring
            rect_x  = int(tip_pos.x() - arc_r)
            rect_y  = int(tip_pos.y() - arc_r)
            rect_d  = arc_r * 2

            # Track ring — faint green circle showing the dwell path
            track_alpha = int(55 * alpha)
            if track_alpha > 0:
                track_col = QColor(_DWELL_TRACK)
                track_col.setAlpha(track_alpha)
                painter.setPen(QPen(track_col, 8.0, Qt.SolidLine, Qt.RoundCap))
                painter.setBrush(Qt.NoBrush)
                painter.drawEllipse(int(tip_pos.x() - arc_r), int(tip_pos.y() - arc_r),
                                    arc_r * 2, arc_r * 2)

            # Progress arc — vivid lime-green, sweeps clockwise from 12 o'clock
            if self._dwell_progress > 0.001:
                span_deg  = self._dwell_progress * 360.0
                # Green channel stays maxed; red/blue components pulse slightly
                # to give a subtle brightness surge as the arc completes
                red_val   = int(40  + 40  * self._dwell_progress)   # 40 → 80
                blue_val  = int(10  + 30  * self._dwell_progress)   # 10 → 40
                arc_alpha = int((0.55 + 0.45 * self._dwell_progress) * 255 * alpha)
                arc_col   = QColor(red_val, 255, blue_val, arc_alpha)
                pen_w     = 10.0 + self._dwell_progress * 4.0       # thickens as it fills
                
                start_angle = 90 * 16
                span_angle  = -int(span_deg * 16)
                
                # Draw black border behind the arc for contrast
                border_pen_w = pen_w + 4.0
                border_col = QColor(0, 0, 0, int(210 * alpha))
                painter.setPen(QPen(border_col, border_pen_w, Qt.SolidLine, Qt.RoundCap))
                painter.setBrush(Qt.NoBrush)
                painter.drawArc(rect_x, rect_y, rect_d, rect_d, start_angle, span_angle)

                painter.setPen(QPen(arc_col, pen_w, Qt.SolidLine, Qt.RoundCap))
                painter.setBrush(Qt.NoBrush)
                # Qt angles: 0° = 3 o'clock, positive = counter-clockwise in 1/16°
                # Start at 12 o'clock (90°), sweep clockwise (− span)
                painter.drawArc(rect_x, rect_y, rect_d, rect_d, start_angle, span_angle)

        # ── CLICK burst ring ──────────────────────────────────────────
        if self._click_pulse > 0.01 and len(lms) >= 9:
            tip_pos = QPointF(self._cursor_x, self._cursor_y)
            pulse_r     = 38 + (1.0 - self._click_pulse) * 50
            ring_alpha  = int(self._click_pulse * 220 * alpha)
            # Click burst ring matches the dwell green for visual coherence
            ring_color  = QColor(57, 255, 20, ring_alpha)
            painter.setPen(QPen(ring_color, 7.0))
            painter.setBrush(Qt.NoBrush)
            painter.drawEllipse(tip_pos, pulse_r, pulse_r)



    def _draw_status(self, painter: Optional[QPainter] = None) -> None:
        """Small 'Connecting…' pill when gesture engine is not reachable."""
        own = painter is None
        if own:
            painter = QPainter(self)
            painter.setRenderHint(QPainter.Antialiasing)

        msg  = "● Connecting to gesture engine…"
        font = QFont("Sans Serif", 12)
        font.setWeight(QFont.Medium)
        painter.setFont(font)

        fm      = painter.fontMetrics()
        text_w  = fm.horizontalAdvance(msg)
        text_h  = fm.height()
        padding = 12
        pill_w  = text_w + padding * 2
        pill_h  = text_h + padding
        x       = (self.width() - pill_w) // 2
        y       = self.height() - 80 - pill_h

        # Background pill
        bg = QColor(20, 20, 30, 190)
        painter.setBrush(QBrush(bg))
        painter.setPen(Qt.NoPen)
        painter.drawRoundedRect(x, y, pill_w, pill_h, pill_h // 2, pill_h // 2)

        # Text
        painter.setPen(QColor(160, 160, 180, 220))
        painter.drawText(x + padding, y + padding // 2 + fm.ascent(), msg)

        if own:
            painter.end()

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def closeEvent(self, event) -> None:  # noqa: N802
        self._ws.stop()
        self._ws.quit()
        self._ws.wait(2000)
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