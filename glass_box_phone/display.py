"""
CASCADE-LATTICE Universal Game Dashboard
=========================================
Rich visualization that adapts to ANY gymnasium game.

Based on sovereign_lattice_eval.py but universal for:
- Box2D (LunarLander, BipedalWalker, etc.)
- Atari (Pong, Breakout, SpaceInvaders, etc.)
- Classic Control (CartPole, MountainCar, Acrobot, etc.)

Shows:
- Game frame with adaptive sizing
- Decision matrix with action probabilities
- Provenance chain (Merkle roots)
- State vector display (when interpretable)
- Session statistics
- Control hints
"""

import pygame
import numpy as np
import math
import time
import threading
import os
import signal
import ctypes
from typing import Optional, Dict, Any, List
from collections import deque


def _check_x11_display_alive() -> bool:
    """
    Check if X11 display is still alive by attempting a safe X11 operation.
    Returns True if display is accessible, False otherwise.
    """
    display = os.environ.get('DISPLAY', ':0')
    try:
        # Try to import and use Xlib for a health check
        # This is faster than reinitializing pygame
        import subprocess
        result = subprocess.run(
            ['xdpyinfo', '-display', display],
            capture_output=True,
            timeout=2
        )
        return result.returncode == 0
    except Exception:
        # If xdpyinfo isn't available or times out, try a pygame check
        return pygame.display.get_init()


def _safe_pygame_operation(func, *args, **kwargs):
    """
    Wrap pygame operations to catch segfaults before they crash Python.
    Returns (success, result) tuple.
    """
    try:
        # Check display is alive before any pygame operation
        if not pygame.display.get_init():
            return False, None
        result = func(*args, **kwargs)
        return True, result
    except pygame.error as e:
        print(f"[DISPLAY] Pygame error: {e}")
        return False, None
    except Exception as e:
        print(f"[DISPLAY] Unexpected error: {e}")
        return False, None


# ═══════════════════════════════════════════════════════════════════════════
# ACTION LABELS PER GAME (expandable)
# ═══════════════════════════════════════════════════════════════════════════

ACTION_LABELS = {
    # Box2D
    "LunarLander": ["NOOP", "LEFT ENGINE", "MAIN ENGINE", "RIGHT ENGINE"],
    "BipedalWalker": ["Hip1", "Knee1", "Hip2", "Knee2"],
    "CarRacing": ["Steer", "Gas", "Brake"],
    
    # Classic Control
    "CartPole": ["← LEFT", "→ RIGHT"],
    "MountainCar": ["← LEFT", "NOOP", "→ RIGHT"],
    "Acrobot": ["−1 TORQUE", "0 NOOP", "+1 TORQUE"],
    "Pendulum": ["Torque"],
    
    # Atari
    "Pong": ["NOOP", "FIRE", "RIGHT", "LEFT", "RIGHT+FIRE", "LEFT+FIRE"],
    "Breakout": ["NOOP", "FIRE", "RIGHT", "LEFT"],
    "SpaceInvaders": ["NOOP", "FIRE", "RIGHT", "LEFT", "RIGHT+FIRE", "LEFT+FIRE"],
    "Asteroids": ["NOOP", "FIRE", "UP", "RIGHT", "LEFT", "DOWN", 
                  "UP+RIGHT", "UP+LEFT", "DOWN+RIGHT", "DOWN+LEFT",
                  "UP+FIRE", "RIGHT+FIRE", "LEFT+FIRE", "DOWN+FIRE"],
    "Qbert": ["NOOP", "FIRE", "UP", "RIGHT", "LEFT", "DOWN"],
    "Seaquest": ["NOOP", "FIRE", "UP", "RIGHT", "LEFT", "DOWN",
                 "UP+RIGHT", "UP+LEFT", "DOWN+RIGHT", "DOWN+LEFT",
                 "UP+FIRE", "RIGHT+FIRE", "LEFT+FIRE", "DOWN+FIRE",
                 "UP+RIGHT+FIRE", "UP+LEFT+FIRE"],
    "BeamRider": ["NOOP", "FIRE", "UP", "RIGHT", "LEFT", "DOWN",
                  "UP+RIGHT", "UP+LEFT", "RIGHT+FIRE", "LEFT+FIRE"],
    "Enduro": ["NOOP", "FIRE", "RIGHT", "LEFT", "DOWN", 
               "DOWN+RIGHT", "DOWN+LEFT", "RIGHT+FIRE", "LEFT+FIRE"],
    "MsPacman": ["NOOP", "UP", "RIGHT", "LEFT", "DOWN",
                 "UP+RIGHT", "UP+LEFT", "DOWN+RIGHT", "DOWN+LEFT"],
}

# State labels for interpretable observation display
STATE_LABELS = {
    "LunarLander": ["X-Pos", "Y-Pos", "X-Vel", "Y-Vel", "Angle", "Ang-Vel", "L-Leg", "R-Leg"],
    "CartPole": ["Cart Pos", "Cart Vel", "Pole Angle", "Pole Vel"],
    "MountainCar": ["Position", "Velocity"],
    "Acrobot": ["cos(θ1)", "sin(θ1)", "cos(θ2)", "sin(θ2)", "θ1-dot", "θ2-dot"],
    "Pendulum": ["cos(θ)", "sin(θ)", "θ-dot"],
}


def get_action_labels(game_name: str, num_actions: int) -> List[str]:
    """Get action labels for a game, with fallback to generic labels."""
    for key, labels in ACTION_LABELS.items():
        if key.lower() in game_name.lower():
            if len(labels) >= num_actions:
                return labels[:num_actions]
    return [f"Action {i}" for i in range(num_actions)]


def get_state_labels(game_name: str, obs_size: int) -> List[str]:
    """Get state labels for interpretable display."""
    for key, labels in STATE_LABELS.items():
        if key.lower() in game_name.lower():
            if len(labels) == obs_size:
                return labels
    return [f"s[{i}]" for i in range(min(obs_size, 16))]


# ═══════════════════════════════════════════════════════════════════════════
# INFORMATIONAL WEALTH TRACKER
# ═══════════════════════════════════════════════════════════════════════════

class InformationalWealth:
    """Tracks decision history and provenance for rich visualization."""
    
    def __init__(self, max_history: int = 30):
        self.action_history = deque(maxlen=max_history)
        self.reward_history = deque(maxlen=max_history)
        self.value_history = deque(maxlen=max_history)
        self.entropy_history = deque(maxlen=max_history)
        self.merkle_chain = deque(maxlen=max_history)
        self.override_points = []
        self.total_overrides = 0
        self.total_bits_earned = 0
        
    def record(self, action: int, reward: float, probs: np.ndarray, 
               value: float, merkle: str, was_override: bool = False):
        """Record a decision point."""
        self.action_history.append(action)
        self.reward_history.append(reward)
        self.value_history.append(value)
        self.merkle_chain.append(merkle[:16] if merkle else "none")
        
        if probs is not None and len(probs) > 0:
            probs_safe = np.clip(probs, 1e-8, 1.0)
            entropy = -np.sum(probs_safe * np.log(probs_safe))
            self.entropy_history.append(entropy)
        
        if was_override:
            self.override_points.append(len(self.action_history) - 1)
            self.total_overrides += 1
    
    def add_bits(self, bits: int):
        self.total_bits_earned += bits
    
    def get_cumulative_reward(self) -> float:
        return sum(self.reward_history)
    
    def get_avg_entropy(self) -> float:
        return np.mean(list(self.entropy_history)) if self.entropy_history else 0
    
    def get_avg_value(self) -> float:
        return np.mean(list(self.value_history)) if self.value_history else 0
    
    def get_action_sequence(self, labels: List[str], n: int = 8) -> str:
        """Get last n actions as a readable sequence."""
        seq = list(self.action_history)[-n:]
        return " → ".join([labels[a][:6] if a < len(labels) else f"A{a}" for a in seq])


# ═══════════════════════════════════════════════════════════════════════════
# UNIVERSAL GAME DASHBOARD
# ═══════════════════════════════════════════════════════════════════════════

class GameDisplay:
    """
    CASCADE-LATTICE Universal Dashboard
    
    Rich visualization that adapts to any gymnasium game.
    """
    
    # Window size - optimized for OBS capture
    WIN_W, WIN_H = 1400, 920

    # Defaults are replaced per instance so the 1200x780 Cocoon phone window
    # does not inherit stale 1400x920 panel math.
    GAME_W, GAME_H = 850, 640
    PANEL_W = WIN_W - GAME_W - 20
    
    def __init__(
        self,
        width: int = 1400,
        height: int = 920,
        title: str = "Cocoon Glass Authority",
        target_fps: int = 30,
    ):
        self.width = max(960, int(width))
        self.height = max(640, int(height))
        self.title = title
        self.target_fps = target_fps
        self._fullscreen = False
        self._windowed_size = (self.width, self.height)

        # Instance layout. Most draw routines use self.GAME_* and self.PANEL_W,
        # so this keeps the existing surface API while making it responsive.
        self._set_layout(self.width, self.height)

        self._theme = {
            "bg": (5, 7, 12),
            "grid": (16, 24, 30),
            "panel": (9, 15, 21),
            "panel_hi": (13, 24, 31),
            "line": (40, 68, 76),
            "teal": (61, 222, 180),
            "cyan": (93, 192, 245),
            "amber": (235, 177, 74),
            "green": (110, 226, 142),
            "red": (232, 96, 76),
            "text": (224, 235, 232),
            "muted": (130, 151, 158),
        }
        
        self._screen: Optional[pygame.Surface] = None
        self._clock: Optional[pygame.time.Clock] = None
        self._running = False
        self._frame_lock = threading.Lock()
        self._menu_lock = threading.Lock()  # Thread-safe menu state access
        
        # Current state
        self._current_frame: Optional[np.ndarray] = None
        self._current_obs: Optional[np.ndarray] = None
        self._game_name = "Loading..."
        self._game_display_name = "Loading..."
        self._num_actions = 4
        self._action_labels: List[str] = ["A0", "A1", "A2", "A3"]
        self._state_labels: List[str] = []
        
        # Decision data
        self._action_probs: Optional[np.ndarray] = None
        self._chosen_action: int = 0
        self._value_estimate: float = 0.0
        self._merkle_root: str = ""
        self._parent_merkle: str = ""
        self._genesis_root: str = ""
        
        # Control state
        self._hold_mode: Optional[str] = None
        self._viewer_name: Optional[str] = None
        self._viewer_timer: float = 0.0
        self._viewer_actions: int = 0
        self._control_mode = "AI"
        self._ai_action = ""
        
        # Action feedback for viewer
        self._last_action_text: str = ""
        self._last_action_time: float = 0.0
        self._action_feedback_duration: float = 0.8  # How long to show action feedback
        
        # Queue
        self._queue: List[Dict] = []
        self._cocoon_overlay: Optional[Dict[str, Any]] = None
        self._operational_surface: Optional[Dict[str, Any]] = None
        self._glass_view_mode = "ops"
        
        # Game selection menu
        self._menu_state: Optional[Dict] = None  # Menu data when open
        self._button_zones: List[Dict[str, Any]] = []
        
        # Stats
        self._episode_reward: float = 0.0
        self._episode_steps: int = 0
        self._total_episodes: int = 0
        self._session_start: float = time.time()
        
        # Wealth tracker
        self.wealth = InformationalWealth()
        
        # Fonts
        self._fonts: Dict[str, pygame.font.Font] = {}

    def _set_layout(self, width: int, height: int):
        self.width = max(960, int(width))
        self.height = max(640, int(height))
        self._top_bar_h = 56
        self._panel_gap = 10
        self._bottom_h = 126
        desired_panel = int(self.width * 0.34)
        self.PANEL_W = max(330, min(440, desired_panel))
        self.GAME_W = max(520, self.width - self.PANEL_W - self._panel_gap)
        self.PANEL_W = max(300, self.width - self.GAME_W - self._panel_gap)
        self.GAME_H = max(390, self.height - self._top_bar_h - self._bottom_h - 6)
        self._game_y = self._top_bar_h
    
    def start(self):
        """Initialize pygame display. Safe to call multiple times."""
        # Allow restart if pygame/display was torn down.
        if self._screen is not None and pygame.display.get_init():
            return  # Already started and display is alive
        if self._screen is not None and not pygame.display.get_init():
            # Stale surface after a pygame quit/crash; allow re-init.
            self._screen = None
            self._clock = None
        
        import os
        # Position window at top-left corner (0,0) for ffmpeg capture
        os.environ['SDL_VIDEO_WINDOW_POS'] = '0,0'
        # Force software rendering to avoid GL context threading issues
        # But respect SDL_VIDEODRIVER=dummy if set for headless testing
        if os.environ.get('SDL_VIDEODRIVER') != 'dummy':
            os.environ['SDL_VIDEODRIVER'] = 'x11'
        os.environ['SDL_RENDER_DRIVER'] = 'software'
        
        pygame.init()
        pygame.display.set_caption(self.title)
        
        # Use software surface (no OpenGL) to avoid threading issues
        self._screen = pygame.display.set_mode((self.width, self.height), pygame.SWSURFACE)
        self._clock = pygame.time.Clock()
        self._running = True
        
        font_family = "DejaVu Sans Mono,Consolas,monospace"
        self._fonts = {
            'xs': pygame.font.SysFont(font_family, 10),
            'sm': pygame.font.SysFont(font_family, 12),
            'md': pygame.font.SysFont(font_family, 15),
            'lg': pygame.font.SysFont(font_family, 24, bold=True),
            'title': pygame.font.SysFont(font_family, 20, bold=True),
            'huge': pygame.font.SysFont(font_family, 38, bold=True),
        }
        
        # Render initial frame so ffmpeg sees something (not X11 background)
        self._screen.fill(self._theme["bg"])
        title = self._fonts['huge'].render("COCOON GLASS AUTHORITY", True, self._theme["teal"])
        subtitle = self._fonts['md'].render("Facility fabric booting through pygame surface", True, self._theme["muted"])
        self._screen.blit(title, (self.width//2 - title.get_width()//2, self.height//2 - 46))
        self._screen.blit(subtitle, (self.width//2 - subtitle.get_width()//2, self.height//2 + 10))
        pygame.display.flip()
        
        print(f"[DISPLAY] Game window opened: {self.width}x{self.height}")

    def toggle_fullscreen(self):
        """Toggle the pygame rendering surface between windowed and fullscreen."""
        if not self._screen or not pygame.display.get_init():
            return
        try:
            if self._fullscreen:
                new_w, new_h = self._windowed_size
                self._screen = pygame.display.set_mode((new_w, new_h), pygame.SWSURFACE)
                self._fullscreen = False
                self._set_layout(new_w, new_h)
                print(f"[DISPLAY] Fullscreen off: {new_w}x{new_h}")
            else:
                self._windowed_size = (self.width, self.height)
                self._screen = pygame.display.set_mode((0, 0), pygame.FULLSCREEN | pygame.SWSURFACE)
                new_w, new_h = self._screen.get_size()
                self._fullscreen = True
                self._set_layout(new_w, new_h)
                print(f"[DISPLAY] Fullscreen on: {new_w}x{new_h}")
            pygame.display.set_caption(self.title)
        except pygame.error as exc:
            print(f"[DISPLAY] Fullscreen toggle failed: {exc}")

    def _fit_text(self, text: Any, font_key: str, max_width: int) -> str:
        """Trim text by rendered width so compact panels do not overflow."""
        text = str(text)
        font = self._fonts[font_key]
        if font.size(text)[0] <= max_width:
            return text
        if max_width <= font.size("...")[0]:
            return ""
        while text and font.size(text + "...")[0] > max_width:
            text = text[:-1]
        return text + "..."

    def _blit_text(self, text: Any, pos: tuple[int, int], color: tuple[int, int, int], font_key: str = "sm", max_width: int | None = None):
        text = self._fit_text(text, font_key, max_width) if max_width is not None else str(text)
        if not text:
            return
        self._screen.blit(self._fonts[font_key].render(text, True, color), pos)

    def _draw_panel(
        self,
        rect: tuple[int, int, int, int],
        fill: tuple[int, int, int] | None = None,
        border: tuple[int, int, int] | None = None,
        alpha: int = 224,
        radius: int = 6,
        width: int = 1,
    ):
        x, y, w, h = rect
        surf = pygame.Surface((w, h), pygame.SRCALPHA)
        panel_fill = fill or self._theme["panel"]
        pygame.draw.rect(surf, (*panel_fill, alpha), (0, 0, w, h), border_radius=radius)
        if border:
            pygame.draw.rect(surf, (*border, min(255, alpha + 20)), (0, 0, w, h), width, border_radius=radius)
        self._screen.blit(surf, (x, y))

    def _reset_button_zones(self):
        self._button_zones = []

    def _register_button(self, rect: tuple[int, int, int, int], action: dict[str, Any]):
        self._button_zones.append({"rect": rect, "action": dict(action)})

    def _button_at(self, pos: tuple[int, int]) -> Optional[Dict[str, Any]]:
        px, py = pos
        for zone in reversed(self._button_zones):
            x, y, w, h = zone["rect"]
            if x <= px <= x + w and y <= py <= y + h:
                return dict(zone["action"])
        return None

    def _draw_background(self, pulse: float, slow_pulse: float):
        self._screen.fill(self._theme["bg"])
        grid_color = self._theme["grid"]
        for gx in range(0, self.width, 32):
            pygame.draw.line(self._screen, grid_color, (gx, 0), (gx, self.height), 1)
        for gy in range(0, self.height, 32):
            pygame.draw.line(self._screen, grid_color, (0, gy), (self.width, gy), 1)

        panel_edge = self.GAME_W + self._panel_gap // 2
        pygame.draw.line(self._screen, self._theme["line"], (panel_edge, 0), (panel_edge, self.height), 2)
        live_color = (
            int(self._theme["teal"][0] + 24 * pulse),
            self._theme["teal"][1],
            int(self._theme["teal"][2] + 30 * slow_pulse),
        )
        pygame.draw.line(self._screen, live_color, (0, self._top_bar_h - 1), (self.width, self._top_bar_h - 1), 2)

    def _draw_top_bar(self, pulse: float):
        bar = pygame.Surface((self.width, self._top_bar_h), pygame.SRCALPHA)
        bar.fill((7, 13, 18, 238))
        self._screen.blit(bar, (0, 0))
        pygame.draw.rect(self._screen, self._theme["line"], (8, 8, self.width - 16, self._top_bar_h - 16), 1, border_radius=6)

        self._blit_text("COCOON GLASS AUTHORITY", (18, 13), self._theme["teal"], "title", max_width=320)
        active = self._game_display_name or "Cocoon"
        self._blit_text(active, (18, 35), self._theme["muted"], "xs", max_width=360)

        overlay = self._cocoon_overlay or {}
        route = overlay.get("route") or self._ai_action or "facility route pending"
        mid_x = max(380, self.width // 3)
        self._blit_text(route, (mid_x, 14), self._theme["cyan"], "sm", max_width=self.width - mid_x - 170)
        teaches = overlay.get("teaches") or "semantic learning surface"
        self._blit_text(teaches, (mid_x, 34), (180, 196, 188), "xs", max_width=self.width - mid_x - 170)

        status = str(overlay.get("status") or self._control_mode or "live").upper()
        afk = self._operational_surface.get("afk", {}) if isinstance(self._operational_surface, dict) else {}
        if afk.get("enabled"):
            status = f"AFK {str(afk.get('status', 'on')).upper()}"
        pip_x = self.width - 132
        pip_color = self._theme["green"] if status in {"OK", "LIVE", "PHONE"} or status.startswith("AFK") else self._theme["amber"]
        pygame.draw.circle(self._screen, pip_color, (pip_x, 27), int(5 + 2 * pulse))
        self._blit_text(status, (pip_x + 12, 18), self._theme["text"], "sm", max_width=110)
    
    def stop(self):
        """Clean shutdown - safe even if X11 is dead."""
        self._running = False
        try:
            if pygame.get_init():
                pygame.quit()
        except Exception as e:
            print(f"[DISPLAY] pygame.quit() failed (X11 dead?): {e}")
        self._screen = None
        self._clock = None
        print("[DISPLAY] Window closed")
    
    def get_screen_frame(self) -> Optional[np.ndarray]:
        """Get the current screen as a numpy array for WebRTC streaming."""
        if not self._running or not self._screen:
            return None
        
        try:
            # Check X11 is alive first
            if not pygame.display.get_init():
                return None
            # Additional check - try to get display info (fails if X11 dead)
            _ = pygame.display.Info()
        except (pygame.error, Exception):
            return None
        
        try:
            # Get pygame surface as numpy array (RGB)
            # Make a copy immediately to avoid race conditions
            frame = pygame.surfarray.array3d(self._screen).copy()
            # pygame returns (W, H, 3), need to transpose to (H, W, 3) for video
            frame = frame.swapaxes(0, 1)
            return frame
        except pygame.error as e:
            # pygame/SDL error - display may be dead
            return None
        except Exception as e:
            return None
    
    @property
    def is_running(self) -> bool:
        return self._running
    
    def set_game(self, game_id: str, display_name: str, num_actions: int, obs_shape: tuple):
        """Configure display for a new game."""
        self._game_name = game_id
        self._game_display_name = display_name
        self._num_actions = num_actions
        self._action_labels = get_action_labels(game_id, num_actions)
        
        if len(obs_shape) == 1:
            self._state_labels = get_state_labels(game_id, obs_shape[0])
        else:
            self._state_labels = []
    
    def update_frame(self, frame: np.ndarray, obs: np.ndarray = None):
        """Update the current frame and observation."""
        with self._frame_lock:
            self._current_frame = frame
            if obs is not None:
                self._current_obs = obs
    
    def update_info(self, game_name: str = None, ai_action: str = None, 
                    viewer_name: str = None, control_mode: str = None):
        """Legacy compatibility method."""
        if game_name is not None:
            self._game_display_name = game_name
        if ai_action is not None:
            self._ai_action = ai_action
        if viewer_name is not None:
            self._viewer_name = viewer_name
        if control_mode is not None:
            self._control_mode = control_mode
            if control_mode == "AI":
                self._hold_mode = None
    
    def update_decision(
        self,
        action_probs: np.ndarray = None,
        chosen_action: int = None,
        value: float = None,
        merkle_root: str = None,
        parent_merkle: str = None,
        reward: float = None,
        was_override: bool = False,
    ):
        """Update decision display data."""
        if action_probs is not None:
            self._action_probs = action_probs
        if chosen_action is not None:
            self._chosen_action = chosen_action
        if value is not None:
            self._value_estimate = value
        if merkle_root is not None:
            self._merkle_root = merkle_root
        if parent_merkle is not None:
            self._parent_merkle = parent_merkle
        
        if reward is not None:
            self._episode_reward += reward
            self._episode_steps += 1
            self.wealth.record(
                action=self._chosen_action,
                reward=reward,
                probs=self._action_probs,
                value=self._value_estimate,
                merkle=self._merkle_root,
                was_override=was_override,
            )
    
    def update_control(self, hold_mode: str = None, viewer_name: str = None,
                       viewer_timer: float = None, viewer_actions: int = None,
                       clear: bool = False):
        """Update control state. Pass clear=True or hold_mode=None to clear all."""
        # If hold_mode is explicitly passed (even if None), update it
        if hold_mode is None and clear:
            # Explicit clear - reset all control state
            self._hold_mode = None
            self._viewer_name = None
            self._viewer_timer = 0.0
            self._viewer_actions = 0
        else:
            # Normal update - only set non-None values
            # Special case: hold_mode=None means AI is playing (no hold)
            self._hold_mode = hold_mode  # Always set this - None means AI
            if viewer_name is not None:
                self._viewer_name = viewer_name
            elif hold_mode is None:
                self._viewer_name = None  # Clear viewer when returning to AI
            if viewer_timer is not None:
                self._viewer_timer = viewer_timer
            if viewer_actions is not None:
                self._viewer_actions = viewer_actions
    
    def show_action_feedback(self, action_name: str, viewer_name: str = None):
        """Show action feedback popup for viewer commands."""
        who = viewer_name or self._viewer_name or "Player"
        self._last_action_text = f"{who} → {action_name}"
        self._last_action_time = time.time()
    
    def update_queue(self, queue: List[Dict]):
        """Update viewer queue display."""
        self._queue = queue

    def set_cocoon_overlay(self, overlay: Optional[Dict[str, Any]]):
        """Set Cocoon-specific operational metadata drawn over the game view."""
        self._cocoon_overlay = dict(overlay) if overlay else None

    def set_operational_surface(self, surface: Optional[Dict[str, Any]]):
        """Set dense Cocoon wealth/cascade/game telemetry for the pygame surface."""
        self._operational_surface = dict(surface) if isinstance(surface, dict) else None

    def set_glass_view_mode(self, mode: str):
        """Switch the main pygame surface between operational views."""
        mode = (mode or "ops").lower()
        self._glass_view_mode = mode if mode in {"ops", "games", "feed"} else "ops"
    
    def set_genesis(self, genesis_root: str):
        """Set genesis root for provenance chain."""
        self._genesis_root = genesis_root
    
    def set_menu_state(self, menu_state: Optional[Dict]):
        """Set game selection menu state (None = hidden). Thread-safe."""
        with self._menu_lock:
            # Deep copy to avoid race conditions with the dict being modified
            if menu_state is not None:
                self._menu_state = dict(menu_state)
            else:
                self._menu_state = None
    
    def new_episode(self):
        """Called when a new episode starts."""
        self._total_episodes += 1
        self._episode_reward = 0.0
        self._episode_steps = 0
    
    def handle_events(self) -> Dict[str, Any]:
        """Handle pygame events, return actions."""
        events = {"quit": False, "key": None, "hold_key": None, "fullscreen": False, "button": None, "mouse": None}
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                events["quit"] = True
            elif event.type == pygame.KEYDOWN:
                events["key"] = event.key
                
                if event.key == pygame.K_h:
                    events["hold_key"] = "H"
                elif event.key == pygame.K_t:
                    events["hold_key"] = "T"
                elif event.key == pygame.K_ESCAPE:
                    events["hold_key"] = "ESC"
                elif event.key in (pygame.K_w, pygame.K_UP):
                    events["hold_key"] = "UP"
                elif event.key in (pygame.K_s, pygame.K_DOWN):
                    events["hold_key"] = "DOWN"
                elif event.key in (pygame.K_a, pygame.K_LEFT):
                    events["hold_key"] = "LEFT"
                elif event.key in (pygame.K_d, pygame.K_RIGHT):
                    events["hold_key"] = "RIGHT"
                elif event.key == pygame.K_SPACE:
                    events["hold_key"] = "FIRE"
                elif event.key == pygame.K_F11 or (event.key in (pygame.K_RETURN, pygame.K_KP_ENTER) and event.mod & pygame.KMOD_ALT):
                    events["fullscreen"] = True
            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                events["mouse"] = event.pos
                events["button"] = self._button_at(event.pos)
        
        return events
    
    def render(self) -> bool:
        """Render one frame. Returns False if should quit."""
        if not self._running or not self._screen:
            print(f"[DISPLAY] render() returning False: _running={self._running}, _screen={self._screen is not None}")
            return False
            
        # Pre-check: Is X11 display still alive?
        # This catches Xvfb crashes before pygame segfaults
        try:
            if not pygame.display.get_init():
                print("[DISPLAY] render() returning False: pygame.display not initialized")
                self._running = False
                return False
            # Additional safety: try to get display info (will fail if X11 is dead)
            _ = pygame.display.Info()
        except pygame.error as e:
            print(f"[DISPLAY] X11 display appears dead: {e}")
            self._running = False
            return False
        except Exception as e:
            print(f"[DISPLAY] Display check failed: {e}")
            self._running = False
            return False
        
        events = self.handle_events()
        if events["quit"]:
            print("[DISPLAY] render() returning False: quit event received")
            self._running = False
            return False
        if events.get("fullscreen"):
            self.toggle_fullscreen()
        
        # Safety check - don't render if pygame quit
        if not self._running or self._screen is None:
            return False
        
        try:
            pulse = (math.sin(time.time() * 6) + 1) / 2
            slow_pulse = (math.sin(time.time() * 2) + 1) / 2
            self._reset_button_zones()
            self._draw_background(pulse, slow_pulse)
            self._draw_top_bar(pulse)
        except pygame.error:
            self._running = False
            return False

        if self._operational_surface:
            if self._glass_view_mode == "games":
                self._draw_games_view(pulse, slow_pulse)
            elif self._glass_view_mode == "feed":
                self._draw_feed_view(pulse, slow_pulse)
            else:
                self._draw_ops_view(pulse, slow_pulse)
            self._draw_glass_right_panel(pulse, slow_pulse)
        else:
            self._draw_game_view(pulse)
            self._draw_control_mesh(pulse, slow_pulse)
            self._draw_cocoon_overlay(pulse)
            self._draw_state_vector()
            self._draw_action_sequence()
            self._draw_right_panel(pulse, slow_pulse)
        
        # Draw game menu OVER everything if open - get thread-safe copy
        menu_state_copy = None
        with self._menu_lock:
            if self._menu_state and self._menu_state.get('visible'):
                menu_state_copy = dict(self._menu_state)
        
        if menu_state_copy:
            try:
                self._draw_game_menu(pulse, menu_state_copy)
            except Exception as e:
                print(f"[DISPLAY] Menu render error (ignored): {e}")
        
        # Safe display flip with X11 error handling
        try:
            pygame.display.flip()
        except pygame.error as e:
            print(f"[DISPLAY] display.flip() failed: {e}")
            self._running = False
            return False
        except Exception as e:
            print(f"[DISPLAY] display.flip() unexpected error: {e}")
            self._running = False
            return False
        # NOTE: Frame timing is handled by orchestrator's game loop
        # Don't do clock.tick() here to avoid double-limiting FPS
        
        return True

    def _main_rect(self) -> tuple[int, int, int, int]:
        x = 10
        y = self._top_bar_h + 10
        return (x, y, self.GAME_W - 20, self.height - y - 10)

    def _age_text(self, timestamp: Any) -> str:
        try:
            age = max(0, time.time() - float(timestamp))
        except Exception:
            return "--"
        if age < 1:
            return "now"
        if age < 60:
            return f"{int(age)}s"
        if age < 3600:
            return f"{int(age // 60)}m"
        return f"{int(age // 3600)}h"

    def _short_hash(self, value: Any, length: int = 18) -> str:
        text = str(value or "")
        return text[:length] if text else "none"

    def _status_color(self, status: Any) -> tuple[int, int, int]:
        text = str(status or "").lower()
        if text in {"ok", "ready", "live", "running", "watching"} or "ready" in text:
            return self._theme["green"]
        if "block" in text or "error" in text or "miss" in text or "disabled" in text:
            return self._theme["red"]
        if "need" in text or "limited" in text or "dependency" in text:
            return self._theme["amber"]
        return self._theme["cyan"]

    def _draw_mode_tabs(self, x: int, y: int, w: int):
        tabs = [("ops", "OPS"), ("games", "GAMES"), ("feed", "FEED")]
        tab_w = min(96, max(72, (w - 8) // len(tabs)))
        for idx, (mode, label) in enumerate(tabs):
            tx = x + idx * (tab_w + 4)
            active = mode == self._glass_view_mode
            fill = self._theme["panel_hi"] if active else (7, 11, 15)
            border = self._theme["teal"] if active else self._theme["line"]
            self._draw_panel((tx, y, tab_w, 24), fill=fill, border=border, alpha=232, radius=4)
            color = self._theme["teal"] if active else self._theme["muted"]
            self._blit_text(label, (tx + 12, y + 6), color, "xs", max_width=tab_w - 20)
            self._register_button((tx, y, tab_w, 24), {"type": "mode", "mode": mode})
        hint = "O/G/F switch views | Enter run | P prime | S sphere | A autopilot | R refresh"
        self._blit_text(hint, (x + len(tabs) * (tab_w + 4) + 12, y + 6), self._theme["muted"], "xs", max_width=max(80, w - len(tabs) * (tab_w + 4) - 20))

    def _draw_metric_card(self, rect: tuple[int, int, int, int], label: str, value: Any, color: tuple[int, int, int]):
        x, y, w, h = rect
        self._draw_panel(rect, fill=(8, 14, 18), border=(26, 48, 54), alpha=220, radius=5)
        self._blit_text(label.upper(), (x + 8, y + 6), self._theme["muted"], "xs", max_width=w - 16)
        self._blit_text(value, (x + 8, y + 22), color, "md", max_width=w - 16)

    def _draw_ops_view(self, pulse: float, slow_pulse: float):
        surface = self._operational_surface or {}
        x, y, w, h = self._main_rect()
        self._draw_mode_tabs(x, y, w)
        y += 34
        h -= 34

        header_h = min(126, max(110, h // 5))
        self._draw_selected_facility((x, y, w, header_h), surface, pulse)

        mid_y = y + header_h + 10
        mid_h = max(250, min(330, int(h * 0.48)))
        graph_w = max(360, int(w * 0.62))
        side_w = w - graph_w - 10
        self._draw_lattice_graph((x, mid_y, graph_w, mid_h), surface, pulse)
        self._draw_context_heat((x + graph_w + 10, mid_y, side_w, mid_h), surface, pulse)

        controller_y = mid_y + mid_h + 10
        self._draw_facility_controller((x, controller_y, w, max(120, y + h - controller_y)), surface, pulse)

    def _draw_selected_facility(self, rect: tuple[int, int, int, int], surface: dict[str, Any], pulse: float):
        x, y, w, h = rect
        selected = surface.get("selected") if isinstance(surface.get("selected"), dict) else {}
        state = surface.get("state") if isinstance(surface.get("state"), dict) else {}
        sphere = surface.get("sphere") if isinstance(surface.get("sphere"), dict) else {}
        diagnostics = surface.get("diagnostics") if isinstance(surface.get("diagnostics"), dict) else {}
        border = selected.get("border") or self._theme["teal"]
        if not isinstance(border, tuple):
            border = self._theme["teal"]
        self._draw_panel(rect, fill=(8, 16, 20), border=border, alpha=232, radius=6, width=2)

        title = selected.get("title") or state.get("active_cocoon_name") or "Cocoon facility fabric"
        self._blit_text(title, (x + 14, y + 10), self._theme["text"], "title", max_width=w - 28)
        self._blit_text(selected.get("route") or "route pending", (x + 14, y + 34), self._theme["cyan"], "xs", max_width=w - 28)
        self._blit_text(selected.get("teaches") or "semantic/reward surface", (x + 14, y + 52), (189, 205, 190), "xs", max_width=w - 28)

        status = selected.get("status", "watching")
        status_color = self._status_color(status)
        pygame.draw.circle(self._screen, status_color, (x + 18, y + h - 22), int(5 + 2 * pulse))
        self._blit_text(f"status {status}", (x + 30, y + h - 29), status_color, "sm", max_width=140)
        self._blit_text(f"blocker {selected.get('blocker', 'none')}", (x + 170, y + h - 28), self._theme["muted"], "xs", max_width=w - 190)
        afk = surface.get("afk") if isinstance(surface.get("afk"), dict) else {}
        if afk:
            afk_color = self._theme["green"] if afk.get("enabled") else self._theme["muted"]
            text = f"afk {str(afk.get('status', 'off'))} worker {afk.get('worker', 'idle')} cycles {afk.get('cycles', 0)}"
            self._blit_text(text, (x + 14, y + 72), afk_color, "xs", max_width=max(180, w - 32))

        metrics = [
            ("vocab", state.get("vocabulary_size", 0), self._theme["teal"]),
            ("relations", state.get("knowledge_relations", 0), self._theme["cyan"]),
            ("cycles", state.get("cycles", 0), self._theme["amber"]),
            ("catch/miss", f"{sphere.get('collective_catches', 0)}/{sphere.get('collective_misses', 0)}", self._theme["green"]),
            ("caps", diagnostics.get("capability_count", len(surface.get("capabilities", []) or [])), (222, 201, 118)),
        ]
        card_w = max(86, min(122, (w - 32) // len(metrics)))
        card_x = x + w - (card_w + 6) * len(metrics) - 8
        for idx, (label, value, color) in enumerate(metrics):
            self._draw_metric_card((card_x + idx * (card_w + 6), y + h - 58, card_w, 48), label, value, color)

    def _draw_lattice_graph(self, rect: tuple[int, int, int, int], surface: dict[str, Any], pulse: float):
        x, y, w, h = rect
        self._draw_panel(rect, fill=(6, 12, 17), border=self._theme["line"], alpha=222, radius=6)
        self._blit_text("CASCADE-LATTICE / FACILITY LINEAGE", (x + 12, y + 10), self._theme["teal"], "sm", max_width=w - 24)
        facility_map = surface.get("facility_map") if isinstance(surface.get("facility_map"), dict) else {}
        nodes = facility_map.get("nodes") if isinstance(facility_map.get("nodes"), list) else []
        edges = facility_map.get("edges") if isinstance(facility_map.get("edges"), list) else []
        if not nodes:
            groups = sorted({str(cap.get("group")) for cap in surface.get("capabilities", []) if isinstance(cap, dict) and cap.get("group")})
            nodes = [{"id": "authority", "label": "Authority", "type": "router", "status": "ready"}] + [
                {"id": group, "label": group.title(), "type": "capability", "status": "ready"} for group in groups[:14]
            ]
            edges = [{"from": "authority", "to": node["id"], "label": "route"} for node in nodes[1:]]

        nodes = nodes[:22]
        node_by_id = {str(node.get("id")): node for node in nodes if isinstance(node, dict)}
        cx = x + w // 2
        cy = y + h // 2 + 8
        radius = max(72, min(w, h) // 2 - 34)
        positions: dict[str, tuple[int, int]] = {}
        if "authority" in node_by_id:
            positions["authority"] = (cx, cy)
        for idx, node_id in enumerate([nid for nid in node_by_id if nid != "authority"]):
            angle = -math.pi / 2 + idx * (math.tau / max(1, len(node_by_id) - 1))
            wobble = 1.0 + 0.04 * math.sin(time.time() * 2 + idx)
            positions[node_id] = (int(cx + math.cos(angle) * radius * wobble), int(cy + math.sin(angle) * radius * wobble))

        for edge in edges[:40]:
            if not isinstance(edge, dict):
                continue
            source = str(edge.get("from", ""))
            target = str(edge.get("to", ""))
            if source in positions and target in positions:
                color = (28, 68, 74)
                pygame.draw.line(self._screen, color, positions[source], positions[target], 1)

        selected_key = str(surface.get("selected_key") or "")
        for node_id, node in node_by_id.items():
            px, py = positions.get(node_id, (cx, cy))
            status_color = self._status_color(node.get("status"))
            active = node_id == selected_key or str(node.get("label", "")).lower() in selected_key
            node_r = 13 if node_id == "authority" else 9
            if active:
                pygame.draw.circle(self._screen, (*self._theme["amber"],), (px, py), node_r + int(4 * pulse), 2)
            pygame.draw.circle(self._screen, status_color, (px, py), node_r)
            pygame.draw.circle(self._screen, (2, 8, 10), (px, py), max(3, node_r - 5))
            label = str(node.get("label") or node_id)
            self._blit_text(label, (px - 42, py + node_r + 4), self._theme["muted"], "xs", max_width=84)

    def _draw_context_heat(self, rect: tuple[int, int, int, int], surface: dict[str, Any], pulse: float):
        x, y, w, h = rect
        self._draw_panel(rect, fill=(7, 13, 18), border=self._theme["line"], alpha=222, radius=6)
        self._blit_text("CONTEXT / WEALTH HEAT", (x + 12, y + 10), self._theme["cyan"], "sm", max_width=w - 24)
        caps = [cap for cap in surface.get("capabilities", []) if isinstance(cap, dict)]
        events = [event for event in surface.get("events", []) if isinstance(event, dict)]
        lessons = [lesson for lesson in surface.get("lessons", []) if isinstance(lesson, dict)]
        groups: dict[str, int] = {}
        for cap in caps:
            group = str(cap.get("group") or "other")
            groups[group] = groups.get(group, 0) + 1
        values = list(groups.values()) or [1]
        max_value = max(values)
        grid_x = x + 14
        grid_y = y + 38
        cell = max(12, min(22, (w - 30) // 10))
        for row in range(6):
            for col in range(10):
                idx = row * 10 + col
                base = values[idx % len(values)] / max(1, max_value)
                event_boost = 0.25 if idx < len(events) else 0.0
                lesson_boost = 0.18 if idx < len(lessons) else 0.0
                heat = min(1.0, base * 0.55 + event_boost + lesson_boost + 0.08 * pulse)
                color = (
                    int(20 + 40 * heat),
                    int(46 + 160 * heat),
                    int(58 + 118 * heat),
                )
                pygame.draw.rect(self._screen, color, (grid_x + col * cell, grid_y + row * cell, cell - 2, cell - 2), border_radius=2)

        label_y = grid_y + 6 * cell + 12
        for idx, (group, count) in enumerate(sorted(groups.items(), key=lambda item: (-item[1], item[0]))[:5]):
            yy = label_y + idx * 18
            self._blit_text(group, (x + 14, yy), self._theme["muted"], "xs", max_width=max(70, w // 2 - 18))
            bar_w = max(28, w - 110)
            pygame.draw.rect(self._screen, (16, 26, 30), (x + w - bar_w - 12, yy + 3, bar_w, 7), border_radius=3)
            pygame.draw.rect(self._screen, self._theme["teal"], (x + w - bar_w - 12, yy + 3, int(bar_w * count / max(1, max_value)), 7), border_radius=3)
            self._blit_text(count, (x + w - 34, yy - 2), self._theme["text"], "xs", max_width=28)

    def _risk_color(self, cap: dict[str, Any]) -> tuple[int, int, int]:
        risk = str(cap.get("risk_class") or "").lower()
        if risk == "external":
            return self._theme["amber"]
        if risk == "overclock" or cap.get("destructive"):
            return self._theme["red"]
        if cap.get("mutates"):
            return self._theme["cyan"]
        return self._theme["green"]

    def _draw_facility_controller(self, rect: tuple[int, int, int, int], surface: dict[str, Any], pulse: float):
        x, y, w, h = rect
        self._draw_panel(rect, fill=(4, 9, 12), border=(37, 70, 68), alpha=232, radius=6)
        self._blit_text("PRIMARY PROCESS CONTROLLER / ALL FACILITIES", (x + 12, y + 9), self._theme["teal"], "sm", max_width=w - 24)
        self._blit_text("click facility to select | side buttons run/study/sequence | mutating/external routes stay visibly marked", (x + 12, y + 27), self._theme["muted"], "xs", max_width=w - 24)

        caps = [cap for cap in surface.get("capabilities", []) if isinstance(cap, dict)]
        if not caps:
            caps = [
                {"key": label.lower().replace(" ", "_"), "label": label, "group": "runtime", "risk_class": "inspect"}
                for label in self._action_labels
            ]
        selected_key = str(surface.get("selected_key") or "")
        group_order = ["diagnostics", "memory", "persistence", "language", "reasoning", "compound", "action", "games", "external", "overclock"]
        grouped: dict[str, list[dict[str, Any]]] = {}
        for cap in caps:
            grouped.setdefault(str(cap.get("group") or "other"), []).append(cap)
        ordered_groups = [group for group in group_order if group in grouped] + sorted(group for group in grouped if group not in group_order)
        ordered_caps = []
        for group in ordered_groups:
            ordered_caps.extend(sorted(grouped[group], key=lambda cap: str(cap.get("key") or cap.get("label"))))

        top = y + 48
        available_h = max(40, h - 58)
        cols = 4 if w >= 680 else 3 if w >= 520 else 2
        gap = 6
        chip_h = 18
        rows = max(1, available_h // (chip_h + 4))
        max_items = cols * rows
        chip_w = (w - 24 - gap * (cols - 1)) // cols

        for idx, cap in enumerate(ordered_caps[:max_items]):
            col = idx // rows
            row = idx % rows
            cx = x + 12 + col * (chip_w + gap)
            cy = top + row * (chip_h + 4)
            key = str(cap.get("key") or "")
            active = key == selected_key
            color = self._risk_color(cap)
            fill = (16, 26, 24) if active else (8, 15, 17)
            border = color if active else (28, 48, 50)
            self._draw_panel((cx, cy, chip_w, chip_h), fill=fill, border=border, alpha=225, radius=3, width=2 if active else 1)
            marker = "*" if cap.get("mutates") else "!" if cap.get("destructive") else ">"
            label = f"{marker} {cap.get('label') or key}"
            self._blit_text(label, (cx + 5, cy + 4), color if active else self._theme["text"], "xs", max_width=chip_w - 10)
            self._register_button((cx, cy, chip_w, chip_h), {"type": "select_cap", "key": key})

        if len(ordered_caps) > max_items:
            hidden = len(ordered_caps) - max_items
            self._blit_text(f"+ {hidden} more facilities hidden at this window size; use fullscreen for full grid", (x + 12, y + h - 16), self._theme["amber"], "xs", max_width=w - 24)

    def _event_line(self, item: dict[str, Any]) -> str:
        name = item.get("name") or item.get("kind") or item.get("type") or item.get("stage") or "event"
        status = item.get("status") or item.get("ok") or item.get("result") or ""
        keys = item.get("keys")
        data = item.get("data")
        if not keys and isinstance(data, dict):
            keys = ", ".join(list(data.keys())[:5])
        age = self._age_text(item.get("time") or item.get("timestamp"))
        status_text = str(status) if status != "" else "seen"
        return f"[{age:>3}] {name}: {status_text} {keys or ''}".strip()

    def _draw_terminal_feed(self, rect: tuple[int, int, int, int], surface: dict[str, Any], limit: int = 14):
        x, y, w, h = rect
        self._draw_panel(rect, fill=(3, 8, 10), border=(34, 58, 52), alpha=232, radius=6)
        self._blit_text("OPERATION TAPE", (x + 12, y + 9), self._theme["green"], "sm", max_width=w - 24)
        rows = []
        for source in ("history", "events", "lessons"):
            values = surface.get(source)
            if isinstance(values, list):
                rows.extend(item for item in values if isinstance(item, dict))
        rows = rows[: max(1, limit)]
        line_y = y + 30
        line_h = 16
        for idx, row in enumerate(rows):
            yy = line_y + idx * line_h
            if yy + line_h > y + h - 6:
                break
            color = self._status_color(row.get("status") or row.get("kind") or row.get("type"))
            pygame.draw.rect(self._screen, (8, 18, 18), (x + 10, yy - 1, w - 20, line_h), border_radius=2)
            self._blit_text(self._event_line(row), (x + 16, yy + 1), color if idx == 0 else self._theme["muted"], "xs", max_width=w - 32)
        if not rows:
            self._blit_text("No event data yet. Prime cache with P or run a capability.", (x + 16, line_y), self._theme["muted"], "xs", max_width=w - 32)

    def _draw_games_view(self, pulse: float, slow_pulse: float):
        surface = self._operational_surface or {}
        x, y, w, h = self._main_rect()
        self._draw_mode_tabs(x, y, w)
        y += 34
        h -= 34

        hero_w = max(430, int(w * 0.62))
        hero_h = h
        self._draw_panel((x, y, hero_w, hero_h), fill=(4, 10, 14), border=self._theme["cyan"], alpha=226, radius=6)
        self._blit_text("SPHERE ARENA LIVE GAME RECONSTRUCTION", (x + 14, y + 12), self._theme["cyan"], "sm", max_width=hero_w - 28)
        sphere_rect = (x + 14, y + 38, hero_w - 28, max(260, hero_h - 168))
        sphere = surface.get("sphere") if isinstance(surface.get("sphere"), dict) else {}
        self._draw_sphere_reconstruction(sphere_rect, sphere, pulse)
        stats_y = sphere_rect[1] + sphere_rect[3] + 12
        stat_lines = [
            f"frames {sphere.get('frames_run', sphere.get('frames_requested', 0))} / requested {sphere.get('frames_requested', 0)}",
            f"catches {sphere.get('collective_catches', 0)}  misses {sphere.get('collective_misses', 0)}  streak {sphere.get('best_streak', sphere.get('streak', 0))}",
            f"receipt {self._short_hash(sphere.get('receipt'))}",
            "frame source: authoritative headless state; raw frame relay appears here when emitted",
        ]
        for idx, line in enumerate(stat_lines):
            self._blit_text(line, (x + 18, stats_y + idx * 18), self._theme["muted"] if idx != 1 else self._theme["green"], "xs", max_width=hero_w - 36)

        cards_x = x + hero_w + 10
        cards_w = w - hero_w - 10
        games = [game for game in surface.get("games", []) if isinstance(game, dict)]
        if not games:
            games = [{"key": "sphere", "label": "Sphere", "status": "ready"}]
        card_h = max(86, (h - 10 * (min(len(games), 5) - 1)) // max(1, min(len(games), 5)))
        for idx, game in enumerate(games[:5]):
            cy = y + idx * (card_h + 10)
            self._draw_game_lane_card((cards_x, cy, cards_w, card_h), game, pulse)

    def _draw_sphere_reconstruction(self, rect: tuple[int, int, int, int], sphere: dict[str, Any], pulse: float):
        x, y, w, h = rect
        self._draw_panel(rect, fill=(2, 7, 10), border=(26, 54, 68), alpha=245, radius=5)
        cx = x + w // 2
        cy = y + h // 2
        radius = min(w, h) // 2 - 24
        pygame.draw.circle(self._screen, (21, 50, 64), (cx, cy), radius, 2)
        for step in range(1, 4):
            pygame.draw.circle(self._screen, (10, 28, 38), (cx, cy), int(radius * step / 4), 1)
        for angle in range(0, 180, 30):
            dx = int(math.cos(math.radians(angle)) * radius)
            dy = int(math.sin(math.radians(angle)) * radius)
            pygame.draw.line(self._screen, (10, 28, 38), (cx - dx, cy - dy), (cx + dx, cy + dy), 1)

        render_state = sphere.get("render_state") if isinstance(sphere.get("render_state"), dict) else {}
        organisms = render_state.get("organisms") if isinstance(render_state.get("organisms"), list) else []
        balls = render_state.get("balls") if isinstance(render_state.get("balls"), list) else []
        tail = sphere.get("render_tail") if isinstance(sphere.get("render_tail"), list) else []

        def project(pos: Any) -> tuple[int, int]:
            try:
                px, py, pz = float(pos[0]), float(pos[1]), float(pos[2])
            except Exception:
                return cx, cy
            scale = radius / 2.35
            return int(cx + (px + 0.25 * pz) * scale), int(cy + (py - 0.15 * pz) * scale)

        for sample in tail[-12:]:
            if not isinstance(sample, dict):
                continue
            for ball in sample.get("balls", []) if isinstance(sample.get("balls"), list) else []:
                pos = project(ball.get("position"))
                pygame.draw.circle(self._screen, (106, 86, 28), pos, 2)

        if not organisms:
            count = max(2, min(8, len(sphere.get("rewards", {}) or {}) or 4))
            organisms = [{"idx": idx, "theta": idx * math.tau / count + pulse, "phi": math.pi / 2, "alive": True} for idx in range(count)]
        for idx, org in enumerate(organisms):
            if "position" in org:
                pos = project(org.get("position"))
            else:
                theta = float(org.get("theta", idx))
                phi = float(org.get("phi", math.pi / 2))
                pos = (int(cx + math.cos(theta) * math.sin(phi) * radius * 0.82), int(cy + math.sin(theta) * math.sin(phi) * radius * 0.82))
            color = self._theme["green"] if org.get("alive", True) else self._theme["red"]
            pygame.draw.circle(self._screen, color, pos, 6)
            pygame.draw.circle(self._screen, (234, 244, 236), pos, 8, 1)
            self._blit_text(str(org.get("idx", idx)), (pos[0] + 8, pos[1] - 6), self._theme["muted"], "xs", max_width=28)

        if not balls:
            balls = [{"position": (math.cos(time.time()) * 1.5, math.sin(time.time() * 0.9) * 1.5, 0.5), "active": True}]
        for ball in balls:
            pos = project(ball.get("position"))
            pygame.draw.circle(self._screen, self._theme["amber"], pos, int(7 + 3 * pulse))
            pygame.draw.circle(self._screen, (255, 245, 170), pos, 12, 1)

    def _draw_game_lane_card(self, rect: tuple[int, int, int, int], game: dict[str, Any], pulse: float):
        x, y, w, h = rect
        status = game.get("status", "unknown")
        color = self._status_color(status)
        self._draw_panel(rect, fill=(8, 14, 18), border=color, alpha=224, radius=6)
        self._blit_text(str(game.get("label") or game.get("key") or "Game").upper(), (x + 12, y + 10), color, "sm", max_width=w - 24)
        self._blit_text(f"status {status}", (x + 12, y + 30), self._theme["muted"], "xs", max_width=w - 24)
        details = game.get("details") if isinstance(game.get("details"), list) else []
        for idx, detail in enumerate(details[:3]):
            self._blit_text(detail, (x + 12, y + 49 + idx * 16), self._theme["text"] if idx == 0 else self._theme["muted"], "xs", max_width=w - 24)
        if game.get("active"):
            pygame.draw.circle(self._screen, color, (x + w - 18, y + 18), int(5 + 2 * pulse))

    def _draw_feed_view(self, pulse: float, slow_pulse: float):
        surface = self._operational_surface or {}
        x, y, w, h = self._main_rect()
        self._draw_mode_tabs(x, y, w)
        y += 34
        h -= 34
        left_w = max(420, int(w * 0.66))
        self._draw_terminal_feed((x, y, left_w, h), surface, limit=max(18, h // 16 - 3))

        right_x = x + left_w + 10
        right_w = w - left_w - 10
        top_h = max(130, int(h * 0.38))
        self._draw_receipt_panel((right_x, y, right_w, top_h), surface)
        self._draw_lessons_panel((right_x, y + top_h + 10, right_w, h - top_h - 10), surface)

    def _draw_receipt_panel(self, rect: tuple[int, int, int, int], surface: dict[str, Any]):
        x, y, w, h = rect
        self._draw_panel(rect, fill=(7, 12, 17), border=self._theme["cyan"], alpha=224, radius=6)
        self._blit_text("RECEIPT CHAIN", (x + 12, y + 10), self._theme["cyan"], "sm", max_width=w - 24)
        receipts = [item for item in surface.get("receipts", []) if isinstance(item, dict)]
        if not receipts:
            state = surface.get("state") if isinstance(surface.get("state"), dict) else {}
            receipts = [{"kind": "state", "receipt": state.get("last_receipt")}]
        for idx, item in enumerate(receipts[: max(1, (h - 34) // 18)]):
            yy = y + 34 + idx * 18
            self._blit_text(str(item.get("kind") or "receipt"), (x + 12, yy), self._theme["muted"], "xs", max_width=max(60, w // 3))
            self._blit_text(self._short_hash(item.get("receipt"), 28), (x + max(82, w // 3), yy), self._theme["green"], "xs", max_width=w - max(96, w // 3) - 8)

    def _draw_lessons_panel(self, rect: tuple[int, int, int, int], surface: dict[str, Any]):
        x, y, w, h = rect
        self._draw_panel(rect, fill=(8, 13, 15), border=self._theme["green"], alpha=224, radius=6)
        self._blit_text("LESSONS / CUES", (x + 12, y + 10), self._theme["green"], "sm", max_width=w - 24)
        lessons = [item for item in surface.get("lessons", []) if isinstance(item, dict)]
        feed_reports = [item for item in surface.get("feed_reports", []) if isinstance(item, dict)]
        rows = lessons[:8] + feed_reports[:4]
        for idx, item in enumerate(rows[: max(1, (h - 34) // 18)]):
            yy = y + 34 + idx * 18
            text = item.get("text") or item.get("note") or item.get("title") or item.get("kind") or item.get("source") or str(item)[:80]
            self._blit_text(text, (x + 12, yy), self._theme["muted"], "xs", max_width=w - 24)
        if not rows:
            self._blit_text("No lessons surfaced yet.", (x + 12, y + 34), self._theme["muted"], "xs", max_width=w - 24)

    def _draw_glass_right_panel(self, pulse: float, slow_pulse: float):
        surface = self._operational_surface or {}
        panel_x = self.GAME_W + self._panel_gap
        x = panel_x + 8
        y = self._top_bar_h + 10
        w = self.width - x - 10
        h = self.height - y - 10
        self._draw_panel((x, y, w, h), fill=(6, 11, 16), border=self._theme["line"], alpha=230, radius=6)
        self._blit_text("LIVE RELAY", (x + 12, y + 10), self._theme["teal"], "title", max_width=w - 24)
        mode = self._glass_view_mode.upper()
        self._blit_text(f"view {mode} | pygame tui relay", (x + 12, y + 36), self._theme["muted"], "xs", max_width=w - 24)

        selected = surface.get("selected") if isinstance(surface.get("selected"), dict) else {}
        afk = surface.get("afk") if isinstance(surface.get("afk"), dict) else {}
        controls_y = y + h - 78
        sy = y + 62
        self._draw_panel((x + 10, sy, w - 20, 76), fill=(9, 17, 22), border=self._status_color(selected.get("status")), alpha=224, radius=5)
        self._blit_text(str(selected.get("title") or "selected facility"), (x + 20, sy + 10), self._theme["text"], "sm", max_width=w - 40)
        self._blit_text(str(selected.get("last_keys") or "keys pending"), (x + 20, sy + 30), self._theme["muted"], "xs", max_width=w - 40)
        self._blit_text(f"receipt {selected.get('receipt', 'none')}", (x + 20, sy + 48), self._theme["green"], "xs", max_width=w - 40)
        if afk:
            afk_color = self._theme["green"] if afk.get("enabled") else self._theme["muted"]
            self._blit_text(f"afk {afk.get('status', 'off')} | {afk.get('worker', 'idle')} | {afk.get('cycles', 0)}", (x + 20, sy + 62), afk_color, "xs", max_width=w - 40)

        dy = sy + 88
        self._draw_command_bank((x + 10, dy, w - 20, 112), pulse)
        dy += 124
        self._draw_compact_decision_list((x + 10, dy, w - 20, 132), pulse)
        dy += 144
        self._draw_receipt_panel((x + 10, dy, w - 20, 96), surface)
        dy += 108
        games = [game for game in surface.get("games", []) if isinstance(game, dict)]
        game_h = max(76, min(108, controls_y - dy - 12))
        self._draw_panel((x + 10, dy, w - 20, game_h), fill=(9, 14, 18), border=self._theme["amber"], alpha=220, radius=5)
        self._blit_text("GAME LANES", (x + 20, dy + 9), self._theme["amber"], "sm", max_width=w - 40)
        for idx, game in enumerate(games[:4]):
            yy = dy + 30 + idx * 18
            self._blit_text(str(game.get("label") or game.get("key")), (x + 20, yy), self._status_color(game.get("status")), "xs", max_width=90)
            self._blit_text(str(game.get("status")), (x + 120, yy), self._theme["muted"], "xs", max_width=w - 150)
        self._draw_panel((x + 10, controls_y, w - 20, 66), fill=(10, 12, 9), border=(90, 82, 45), alpha=218, radius=5)
        self._blit_text("CONTROLS", (x + 20, controls_y + 8), self._theme["amber"], "sm", max_width=w - 40)
        self._blit_text("O ops  G games  F feed  V cycle", (x + 20, controls_y + 30), self._theme["text"], "xs", max_width=w - 40)
        self._blit_text("Enter/P/I/N/E/L/M/S/A/D/R/Q", (x + 20, controls_y + 46), self._theme["muted"], "xs", max_width=w - 40)

    def _draw_command_bank(self, rect: tuple[int, int, int, int], pulse: float):
        x, y, w, h = rect
        self._draw_panel(rect, fill=(9, 14, 13), border=(48, 64, 42), alpha=222, radius=5)
        self._blit_text("PROCESS BUTTONS", (x + 10, y + 8), self._theme["amber"], "sm", max_width=w - 20)
        commands = [
            ("RUN", "run_selected", self._theme["green"]),
            ("STUDY", "study_selected", self._theme["cyan"]),
            ("NEXT", "followup", self._theme["teal"]),
            ("SEQ", "sequence", (202, 186, 98)),
            ("AFK", "afk", (112, 230, 150)),
            ("PRIME", "prime", self._theme["amber"]),
            ("SPHERE", "sphere", (226, 182, 82)),
            ("AUTO", "autopilot", (114, 190, 246)),
            ("HARDEN", "harden", (230, 122, 92)),
            ("REFRESH", "refresh", self._theme["muted"]),
            ("LOGS", "logs", (174, 198, 224)),
            ("MAP", "map", (174, 198, 224)),
            ("FULL", "fullscreen", self._theme["teal"]),
        ]
        cols = 5
        gap = 5
        btn_w = max(54, (w - 20 - gap * (cols - 1)) // cols)
        btn_h = 20
        start_y = y + 30
        for idx, (label, action, color) in enumerate(commands):
            col = idx % cols
            row = idx // cols
            bx = x + 10 + col * (btn_w + gap)
            by = start_y + row * (btn_h + 5)
            if by + btn_h > y + h - 4:
                break
            self._draw_panel((bx, by, btn_w, btn_h), fill=(11, 19, 19), border=color, alpha=226, radius=3)
            self._blit_text(label, (bx + 6, by + 5), color, "xs", max_width=btn_w - 12)
            self._register_button((bx, by, btn_w, btn_h), {"type": "command", "command": action})

    def _draw_compact_decision_list(self, rect: tuple[int, int, int, int], pulse: float):
        x, y, w, h = rect
        self._draw_panel(rect, fill=(8, 13, 18), border=self._theme["cyan"], alpha=220, radius=5)
        self._blit_text("CAPABILITY LATTICE", (x + 10, y + 8), self._theme["cyan"], "sm", max_width=w - 20)
        if self._action_probs is None:
            self._blit_text("No capability probabilities yet.", (x + 10, y + 32), self._theme["muted"], "xs", max_width=w - 20)
            return
        max_actions = min(len(self._action_probs), 8)
        max_idx = int(np.argmax(self._action_probs)) if len(self._action_probs) else 0
        bar_w = max(70, w - 138)
        for i in range(max_actions):
            p = float(self._action_probs[i]) if i < len(self._action_probs) else 0.0
            label = self._action_labels[i] if i < len(self._action_labels) else f"A{i}"
            yy = y + 32 + i * 16
            color = self._theme["amber"] if i == self._chosen_action else self._theme["teal"] if i == max_idx else self._theme["muted"]
            self._blit_text(label, (x + 10, yy), color, "xs", max_width=86)
            pygame.draw.rect(self._screen, (18, 28, 34), (x + 102, yy + 4, bar_w, 6), border_radius=3)
            pygame.draw.rect(self._screen, color, (x + 102, yy + 4, int(bar_w * min(1.0, max(0.0, p))), 6), border_radius=3)
            self._blit_text(f"{p:.0%}", (x + 108 + bar_w, yy - 2), self._theme["text"], "xs", max_width=34)

    def _draw_cocoon_overlay(self, pulse: float):
        """Draw Cocoon route, cache, and sequence telemetry from real API data."""
        overlay = self._cocoon_overlay
        if not overlay:
            return
        game_y = 50
        panel_w = min(420, self.GAME_W - 24)
        panel_h = 218
        x = 12
        y = game_y + 12
        try:
            panel = pygame.Surface((panel_w, panel_h), pygame.SRCALPHA)
            panel.fill((4, 10, 18, 214))
            border = overlay.get("border", (0, 180, 160))
            glow = int(35 * pulse)
            pygame.draw.rect(panel, (min(255, border[0] + glow), min(255, border[1] + glow), min(255, border[2] + glow)), (0, 0, panel_w, panel_h), 2)
            self._screen.blit(panel, (x, y))
        except pygame.error:
            return

        def draw(line: str, dy: int, color=(220, 235, 235), font="sm"):
            text = str(line)
            limit = 58 if font == "xs" else 46
            if len(text) > limit:
                text = text[: max(0, limit - 3)] + "..."
            lbl = self._fonts[font].render(text, True, color)
            self._screen.blit(lbl, (x + 12, y + dy))

        title = overlay.get("title", "Cocoon Facility")
        draw(title, 10, (245, 246, 220), "md")
        draw(overlay.get("route", ""), 35, (130, 220, 255), "xs")
        draw(overlay.get("teaches", ""), 54, (210, 215, 205), "xs")

        status = overlay.get("status", "watching")
        status_color = (80, 230, 150) if status == "ok" else (240, 190, 70) if status in {"running", "blocked"} else (180, 205, 225)
        draw(f"status: {status}", 77, status_color, "sm")
        draw(f"last keys: {overlay.get('last_keys', 'none')}", 98, (175, 205, 225), "xs")
        draw(f"cache: {overlay.get('cache', 'empty')}", 116, (175, 225, 190), "xs")
        draw(f"sequence: {overlay.get('sequence', 'none')}", 134, (230, 205, 130), "xs")
        draw(f"receipt: {overlay.get('receipt', 'none')}", 152, (150, 190, 255), "xs")
        draw(f"blocker: {overlay.get('blocker', 'none')}", 170, (255, 160, 130), "xs")
        draw("Enter run | P prime | I study | N followup | L logs | M map", 194, (150, 245, 210), "xs")
    
    def _draw_game_view(self, pulse: float):
        """Draw the game frame."""
        game_y = 50
        
        with self._frame_lock:
            if self._current_frame is not None:
                frame = self._current_frame
                if len(frame.shape) == 2:
                    frame = np.stack([frame] * 3, axis=-1)
                
                try:
                    surf = pygame.surfarray.make_surface(frame.swapaxes(0, 1))
                    # Use scale (nearest-neighbor) instead of smoothscale to preserve
                    # tiny pixel details like bullets in Space Invaders
                    surf = pygame.transform.scale(surf, (self.GAME_W, self.GAME_H))
                    self._screen.blit(surf, (0, game_y))
                except:
                    pygame.draw.rect(self._screen, (20, 20, 30), (0, game_y, self.GAME_W, self.GAME_H))
            else:
                pygame.draw.rect(self._screen, (20, 20, 30), (0, game_y, self.GAME_W, self.GAME_H))
        
        # Normalize hold_mode for comparison
        hold_mode_upper = (self._hold_mode or "").upper()
        is_held = hold_mode_upper in ("CONTINUOUS", "FREEZE", "TAKEOVER", "HOLD", "SINGLE")
        is_watching = hold_mode_upper == "WATCHING"
        
        if is_held:
            if hold_mode_upper == "CONTINUOUS":
                border_color = (int(200 + 55 * pulse), int(150 + 50 * pulse), 0)  # Orange
            else:
                border_color = (int(200 + 55 * pulse), 50, 50)  # Red
        elif is_watching:
            border_color = (0, int(200 + 55 * pulse), 255)  # Blue - viewer watching AI
        else:
            border_color = (0, 255, 150)  # Green - no viewer
        pygame.draw.rect(self._screen, border_color, (0, game_y, self.GAME_W, self.GAME_H), 3)
        
        # Show viewer overlay when someone has an active session (timer > 0)
        # This shows regardless of control mode (AI playing, HOLD, or CONTINUOUS)
        if self._viewer_name and self._viewer_timer and self._viewer_timer > 0:
            self._draw_viewer_overlay(game_y, pulse, is_watching)
        
        lbl = self._fonts['md'].render(f"{self._game_display_name}", True, (255, 255, 255))
        self._screen.blit(lbl, (20, 15))
        
        ep_txt = f"Episode {self._total_episodes} • Step {self._episode_steps} • Reward: {self._episode_reward:+.1f}"
        ep_lbl = self._fonts['xs'].render(ep_txt, True, (150, 150, 150))
        self._screen.blit(ep_lbl, (20, 35))
    
    def _draw_viewer_overlay(self, game_y: int, pulse: float, is_watching: bool = False):
        """Draw prominent viewer control overlay on game area."""
        # Top banner showing player name
        banner_h = 40
        banner_surf = pygame.Surface((self.GAME_W, banner_h), pygame.SRCALPHA)
        banner_surf.fill((0, 0, 0, 180))
        self._screen.blit(banner_surf, (0, game_y))
        
        # Player name with status
        if is_watching:
            name_color = (0, int(200 + 55 * pulse), 255)  # Blue pulsing
            player_text = f"🎮 {self._viewer_name}'s TURN - Press H/T to control!"
        else:
            name_color = (
                int(255 * (0.7 + 0.3 * pulse)),
                int(200 * (0.8 + 0.2 * pulse)),
                0
            )
            player_text = f"🎮 {self._viewer_name} IS PLAYING!"
        player_lbl = self._fonts['lg'].render(player_text, True, name_color)
        self._screen.blit(player_lbl, (15, game_y + 5))
        
        # Timer on the right side
        time_remaining = max(0, self._viewer_timer or 0)
        mins = int(time_remaining // 60)
        secs = int(time_remaining % 60)
        
        # Color based on time remaining
        if time_remaining < 30:
            timer_color = (255, int(50 + 100 * pulse), 50)  # Red/pulsing when low
        elif time_remaining < 60:
            timer_color = (255, 200, 50)  # Yellow when getting low
        else:
            timer_color = (0, 255, 150)  # Green when plenty of time
        
        timer_text = f"{mins}:{secs:02d}"
        timer_lbl = self._fonts['lg'].render(timer_text, True, timer_color)
        timer_w = timer_lbl.get_width()
        self._screen.blit(timer_lbl, (self.GAME_W - timer_w - 15, game_y + 5))
        
        # Actions counter
        actions_text = f"Actions: {self._viewer_actions or 0}"
        actions_lbl = self._fonts['sm'].render(actions_text, True, (200, 200, 200))
        actions_w = actions_lbl.get_width()
        self._screen.blit(actions_lbl, (self.GAME_W - actions_w - 15, game_y + 25))
        
        # Action feedback popup (center of screen, fades out)
        action_age = time.time() - self._last_action_time
        if action_age < self._action_feedback_duration and self._last_action_text:
            # Fade out as it ages
            alpha = int(255 * (1.0 - action_age / self._action_feedback_duration))
            feedback_color = (0, 255, 200)
            feedback_lbl = self._fonts['lg'].render(self._last_action_text, True, feedback_color)
            
            # Create surface with alpha
            feedback_surf = pygame.Surface((feedback_lbl.get_width() + 20, 45), pygame.SRCALPHA)
            feedback_surf.fill((0, 0, 0, int(alpha * 0.7)))
            feedback_surf.blit(feedback_lbl, (10, 8))
            
            # Center it
            fx = (self.GAME_W - feedback_surf.get_width()) // 2
            fy = game_y + (self.GAME_H // 2) - 20
            self._screen.blit(feedback_surf, (fx, fy))
        
        # Bottom control hints bar
        hint_h = 30
        hint_y = game_y + self.GAME_H - hint_h
        hint_surf = pygame.Surface((self.GAME_W, hint_h), pygame.SRCALPHA)
        hint_surf.fill((0, 0, 0, 150))
        self._screen.blit(hint_surf, (0, hint_y))
        
        # Control hints
        hints = "WASD + SPACE | Use extension overlay to play"
        hint_lbl = self._fonts['sm'].render(hints, True, (180, 180, 200))
        self._screen.blit(hint_lbl, (15, hint_y + 5))

    def _draw_game_menu(self, pulse: float, menu_state: Dict = None):
        """
        Draw game selection menu overlay - ARCADE STYLE with colored game cards!
        
        Args:
            pulse: Animation pulse value 0-1
            menu_state: Menu state dict (thread-safe copy). If None, uses self._menu_state
        """
        # Use passed menu_state for thread safety, fall back to instance variable
        ms = menu_state if menu_state is not None else self._menu_state
        if not ms:
            return
        
        # Game color schemes - vibrant arcade colors!
        GAME_COLORS = {
            'Qbert': {'bg': (255, 100, 50), 'accent': (255, 200, 100)},      # Orange
            'Q*bert': {'bg': (255, 100, 50), 'accent': (255, 200, 100)},     # Orange
            'Asteroids': {'bg': (100, 100, 200), 'accent': (200, 200, 255)}, # Blue
            'MsPacman': {'bg': (255, 200, 50), 'accent': (255, 255, 150)},   # Yellow
            'Ms. Pac-Man': {'bg': (255, 200, 50), 'accent': (255, 255, 150)},# Yellow
            'RoadRunner': {'bg': (100, 200, 255), 'accent': (200, 240, 255)},# Cyan
            'Road Runner': {'bg': (100, 200, 255), 'accent': (200, 240, 255)},# Cyan
            'BeamRider': {'bg': (200, 50, 200), 'accent': (255, 150, 255)},  # Purple
            'Beam Rider': {'bg': (200, 50, 200), 'accent': (255, 150, 255)}, # Purple
            'Seaquest': {'bg': (50, 150, 200), 'accent': (150, 220, 255)},   # Ocean blue
            'LunarLander': {'bg': (80, 80, 120), 'accent': (180, 180, 220)}, # Gray/space
            'Lunar Lander': {'bg': (80, 80, 120), 'accent': (180, 180, 220)},# Gray/space
        }
        DEFAULT_COLOR = {'bg': (80, 80, 100), 'accent': (150, 150, 200)}
        
        # Safety check - ensure screen is valid
        if not self._screen or not self._running:
            return
        
        try:
            # Semi-transparent dark overlay
            overlay = pygame.Surface((self.width, self.height), pygame.SRCALPHA)
            overlay.fill((5, 5, 20, 230))
            self._screen.blit(overlay, (0, 0))
        except pygame.error as e:
            print(f"[DISPLAY] Menu overlay surface error: {e}")
            return
        
        # Animated border
        border_glow = int(100 + 50 * pulse)
        pygame.draw.rect(self._screen, (0, border_glow, border_glow // 2), 
                        (20, 20, self.width - 40, self.height - 40), 3)
        
        # Title with arcade styling
        viewer = ms.get('viewer_name', 'Player')
        title_text = f"SELECT YOUR GAME"
        title_lbl = self._fonts['huge'].render(title_text, True, (0, 255, 150))
        title_x = (self.width - title_lbl.get_width()) // 2
        self._screen.blit(title_lbl, (title_x, 40))
        
        # Player name
        player_text = f"Player: {viewer}"
        player_lbl = self._fonts['lg'].render(player_text, True, (255, 255, 100))
        player_x = (self.width - player_lbl.get_width()) // 2
        self._screen.blit(player_lbl, (player_x, 90))
        
        # Timer with urgency colors
        time_remaining = ms.get('time_remaining', 0)
        secs = int(time_remaining)
        if secs < 5:
            timer_color = (255, 50, 50)
            timer_bg = (100, 0, 0)
        elif secs < 10:
            timer_color = (255, 200, 50)
            timer_bg = (80, 60, 0)
        else:
            timer_color = (50, 255, 100)
            timer_bg = (0, 60, 20)
        
        timer_text = f" {secs}s "
        timer_lbl = self._fonts['lg'].render(timer_text, True, timer_color)
        timer_x = (self.width - timer_lbl.get_width()) // 2
        # Timer background
        pygame.draw.rect(self._screen, timer_bg, 
                        (timer_x - 10, 125, timer_lbl.get_width() + 20, 35))
        self._screen.blit(timer_lbl, (timer_x, 128))
        
        # Games as CARDS in a grid with SCROLLING
        games = ms.get('games', [])
        selected = ms.get('selected_index', 0)
        
        # Card dimensions - smaller to fit more
        card_width = 165
        card_height = 70
        cards_per_row = 5
        row_height = card_height + 10
        
        # Calculate visible area - leave room for metadata panel (200px) + controls (70px)
        menu_top = 175
        menu_bottom = self.height - 280  # Leave room for metadata + controls
        visible_height = menu_bottom - menu_top
        visible_rows = max(1, visible_height // row_height)
        
        total_rows = (len(games) + cards_per_row - 1) // cards_per_row
        
        # Calculate scroll offset to keep selected visible
        selected_row = selected // cards_per_row
        
        # Determine which row to start displaying
        scroll_row = max(0, min(selected_row - visible_rows // 2, total_rows - visible_rows))
        scroll_row = max(0, scroll_row)  # Clamp to 0
        
        start_x = (self.width - (cards_per_row * (card_width + 10) - 10)) // 2
        
        # Draw visible cards only
        for i, game in enumerate(games):
            row = i // cards_per_row
            col = i % cards_per_row
            
            # Skip rows above visible area
            if row < scroll_row:
                continue
            # Stop if below visible area  
            if row >= scroll_row + visible_rows + 1:
                break
            
            game_name = game.get('name', game.get('display_name', game.get('id', f'Game {i}')))
            
            # Get color scheme for this game
            colors = GAME_COLORS.get(game_name, DEFAULT_COLOR)
            
            card_x = start_x + col * (card_width + 10)
            card_y = menu_top + (row - scroll_row) * row_height
            
            # Skip if card would be below visible area
            if card_y + card_height > menu_bottom:
                continue
            
            is_selected = (i == selected)
            
            # Card surface
            card = pygame.Surface((card_width, card_height), pygame.SRCALPHA)
            
            if is_selected:
                # Selected card - bright with animated glow
                glow = int(200 + 55 * pulse)
                card.fill((*colors['bg'], 255))
                # Glowing border
                pygame.draw.rect(card, (glow, glow, 255), (0, 0, card_width, card_height), 3)
            else:
                # Unselected card - dimmed
                card.fill((*[c // 2 for c in colors['bg']], 180))
                pygame.draw.rect(card, (60, 60, 80), (0, 0, card_width, card_height), 2)
            
            self._screen.blit(card, (card_x, card_y))
            
            # Game number in corner
            num_text = f"{i + 1}"
            num_color = (255, 255, 255) if is_selected else (150, 150, 150)
            num_lbl = self._fonts['xs'].render(num_text, True, num_color)
            self._screen.blit(num_lbl, (card_x + 5, card_y + 3))
            
            # Game name centered on card (truncate if needed)
            display_name = game_name[:14] + '..' if len(game_name) > 16 else game_name
            name_color = (255, 255, 255) if is_selected else (180, 180, 180)
            name_lbl = self._fonts['sm'].render(display_name, True, name_color)
            name_x = card_x + (card_width - name_lbl.get_width()) // 2
            name_y = card_y + (card_height - name_lbl.get_height()) // 2
            self._screen.blit(name_lbl, (name_x, name_y))
            
            # Highlight effect for selected
            if is_selected:
                highlight = pygame.Surface((card_width - 6, 2), pygame.SRCALPHA)
                highlight.fill((255, 255, 255, int(150 * pulse)))
                self._screen.blit(highlight, (card_x + 3, card_y + card_height - 5))
        
        # Scroll indicators
        if scroll_row > 0:
            # Up arrow indicator
            up_text = f"▲ {scroll_row} more above"
            up_lbl = self._fonts['sm'].render(up_text, True, (150, 200, 255))
            up_x = (self.width - up_lbl.get_width()) // 2
            self._screen.blit(up_lbl, (up_x, menu_top - 20))
        
        if scroll_row + visible_rows < total_rows:
            # Down arrow indicator
            remaining = total_rows - scroll_row - visible_rows
            down_text = f"▼ {remaining} more below"
            down_lbl = self._fonts['sm'].render(down_text, True, (150, 200, 255))
            down_x = (self.width - down_lbl.get_width()) // 2
            self._screen.blit(down_lbl, (down_x, menu_bottom + 2))
        
        # Page indicator
        page_text = f"Game {selected + 1} of {len(games)}"
        page_lbl = self._fonts['sm'].render(page_text, True, (200, 200, 200))
        page_x = self.width - page_lbl.get_width() - 40
        self._screen.blit(page_lbl, (page_x, 130))
        
        # =====================================================
        # METADATA PANEL - Show detailed info for selected game
        # =====================================================
        selected_meta = ms.get('selected_game', {})
        if selected_meta:
            # Panel position - fixed location below the game grid
            panel_y = menu_bottom + 20
            panel_height = 160  # Fixed height for metadata panel
            
            # Semi-transparent panel background
            panel = pygame.Surface((self.width - 60, panel_height), pygame.SRCALPHA)
            panel.fill((20, 25, 40, 240))
            pygame.draw.rect(panel, (60, 80, 120), (0, 0, self.width - 60, panel_height), 2)
            self._screen.blit(panel, (30, panel_y))
            
            # Game title with year
            year_str = f" ({selected_meta.get('year', '')})" if selected_meta.get('year') else ""
            title_text = f"{selected_meta.get('name', 'Unknown')}{year_str}"
            title_lbl = self._fonts['lg'].render(title_text, True, (255, 220, 100))
            self._screen.blit(title_lbl, (45, panel_y + 10))
            
            # Developer and platform
            dev = selected_meta.get('developer', 'Unknown')
            platform = selected_meta.get('platform', '')
            dev_text = f"Developer: {dev}" + (f"  •  Platform: {platform}" if platform else "")
            dev_lbl = self._fonts['sm'].render(dev_text, True, (150, 180, 220))
            self._screen.blit(dev_lbl, (45, panel_y + 40))
            
            # Genre and difficulty
            genre = selected_meta.get('genre', 'Unknown')
            difficulty = selected_meta.get('difficulty', 1)
            diff_stars = "★" * difficulty + "☆" * (5 - difficulty)
            genre_text = f"Genre: {genre}  •  Difficulty: {diff_stars}"
            genre_lbl = self._fonts['sm'].render(genre_text, True, (150, 180, 220))
            self._screen.blit(genre_lbl, (45, panel_y + 60))
            
            # Description
            desc = selected_meta.get('description', '')
            if desc:
                desc_lbl = self._fonts['sm'].render(desc, True, (200, 200, 200))
                self._screen.blit(desc_lbl, (45, panel_y + 85))
            
            # Records section - separator line
            pygame.draw.line(self._screen, (60, 80, 120), 
                           (50, panel_y + 110), (self.width - 70, panel_y + 110), 1)
            
            # Record labels with formatting
            human_rec = selected_meta.get('human_record', 0)
            ai_rec = selected_meta.get('ai_record', 0)
            human_fmt = f"{human_rec:,.0f}" if human_rec >= 0 else f"{human_rec:,.0f}"
            ai_fmt = f"{ai_rec:,.0f}" if ai_rec >= 0 else f"{ai_rec:,.0f}"
            
            rec_label = self._fonts['sm'].render("HIGH SCORES:", True, (100, 150, 200))
            self._screen.blit(rec_label, (45, panel_y + 120))
            
            human_text = f"Human Record: {human_fmt}"
            human_lbl = self._fonts['sm'].render(human_text, True, (255, 215, 0))
            self._screen.blit(human_lbl, (200, panel_y + 120))
            
            ai_text = f"AI Best: {ai_fmt}"
            ai_lbl = self._fonts['sm'].render(ai_text, True, (0, 200, 255))
            self._screen.blit(ai_lbl, (450, panel_y + 120))
        
        # Controls hint at bottom - more prominent
        hint_bg = pygame.Surface((600, 50), pygame.SRCALPHA)
        hint_bg.fill((0, 50, 80, 200))
        hint_x = (self.width - 600) // 2
        hint_y = self.height - 70
        self._screen.blit(hint_bg, (hint_x, hint_y))
        
        # W/S or arrows - updated controls
        ctrl_text = "[ W/S ] Navigate    [ A/D ] Row Skip    [ SPACE ] Select"
        ctrl_lbl = self._fonts['sm'].render(ctrl_text, True, (150, 255, 200))
        ctrl_x = (self.width - ctrl_lbl.get_width()) // 2
        self._screen.blit(ctrl_lbl, (ctrl_x, hint_y + 15))

    def _draw_control_mesh(self, pulse: float, slow_pulse: float):
        """
        Draw visual control mesh overlay when user has control.
        
        This is the DEFINITIVE visual connection between user and game:
        - Pulsing corner brackets
        - Edge glow effect
        - Central "USER CONTROL" indicator with live key presses
        - Action ripple effects
        """
        hold_mode_upper = (self._hold_mode or "").upper()
        is_user_control = hold_mode_upper in ("CONTINUOUS", "FREEZE", "TAKEOVER", "HOLD", "SINGLE")
        
        if not is_user_control:
            return  # No mesh when AI is in control
        
        game_y = 50
        
        # Color scheme based on mode
        if hold_mode_upper in ("FREEZE", "HOLD", "SINGLE", "TAKEOVER"):
            # HOLD mode - amber/gold (time frozen)
            primary = (255, int(140 + 80 * pulse), 0)
            secondary = (255, 100, 0)
            glow_alpha = int(80 + 60 * pulse)
            mode_text = "⏸ HOLD - TIME FROZEN"
        else:
            # CONTINUOUS mode - cyan/teal (flowing control)
            primary = (0, int(220 + 35 * pulse), int(200 + 55 * pulse))
            secondary = (0, 180, 255)
            glow_alpha = int(60 + 40 * pulse)
            mode_text = "▶ CONTINUOUS CONTROL"
        
        # === CORNER BRACKETS - visual anchors ===
        bracket_size = 50
        bracket_thick = 4
        corners = [
            (0, game_y),                                    # top-left
            (self.GAME_W - bracket_size, game_y),           # top-right
            (0, game_y + self.GAME_H - bracket_size),       # bottom-left
            (self.GAME_W - bracket_size, game_y + self.GAME_H - bracket_size),  # bottom-right
        ]
        
        for i, (cx, cy) in enumerate(corners):
            # Horizontal bracket
            if i % 2 == 0:  # left corners
                pygame.draw.rect(self._screen, primary, (cx, cy, bracket_size, bracket_thick))
                pygame.draw.rect(self._screen, primary, (cx, cy, bracket_thick, bracket_size))
            else:  # right corners
                pygame.draw.rect(self._screen, primary, (cx, cy, bracket_size, bracket_thick))
                pygame.draw.rect(self._screen, primary, (cx + bracket_size - bracket_thick, cy, bracket_thick, bracket_size))
        
        # === PULSING EDGE GLOW - game border ===
        glow_surf = pygame.Surface((self.GAME_W, self.GAME_H), pygame.SRCALPHA)
        
        # Top edge glow
        for i in range(8):
            alpha = glow_alpha - i * 10
            if alpha > 0:
                pygame.draw.rect(glow_surf, (*secondary, alpha), (0, i, self.GAME_W, 1))
        # Bottom edge glow
        for i in range(8):
            alpha = glow_alpha - i * 10
            if alpha > 0:
                pygame.draw.rect(glow_surf, (*secondary, alpha), (0, self.GAME_H - i - 1, self.GAME_W, 1))
        # Left edge glow
        for i in range(8):
            alpha = glow_alpha - i * 10
            if alpha > 0:
                pygame.draw.rect(glow_surf, (*secondary, alpha), (i, 0, 1, self.GAME_H))
        # Right edge glow
        for i in range(8):
            alpha = glow_alpha - i * 10
            if alpha > 0:
                pygame.draw.rect(glow_surf, (*secondary, alpha), (self.GAME_W - i - 1, 0, 1, self.GAME_H))
        
        self._screen.blit(glow_surf, (0, game_y))
        
        # === CENTRAL USER CONTROL INDICATOR ===
        indicator_w = 280
        indicator_h = 70
        indicator_x = (self.GAME_W - indicator_w) // 2
        indicator_y = game_y + 60
        
        # Background with gradient
        indicator_surf = pygame.Surface((indicator_w, indicator_h), pygame.SRCALPHA)
        indicator_surf.fill((0, 0, 0, int(200 * (0.8 + 0.2 * slow_pulse))))
        
        # Border with pulse
        pygame.draw.rect(indicator_surf, primary, (0, 0, indicator_w, indicator_h), 3, border_radius=10)
        
        # Mode text
        mode_lbl = self._fonts['md'].render(mode_text, True, primary)
        mode_x = (indicator_w - mode_lbl.get_width()) // 2
        indicator_surf.blit(mode_lbl, (mode_x, 8))
        
        # Player name
        player_name = self._viewer_name or "Local Player"
        player_lbl = self._fonts['sm'].render(f"🎮 {player_name}", True, (255, 255, 255))
        player_x = (indicator_w - player_lbl.get_width()) // 2
        indicator_surf.blit(player_lbl, (player_x, 32))
        
        # Key input visualization
        keys_text = "WASD / ↑↓←→ / SPACE"
        keys_lbl = self._fonts['xs'].render(keys_text, True, (150, 150, 180))
        keys_x = (indicator_w - keys_lbl.get_width()) // 2
        indicator_surf.blit(keys_lbl, (keys_x, 52))
        
        self._screen.blit(indicator_surf, (indicator_x, indicator_y))
        
        # === LIVE INPUT DISPLAY - shows current/last key press ===
        action_age = time.time() - self._last_action_time
        if action_age < 1.5 and self._last_action_text:
            # Show the action that was just taken with a ripple effect
            ripple_alpha = int(255 * max(0, 1.0 - action_age / 1.5))
            ripple_size = int(100 + action_age * 80)
            
            # Central ripple circle
            center_x = self.GAME_W // 2
            center_y = game_y + self.GAME_H // 2 + 50
            
            ripple_surf = pygame.Surface((ripple_size * 2, ripple_size * 2), pygame.SRCALPHA)
            pygame.draw.circle(ripple_surf, (*secondary, ripple_alpha // 3), (ripple_size, ripple_size), ripple_size, 3)
            self._screen.blit(ripple_surf, (center_x - ripple_size, center_y - ripple_size))
            
            # Action text
            action_lbl = self._fonts['lg'].render(self._last_action_text, True, primary)
            action_x = center_x - action_lbl.get_width() // 2
            action_y = center_y - 15
            
            # Shadow
            shadow_lbl = self._fonts['lg'].render(self._last_action_text, True, (0, 0, 0))
            self._screen.blit(shadow_lbl, (action_x + 2, action_y + 2))
            self._screen.blit(action_lbl, (action_x, action_y))

    def _draw_state_vector(self):
        """Draw observation state vector (for vector obs only)."""
        if not self._state_labels or self._current_obs is None:
            return
        if len(self._current_obs.shape) != 1:
            return
        
        state_y = 50 + self.GAME_H + 10
        pygame.draw.rect(self._screen, (15, 15, 25), (10, state_y - 5, self.GAME_W - 20, 100), border_radius=6)
        
        self._screen.blit(self._fonts['sm'].render("OBSERVATION VECTOR", True, (100, 150, 200)), (20, state_y))
        
        obs = self._current_obs.flatten()[:16]
        labels = self._state_labels[:len(obs)]
        
        for i, (label, val) in enumerate(zip(labels, obs)):
            col = i % 4
            row = i // 4
            x = 20 + col * 145
            y = state_y + 20 + row * 35
            
            if abs(val) < 0.3:
                color = (100, 200, 100)
            elif abs(val) < 0.7:
                color = (200, 200, 100)
            else:
                color = (255, 120, 80)
            
            self._screen.blit(self._fonts['xs'].render(label, True, (120, 120, 140)), (x, y))
            self._screen.blit(self._fonts['sm'].render(f"{val:+.3f}", True, color), (x, y + 12))
    
    def _draw_action_sequence(self):
        """Draw action history sequence."""
        seq_y = 50 + self.GAME_H + 115
        pygame.draw.rect(self._screen, (10, 20, 15), (10, seq_y, self.GAME_W - 20, 70), border_radius=6)
        
        self._screen.blit(self._fonts['sm'].render("ACTION SEQUENCE", True, (80, 180, 120)), (20, seq_y + 5))
        
        seq_text = self.wealth.get_action_sequence(self._action_labels, 10)
        self._screen.blit(self._fonts['xs'].render(seq_text, True, (150, 200, 150)), (20, seq_y + 25))
        
        cum_reward = self.wealth.get_cumulative_reward()
        avg_entropy = self.wealth.get_avg_entropy()
        metrics_txt = f"Σ Reward: {cum_reward:+.1f}  |  Entropy: {avg_entropy:.3f}  |  Overrides: {self.wealth.total_overrides}"
        self._screen.blit(self._fonts['xs'].render(metrics_txt, True, (100, 140, 100)), (20, seq_y + 45))
    
    def _draw_right_panel(self, pulse: float, slow_pulse: float):
        """Draw the right info panel."""
        panel_x = self.GAME_W + 10
        hold_mode_upper = (self._hold_mode or "").upper()
        is_held = hold_mode_upper in ("CONTINUOUS", "FREEZE", "TAKEOVER")
        
        pygame.draw.rect(self._screen, (8, 8, 15), (self.GAME_W, 0, self.PANEL_W + 20, self.height))
        pygame.draw.line(self._screen, (0, 200, 150), (self.GAME_W, 0), (self.GAME_W, self.height), 2)
        
        hdr = self._fonts['lg'].render("CASCADE-LATTICE", True, (0, 255, 150))
        self._screen.blit(hdr, (panel_x + 20, 15))
        
        # Control status - normalize to uppercase for comparison
        hold_mode_upper = (self._hold_mode or "").upper()
        
        if hold_mode_upper == "FREEZE":
            status_color = (int(200 + 55 * pulse), 50, 50)
            status_label = "◆ HOLD-FREEZE: TIME STOPPED"
            status_detail = "Inspect AI • WASD to override • ESC to release"
        elif hold_mode_upper == "TAKEOVER":
            status_color = (int(200 + 55 * pulse), int(150 + 50 * pulse), 0)
            status_label = "◆ HOLD-SINGLE: HUMAN ACTION"
            status_detail = "One action, then returns to AI"
        elif hold_mode_upper == "CONTINUOUS":
            status_color = (0, int(200 + 55 * pulse), int(255 * pulse))
            status_label = f"🎮 VIEWER: {self._viewer_name or 'Player'}"
            status_detail = f"Time: {self._viewer_timer:.1f}s • Actions: {self._viewer_actions}"
        else:
            status_color = (0, 255, int(150 + 105 * slow_pulse))
            status_label = "◆ AI SOVEREIGNTY"
            status_detail = "Click PLAY in extension overlay to take control"
        
        self._screen.blit(self._fonts['md'].render(status_label, True, status_color), (panel_x + 20, 50))
        self._screen.blit(self._fonts['xs'].render(status_detail, True, (140, 140, 160)), (panel_x + 20, 70))
        
        self._draw_decision_matrix(panel_x, pulse)
        self._draw_provenance(panel_x)
        self._draw_queue(panel_x)
        self._draw_stats(panel_x)
        self._draw_controls(panel_x, is_held)
        
        footer_y = self.height - 25
        pygame.draw.line(self._screen, (30, 30, 50), (panel_x, footer_y - 5), (self.width - 10, footer_y - 5))
        legend = "pip install cascade-lattice • cascade.hold | cascade.store | cascade.core.provenance"
        self._screen.blit(self._fonts['xs'].render(legend, True, (70, 70, 90)), (panel_x + 20, footer_y))
    
    def _draw_decision_matrix(self, panel_x: int, pulse: float):
        """Draw action probability matrix."""
        matrix_y = 95
        self._screen.blit(self._fonts['title'].render("DECISION MATRIX", True, (0, 180, 255)), (panel_x + 20, matrix_y))
        
        if self._action_probs is None:
            return
        
        start_y = matrix_y + 30
        max_actions = min(len(self._action_probs), 8)
        row_h = 45 if max_actions <= 4 else 35
        
        max_idx = np.argmax(self._action_probs)
        
        for i in range(max_actions):
            p = self._action_probs[i] if i < len(self._action_probs) else 0
            label = self._action_labels[i] if i < len(self._action_labels) else f"A{i}"
            
            y = start_y + i * row_h
            is_winner = (i == max_idx)
            is_chosen = (i == self._chosen_action)
            
            bg_color = (30, 35, 20) if is_winner else (20, 25, 35)
            pygame.draw.rect(self._screen, bg_color, (panel_x + 15, y, self.PANEL_W - 30, row_h - 5), border_radius=4)
            
            if is_chosen and self._hold_mode:
                lbl_color = (0, 255, 255)
                prefix = "→ "
            elif is_winner:
                lbl_color = (int(200 + 55 * pulse), 220, 0)
                prefix = "★ "
            else:
                lbl_color = (160, 160, 170)
                prefix = "  "
            
            self._screen.blit(self._fonts['sm'].render(f"{prefix}{label}", True, lbl_color), (panel_x + 25, y + 5))
            
            bar_w = 200
            bar_x = panel_x + 25
            bar_y = y + 22
            pygame.draw.rect(self._screen, (25, 25, 40), (bar_x, bar_y, bar_w, 10), border_radius=3)
            
            if is_chosen and self._hold_mode:
                fill_c = (0, 180, 220)
            elif is_winner:
                fill_c = (int(180 + 55 * pulse), int(140 + 50 * pulse), 0)
            else:
                fill_c = (50, 80, 110)
            pygame.draw.rect(self._screen, fill_c, (bar_x, bar_y, int(p * bar_w), 10), border_radius=3)
            
            self._screen.blit(self._fonts['sm'].render(f"{p:.1%}", True, (200, 200, 200)), (bar_x + bar_w + 10, bar_y - 2))
    
    def _draw_provenance(self, panel_x: int):
        """Draw provenance chain info."""
        prov_y = 420
        pygame.draw.rect(self._screen, (12, 18, 25), (panel_x + 15, prov_y, self.PANEL_W - 30, 140), border_radius=6)
        pygame.draw.rect(self._screen, (0, 150, 200), (panel_x + 15, prov_y, self.PANEL_W - 30, 140), 1, border_radius=6)
        
        self._screen.blit(self._fonts['title'].render("PROVENANCE CHAIN", True, (0, 180, 220)), (panel_x + 25, prov_y + 8))
        
        merkle_short = (self._merkle_root[:40] + "...") if self._merkle_root else "initializing..."
        self._screen.blit(self._fonts['xs'].render("Merkle Root:", True, (100, 120, 140)), (panel_x + 25, prov_y + 35))
        self._screen.blit(self._fonts['sm'].render(merkle_short, True, (0, 255, 150)), (panel_x + 25, prov_y + 48))
        
        parent_short = (self._parent_merkle[:32] + "...") if self._parent_merkle else "GENESIS"
        self._screen.blit(self._fonts['xs'].render("Parent:", True, (100, 120, 140)), (panel_x + 25, prov_y + 68))
        self._screen.blit(self._fonts['xs'].render(parent_short, True, (80, 140, 100)), (panel_x + 70, prov_y + 68))
        
        self._screen.blit(self._fonts['sm'].render(f"Value Estimate: {self._value_estimate:+.4f}", True, (180, 200, 255)), (panel_x + 25, prov_y + 88))
        
        avg_entropy = self.wealth.get_avg_entropy()
        entropy_color = (100, 200, 100) if avg_entropy < 0.5 else (200, 200, 100) if avg_entropy < 1.0 else (255, 150, 100)
        self._screen.blit(self._fonts['sm'].render(f"Decision Entropy: {avg_entropy:.4f}", True, entropy_color), (panel_x + 25, prov_y + 108))
        
        genesis_short = (self._genesis_root[:24] + "...") if self._genesis_root else "N/A"
        self._screen.blit(self._fonts['xs'].render(f"Genesis: {genesis_short}", True, (60, 80, 100)), (panel_x + 25, prov_y + 125))
    
    def _draw_queue(self, panel_x: int):
        """Draw viewer queue."""
        queue_y = 570
        pygame.draw.rect(self._screen, (15, 10, 20), (panel_x + 15, queue_y, self.PANEL_W - 30, 100), border_radius=6)
        
        queue_count = len(self._queue)
        self._screen.blit(self._fonts['md'].render(f"QUEUE ({queue_count})", True, (200, 100, 200)), (panel_x + 25, queue_y + 8))
        
        if self._queue:
            for i, viewer in enumerate(self._queue[:4]):
                name = viewer.get('name', 'Anonymous')[:20]
                bits = viewer.get('bits', 0)
                y = queue_y + 30 + i * 16
                self._screen.blit(self._fonts['xs'].render(f"{i+1}. {name}", True, (200, 150, 200)), (panel_x + 30, y))
                self._screen.blit(self._fonts['xs'].render(f"{bits} bits", True, (100, 100, 120)), (panel_x + 200, y))
        else:
            self._screen.blit(self._fonts['xs'].render("Click PLAY in extension!", True, (100, 100, 120)), (panel_x + 30, queue_y + 35))
    
    def _draw_stats(self, panel_x: int):
        """Draw session statistics."""
        stats_y = 680
        pygame.draw.rect(self._screen, (15, 12, 20), (panel_x + 15, stats_y, self.PANEL_W - 30, 70), border_radius=6)
        
        self._screen.blit(self._fonts['md'].render("SESSION", True, (180, 140, 200)), (panel_x + 25, stats_y + 8))
        
        session_time = int(time.time() - self._session_start)
        bits = self.wealth.total_bits_earned
        
        stats_lines = [
            f"Episodes: {self._total_episodes}  |  Overrides: {self.wealth.total_overrides}  |  Bits: {bits}",
            f"Runtime: {session_time//60}m {session_time%60}s  |  Avg Value: {self.wealth.get_avg_value():+.3f}",
        ]
        for i, s in enumerate(stats_lines):
            self._screen.blit(self._fonts['xs'].render(s, True, (140, 130, 160)), (panel_x + 25, stats_y + 28 + i * 16))
    
    def _draw_controls(self, panel_x: int, is_held: bool):
        """Draw control hints."""
        ctrl_y = 760
        pygame.draw.rect(self._screen, (20, 15, 10), (panel_x + 15, ctrl_y, self.PANEL_W - 30, 100), border_radius=6)
        
        self._screen.blit(self._fonts['md'].render("CONTROLS", True, (255, 180, 80)), (panel_x + 25, ctrl_y + 8))
        
        if is_held:
            controls = [
                "WASD / Arrows = Control",
                "SPACE = Fire/Action",
                "ESC = Release HOLD",
            ]
            ctrl_color = (200, 180, 150)
        else:
            controls = [
                "Use Extension Overlay to Play",
                "WASD + SPACE for controls",
                "Click PLAY button to start",
            ]
            ctrl_color = (150, 150, 170)
        
        for i, c in enumerate(controls):
            self._screen.blit(self._fonts['sm'].render(c, True, ctrl_color), (panel_x + 25, ctrl_y + 30 + i * 22))
    
    def run_loop(self):
        """Run the display loop (blocking)."""
        self.start()
        while self._running:
            if not self.render():
                break
        self.stop()
