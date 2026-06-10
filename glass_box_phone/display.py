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
    
    # Game display - BIGGER! Takes most of the screen
    GAME_W, GAME_H = 850, 640
    
    # Right panel for cascade-lattice data
    PANEL_W = WIN_W - GAME_W - 20
    
    def __init__(
        self,
        width: int = 1400,
        height: int = 920,
        title: str = "🎰 CASCADE-LATTICE ARCADE",
        target_fps: int = 30,
    ):
        self.width = width
        self.height = height
        self.title = title
        self.target_fps = target_fps
        
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
        
        # Game selection menu
        self._menu_state: Optional[Dict] = None  # Menu data when open
        
        # Stats
        self._episode_reward: float = 0.0
        self._episode_steps: int = 0
        self._total_episodes: int = 0
        self._session_start: float = time.time()
        
        # Wealth tracker
        self.wealth = InformationalWealth()
        
        # Fonts
        self._fonts: Dict[str, pygame.font.Font] = {}
    
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
        
        self._fonts = {
            'xs': pygame.font.SysFont("Consolas", 10),
            'sm': pygame.font.SysFont("Consolas", 12),
            'md': pygame.font.SysFont("Consolas", 16),
            'lg': pygame.font.SysFont("Consolas", 28, bold=True),
            'title': pygame.font.SysFont("Consolas", 22, bold=True),
            'huge': pygame.font.SysFont("Consolas", 48, bold=True),
        }
        
        # Render initial frame so ffmpeg sees something (not X11 background)
        self._screen.fill((2, 2, 8))  # Dark background
        title = self._fonts['huge'].render("GLASS BOX ARCADE", True, (0, 255, 200))
        subtitle = self._fonts['lg'].render("Initializing...", True, (150, 150, 150))
        self._screen.blit(title, (self.width//2 - title.get_width()//2, self.height//2 - 50))
        self._screen.blit(subtitle, (self.width//2 - subtitle.get_width()//2, self.height//2 + 20))
        pygame.display.flip()
        
        print(f"[DISPLAY] Game window opened: {self.width}x{self.height}")
    
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
        events = {"quit": False, "key": None, "hold_key": None}
        
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
        
        # Safety check - don't render if pygame quit
        if not self._running or self._screen is None:
            return False
        
        try:
            pulse = (math.sin(time.time() * 6) + 1) / 2
            slow_pulse = (math.sin(time.time() * 2) + 1) / 2
            
            self._screen.fill((2, 2, 8))
        except pygame.error:
            self._running = False
            return False
        
        self._draw_game_view(pulse)
        self._draw_control_mesh(pulse, slow_pulse)  # NEW: Visual mesh for user control
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
