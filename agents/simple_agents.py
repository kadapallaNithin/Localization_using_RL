from .base import Agent
import sys
import numpy as np
from state_builder import extract_state_components

# ── raw single-keypress (Linux/macOS) ────────────────────────────────────────
def _getch():
    import tty, termios
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        ch = sys.stdin.read(1)
        if ch == '\x1b':
            nxt = sys.stdin.read(1)
            if nxt == '[':
                return '\x1b[' + sys.stdin.read(1)   # arrow key
            return '\x1b'
        return ch
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)

# direction index → (arrow-symbol, compass-label)
_DIR_LABEL = {0:'→E', 1:'↗NE', 2:'↑N', 3:'↖NW', 4:'←W', 5:'↙SW', 6:'↓S', 7:'↘SE'}

# key → direction index
_KEY_DIR = {
    'd': 0, '\x1b[C': 0,   # East     (D / →)
    'e': 1,                  # NE       (E)
    'w': 2, '\x1b[A': 2,   # North    (W / ↑)
    'q': 3,                  # NW       (Q)
    'a': 4, '\x1b[D': 4,   # West     (A / ←)
    'z': 5,                  # SW       (Z)
    's': 6, '\x1b[B': 6,   # South    (S / ↓)
    'c': 7,                  # SE       (C)
}

class ManualAgent(Agent):
    """
    Keyboard-driven agent.
      Movement : WASD / arrow keys (4 cardinal) + Q E Z C (4 diagonals)
      Speed     : 1–5  (default 2)
      No-op     : Space or F
      Quit      : X or Ctrl-C
    """
    def __init__(self):
        self._speed = 1   # index 0–4

    def act(self, state):
        self._print_hud(state)
        while True:
            key = _getch()
            if key in ('\x03', 'x'):          # Ctrl-C or x → quit
                raise KeyboardInterrupt
            if key in ('1','2','3','4','5'):   # change speed
                self._speed = int(key) - 1
                spd_bar = ''.join('█' if i == self._speed else '░' for i in range(5))
                print(f'\r  Speed [{spd_bar}] {self._speed+1}/5   ', end='', flush=True)
                continue
            if key in (' ', 'f'):             # no-op
                print(f'\r  · no-op                      ', end='', flush=True)
                return 40
            if key in _KEY_DIR:
                direction = _KEY_DIR[key]
                action = self._speed * 8 + direction
                print(f'\r  {_DIR_LABEL[direction]}  spd={self._speed+1}  act={action}    ',
                      end='', flush=True)
                return action

    def _print_hud(self, state):
        sig = float(state[0]) if len(state) > 0 else 0.0
        spd_bar = ''.join('█' if i == self._speed else '░' for i in range(5))
        bar_len = int(sig * 20)
        sig_bar = '█' * bar_len + '░' * (20 - bar_len)
        print(f'\n  Signal [{sig_bar}] {sig:.4f}')
        print(f'  Speed  [{spd_bar}] {self._speed+1}/5')
        print('  Move: WASD/arrows · diag: QEZC · speed: 1-5 · noop: Space · quit: x')
        print('  > ', end='', flush=True)

    def update(self, state, action, reward, next_state, done):
        sig = float(next_state[0]) if len(next_state) > 0 else 0.0
        status = '✓ FOUND' if done else f'reward={reward:+.2f}'
        print(f'\r  {status}  signal={sig:.4f}                    ', end='', flush=True)

    def save(self, *args, **kwargs): pass
    def load(self, *args, **kwargs): pass

class UniformSearchAgent(Agent):
    def __init__(self, area_size=200, sweep_step=3, path=None):
        self.area_size = area_size
        self.sweep_step = sweep_step
        self.direction = 1      # +1 = moving right, -1 = moving left
        self.vertical_dir = 1   # +1 = going up, -1 = going down
        self.position = None

    def set_uav_position(self, pos):
        self.position = list(pos)

    def reset(self, start_pos):
        self.position = list(start_pos)
        self.direction = 1
        self.vertical_dir = 1   # start moving upward
        return self.position

    def act(self, state=None, y=None):
        if y is None:
            if self.position is None:
                return 40
            x, y = self.position
        else:
            x = state

        # --- Horizontal Sweep ---
        if self.direction == 1:  # moving right
            if x < self.area_size - self.sweep_step:
                target = (x + self.sweep_step, y)
            else:
                if not self._within_vertical_bounds(y + self.vertical_dir * self.sweep_step):
                    # flip vertical direction and start opposite sweep
                    self.vertical_dir *= -1
                target = (x, y + self.vertical_dir * self.sweep_step)
                self.direction = -1

        else:  # moving left
            if x > self.sweep_step:
                target = (x - self.sweep_step, y)
            else:
                if not self._within_vertical_bounds(y + self.vertical_dir * self.sweep_step):
                    # flip vertical direction and start opposite sweep
                    self.vertical_dir *= -1
                target = (x, y + self.vertical_dir * self.sweep_step)
                self.direction = 1

        return self._get_action_to_target(x, y, target)

    def _within_vertical_bounds(self, y):
        return 0 <= y <= self.area_size

    def _get_action_to_target(self, x, y, target):
        tx, ty = target
        dx, dy = tx - x, ty - y
        angle = np.arctan2(dy, dx)
        direction = int((angle % (2 * np.pi)) / (np.pi / 4)) % 8
        speed = 2  # medium speed (0–2)
        action = speed * 8 + direction
        return action

    def update(self, *args):
        pass

    def save(self, *args, **kwargs):
        pass

    def load(self, *args, **kwargs):
        pass

# ================= GRADIENT-BASED (HILL CLIMBING) AGENT =================
class GradientBasedAgent(Agent):
    def __init__(self, action_dim, n_speed_levels=1, speed_factor=1.0, state_builder_type="signal_action_quadrant", path=None):
        """
        Hill climbing: move in direction that increases signal intensity.
        Args:
            action_dim (int): total number of actions (8 × n_speed_levels)
            n_speed_levels (int): number of speed levels
            state_builder_type (str): type of state builder (for proper state component extraction)
        """
        self.action_dim = action_dim
        self.n_speed_levels = n_speed_levels
        self.speed_factor = speed_factor
        self.state_builder_type = state_builder_type
        self.prev_pos = None
        self.prev_signal = None
        self.last_action = None

    def act(self, state):
        """
        Greedy hill climbing using signal gradient from state.
        Uses extract_state_components() to properly extract signal values 
        and last action regardless of state structure.
        Returns action = speed * 8 + direction.
        """
        # Basic safety check
        if len(state) < 2:
            return np.random.randint(self.action_dim)

        # Extract state components using the getter function
        prev_signal, curr_signal, last_action = extract_state_components(
            state, self.state_builder_type
        )

        # Continue in same direction if signal improved
        # print(f'@ {curr_signal:.4f}, {prev_signal:.4f} action:{last_action}')
        if curr_signal > prev_signal:
            return int(last_action)

        # Otherwise, explore: pick a new random direction and speed
        speed = np.random.randint(self.n_speed_levels)
        direction = np.random.randint(8)
        action = speed * 8 + direction
        self.last_action = action
        # print('act', action)
        return action

    def update(self, state, action, reward, next_state, done):
        pass  # No learning needed for this agent

    def save(self, filename=None, **kwargs):
        return super().save(filename, **kwargs)

    def load(self, filename=None, **kwargs):
        return super().load(filename, **kwargs)

class GradWrapAgent(Agent):
    def __init__(self, agent):
        super().__init__()
        self.agent = agent

    def act(self, state):
        return self.agent.act(state)

    def reset(self):
        return self.agent.reset()

    def update(self, *args):
        return self.agent.update(*args)

    def maybe_learn(self, *args, **kwargs):
        return self.agent.maybe_learn(*args, **kwargs)
