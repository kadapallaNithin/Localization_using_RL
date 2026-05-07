import json

try:
    import gymnasium as gym
    from gymnasium import spaces
except ModuleNotFoundError:
    class _FallbackEnv:
        def reset(self, seed=None):
            if seed is not None:
                np.random.seed(seed)

        def close(self):
            pass

    class _FallbackDiscrete:
        def __init__(self, n):
            self.n = n

        def sample(self):
            return np.random.randint(self.n)

    class _FallbackBox:
        def __init__(self, low, high, dtype=None):
            self.low = np.array(low, dtype=dtype)
            self.high = np.array(high, dtype=dtype)
            self.dtype = dtype
            self.shape = self.low.shape

    class _FallbackGym:
        Env = _FallbackEnv

    class _FallbackSpaces:
        Discrete = _FallbackDiscrete
        Box = _FallbackBox

    gym = _FallbackGym()
    spaces = _FallbackSpaces()
from abc import ABC, abstractmethod
from collections import deque
# from gym import spaces
import matplotlib.pyplot as plt
import numpy as np

from rewards import RewardFunction
from sensors import RadiationSensor
from state_builder import StateBuilder

# ========= Abstract UAV Interface =========
class UAVInterface(ABC):
    @abstractmethod
    def reset(self, area_size: int, pos=None, head=None):
        """Reset UAV state (random pos, heading)."""
        pass

    @abstractmethod
    def step(self, action: int):
        """Apply an action to the UAV (update state)."""
        pass


    @abstractmethod
    def time(self):
        """Time on the UAV"""
        pass

    @abstractmethod
    def get_position(self):
        """Return UAV position [x, y]."""
        pass

    @abstractmethod
    def get_head(self):
        """Return UAV heading angle"""
        pass

    @abstractmethod
    def get_quadrant(self, area_size: int):
        """Return quadrant index (0-3)."""
        pass


# ========= Simple Simulated UAV =========
class SimUAV(UAVInterface):
    def __init__(self, speed_factor=5, size=200):
        self.speed_factor = speed_factor
        self.pos = None
        self.head = None
        self.size = size
        self._time = 0
        self.dt = 1

    def reset(self, area_size=None, pos=None, head=None):
        if pos is None:
            self.pos = [np.random.randint(0, self.size), np.random.randint(0, self.size)]
        else:
            self.pos = pos
        if head is None:
            self.head = np.random.randint(0, 8)
        else:
            self.head = head
        self._time = 0
        return self.pos

    def step(self, action: int):
        # no_op action does nothing
        if action == 40:
            return self.pos
        
        speed = action // 8
        direction = action % 8
        v = self.speed_factor * (speed + 1)
        angle = (np.pi / 4) * direction
        x, y = self.pos
        x += v * np.cos(angle) * self.dt
        y += v * np.sin(angle) * self.dt
        self._time += self.dt
        if x < 0:
            x = 0
        elif x > self.size:
            x = self.size
        if y < 0:
            y = 0 
        elif y > self.size:
            y = self.size

        new_pos = [x, y]
        if self.env._collides(new_pos):
            new_pos = self.pos

        self.pos = new_pos
        self.head = angle
        return self.pos

    def time(self):
        return self._time

    def get_position(self):
        return self.pos

    def get_head(self):
        return self.head

    def get_quadrant(self, area_size: int):
        x, y = self.pos
        m = area_size / 2
        if x < m:
            return 0 if y < m else 1
        return 2 if y < m else 3


# ========= Environment =========
def distance_square(a, b):
    x1, y1 = a
    x2, y2 = b
    z = 10
    return (x1 - x2) ** 2 + (y1 - y2) ** 2 + (z - 0) ** 2


class UAVEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 10}

    def __init__(self,
                 uav: UAVInterface,
                 state_builder: StateBuilder,
                 reward_fn: RewardFunction,
                 sensor: RadiationSensor,
                 source_intensity=2e7,
                 size=200,
                 max_steps=200,
                 observer=None,
                 history_length=3,
                 readings=None,
        ):
        super().__init__()
        self.size = size
        self.source_intensity = source_intensity
        # self.sensor_area = 0.035 * 0.035
        self.sensor_area = 0.005 * 0.005
        self.max_steps = max_steps
        self.uav = uav
        self.sensor = sensor
        self.state_builder = state_builder
        self.reward_fn = reward_fn
        self.observer = observer
        print("Observer:", observer)
        self.history_length = history_length
        self.readings = readings
        self.no_op_action = 40  # Define no-op action index
        self.action_space = spaces.Discrete(41)  # 40 movements + 1 no_op
        low, high = state_builder.observation_space(history_length)
        self.observation_space = spaces.Box(low, high, dtype=np.float32)

        # Initialize history tracking (deques with max length)
        # signal_history stores signal measurements: [mt0, mt1, mt2, ..., mt_k]
        self.signal_history = deque(maxlen=history_length + 1)
        # action_history stores actions: [action0, action1, ..., action_k-1]
        self.action_history = deque(maxlen=history_length)

        self.trajectory = []
        self.current_step = 0
        self.source_pos = None
        self.obstacles = []
        self.uav.env = self

        self.fig, self.ax = None, None

    def _pad_history(self):
        # Pad signal history with zeros if not enough measurements
        while len(self.signal_history) < self.history_length + 1:
            self.signal_history.appendleft(0.0)
        # Pad action history with no-op if not enough actions
        while len(self.action_history) < self.history_length:
            self.action_history.appendleft(self.no_op_action)


        # # Pad if necessary (in case we're at the beginning of episode)
        # max_state_size = env.history_length * 2 + 1
        # # Fill with same signal and no_op actions if history is short
        # if len(signal_list) < env.history_length:
        #     if len(signal_list) > 0:
        #         signal = float(signal_list[-1])  # Use most recent signal for padding
        #     else:
        #         signal = 0.0  # Default signal if no history
        #     while len(state) < max_state_size:
        #         if len(state) % 2 == 1:  # Need an action
        #             state.append(signal)
        #         else:  # Need a gradient
        #             state.append(no_op)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if options is None:
            options = {}
        self.source_pos = options.get('source_pos', self._get_random_position())
        self.current_step = 0
        self.uav.reset(pos=options.get('uav_pos'), head=options.get('uav_head'))
        reset_info = {
            'source_pos': self.source_pos,
            'uav_pos': self.uav.get_position(),
            'uav_head': self.uav.get_head(),
        }
        self.trajectory = [{
            "pos": self.uav.get_position(),
            "reward": 0.0
        }]

        # Clear histories
        self.signal_history.clear()
        self.action_history.clear()

        mt0_meas = self.sensor.read(self._true_reading())
        if self.observer is not None:
            # Initialize observer state at first measurement
            self.observer.reset(mt0_meas)
            mt0 = self.observer.x_hat
        else:
            mt0 = mt0_meas
        
        # Add initial signal to history
        self.signal_history.append(mt0)
        
        action = options.get('action', self.action_space.sample())
        reset_info['action'] = action
        self.uav.step(action)
        
        # Add action to history
        self.action_history.append(action)
        
        mt1_meas = self.sensor.read(self._true_reading())
        if self.observer is not None:
            mt1 = self.observer.update(mt1_meas)
        else:
            mt1 = mt1_meas

        if self.readings is not None:
            self.readings['measurements'].append(mt0_meas)
            self.readings['measurements'].append(mt1_meas)
            if self.observer is not None:
                self.readings['observed_readings'].append(mt0)
                self.readings['observed_readings'].append(mt1)
        # Add second signal to history
        self.signal_history.append(mt1)
        
        # Extract values for state builder (support both legacy and new formats)
        signal_list = list(self.signal_history)
        action_list = list(self.action_history)
        # For legacy builders: use first and last signal from history, and last action
        mt0_legacy = signal_list[0] if len(signal_list) > 0 else 0.0
        mt1_legacy = signal_list[-1] if len(signal_list) > 0 else 0.0
        action_legacy = action_list[-1] if len(action_list) > 0 else 0
        self._pad_history()  # Ensure histories are padded to max length
        # Pass both formats: legacy and new (state builder will use what it needs)
        obs = self.state_builder.build(
            mt0_legacy, mt1_legacy, action_legacy, self,
            signal_history=self.signal_history, 
            action_history=self.action_history
        )
        return obs, {'pos': self.uav.get_position(), 'reset':reset_info}

    def step(self, action):
        self.current_step += 1
        
        # Add action to history
        self.action_history.append(action)
        
        mt0_meas = self.sensor.read(self._true_reading())
        self.uav.step(action)
        mt1_meas = self.sensor.read(self._true_reading())

        if self.observer is not None:
            mt0 = self.observer.update(mt0_meas)
            mt1 = self.observer.update(mt1_meas)
        else:
            mt0, mt1 = mt0_meas, mt1_meas

        if self.readings is not None:
            self.readings['measurements'].append(mt0_meas)
            self.readings['measurements'].append(mt1_meas)
            if self.observer is not None:
                self.readings['observed_readings'].append(mt0)
                self.readings['observed_readings'].append(mt1)
            else:
                raise ValueError("Observer must be defined to record observed readings.")
        # Add new signal to history
        self.signal_history.append(mt1)
        
        # Extract values for state builder (support both legacy and new formats)
        signal_list = list(self.signal_history)
        action_list = list(self.action_history)
        # For legacy builders: use first and last signal from history, and last action
        mt0_legacy = signal_list[0] if len(signal_list) > 0 else 0.0
        mt1_legacy = signal_list[-1] if len(signal_list) > 0 else 0.0
        action_legacy = action_list[-1] if len(action_list) > 0 else 0
        
        # Pass both formats: legacy and new (state builder will use what it needs)
        obs = self.state_builder.build(
            mt0_legacy, mt1_legacy, action_legacy, self,
            signal_history=self.signal_history, 
            action_history=self.action_history
        )
        reward, terminated = self.reward_fn.compute(mt0, mt1, self.current_step, self.max_steps)
        self.trajectory.append({
            "pos": self.uav.get_position(),
            "reward": reward
        })
        # print('action', action)
        truncated = (self.current_step >= self.max_steps) and not terminated
        return obs, reward, terminated, truncated, {'pos':self.uav.get_position()}

    def render(
            self,
            mode="human",
            alpha=0.3,
            state=0,
            pause_interval=0.1,
            name='',
            plotting = False,
            belief_visualizer=None,
            belief=None
        ):
        if self.fig is None:
            self.fig, self.ax = plt.subplots()

        self.ax.clear()

        c, marker, alpha = {1:('g','*',1),-1:('r','x',1),0:('g','+',alpha)}[state]
        # traj = np.array(self.trajectory)
        # traj_x = traj[:, 0]
        # traj_y = traj[:, 1]
        traj = self.trajectory

        positions = np.array([t["pos"] for t in traj])
        rewards = np.array([t["reward"] for t in traj])

        traj_x = positions[:, 0]
        traj_y = positions[:, 1]

        def reward_to_rgb(r):
            if r > 0:
                return (0.0, 0.8, 0.0)   # green
            elif r < 0:
                return (0.8, 0.0, 0.0)   # red
            else:
                return (0.0, 0.0, 0.0)   # black

        # draw trajectory segments
        for i in range(1, len(traj_x)):
            color = reward_to_rgb(rewards[i])
            self.ax.plot(
                traj_x[i-1:i+1],
                traj_y[i-1:i+1],
                color=color,
                linewidth=2
            )

        # arrows (optional, same color)
        for i in range(1, len(traj_x), 5):
            color = reward_to_rgb(rewards[i])
            self.ax.annotate(
                '',
                xy=(traj_x[i], traj_y[i]),
                xytext=(traj_x[i-1], traj_y[i-1]),
                arrowprops=dict(
                    color=color,
                    shrink=0.05,
                    width=0,
                    headwidth=5
                )
            )


        # for i in range(1, len(traj_x), 5):  
            # plt.annotate('', xy=(traj_x[i], traj_y[i]), xytext=(traj_x[i-1], traj_y[i-1]),
            #             arrowprops=dict(color="#23788b", shrink=0.05, width=0, headwidth=5))

        if plotting:
            self.ax.scatter(190,190, c='g', marker='+', s=300, label="UAV start position")
            self.ax.scatter(*self.source_pos, c="r", marker="*", s=300, label="Source")
        else:
            self.ax.scatter(*self.uav.get_position(), c=c, marker=marker, s=300, label="UAV", alpha=alpha)
            self.ax.scatter(*self.uav.get_position(), c='b', marker='.')
            self.ax.scatter(*self.source_pos, c="y", marker="*", s=300, label="Source")
            self.ax.scatter(*self.source_pos, c="r", marker="o")
        self.ax.set_xlim(0, self.size + 10)
        self.ax.set_ylim(0, self.size + 10)
        plt.title(name)
        self.ax.legend()

        if belief_visualizer is not None and belief is not None:
            belief_visualizer.draw(self.ax, belief)

        plt.pause(pause_interval)

        # plt.clf()
        # c, marker, alpha = {1:('g','*',1),-1:('r','x',1),0:('g','+',alpha)}[state]
        # plt.scatter(self.uav_posi[0],self.uav_posi[1],c=c, marker=marker, label='UAV', s=300, alpha=alpha)
        # plt.scatter(self.source_pos[0],self.source_pos[1],c='y',marker='*', label='Source', s=300, alpha=0.8)
        # plt.title('UAV-RL')
        # plt.xlim(0,self.size + 10)
        # plt.ylim(0,self.size + 10)
        # plt.legend()
        # plt.show(block=False)
        # plt.pause(pause_interal) # Short pause to allow the plot to be displayed



    def time(self):
        return self.uav.time()


    def close(self):
        json.dump(self.readings, open("readings.json", "w"))
        if self.fig:
            plt.close(self.fig)
            self.fig = None

    # --- helpers ---
    def _get_random_position(self):
        return [np.random.randint(0, self.size), np.random.randint(0, self.size)]

    def set_obstacles(self, obstacles):
        """
        obstacles: list of (x, y, radius)
        """
        self.obstacles = obstacles

    def _collides(self, pos):
        for ox, oy, r in self.obstacles:
            if np.linalg.norm(np.array(pos) - np.array([ox, oy])) <= r:
                return True
        return False

    def _true_reading(self):
        """
        Physics-only radiation model (no noise)
        """
        si = self.source_intensity * 60
        a = self.sensor_area
        eff = 0.8
        td = 0.020

        x1, y1 = self.uav.get_position()
        x2, y2 = self.source_pos
        z = 10  # altitude
        # ds2 = max(distance_square(self.uav.get_position(), self.source_pos), 1e-6)

        ds2 = max((x1 - x2) ** 2 + (y1 - y2) ** 2 + z ** 2, 1e-6)
        # print('si',si/10000, 'a',a, 'eff',eff, 'td',td, 'ds2',ds2)
        # photos/min * m*m * sec
        l = (si * a * eff * td) / ds2
        reading = 1 - np.exp(-l)
        if self.readings:
            self.readings['true_readings'].append(reading)
        return reading
