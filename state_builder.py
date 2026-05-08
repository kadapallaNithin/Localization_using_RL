from abc import ABC, abstractmethod
import numpy as np

action_space_size = 40  # 8 directions * 5 speeds (including no_op)
no_op = 40  # Define no_op action

class StateBuilder(ABC):
    @abstractmethod
    def observation_space(self, history_length=None):
        pass

    @abstractmethod
    def build(self, mt0, mt1, action, env, signal_history=None, action_history=None):
        """
        Build state from measurements and action.

        Args:
            mt0: First signal measurement (float)
            mt1: Second signal measurement (float)
            action: Action taken (int)
            env: Environment instance
            signal_history: Full deque of signal measurements (for new builders)
            action_history: Full deque of actions (for new builders)
        """
        pass

    # @abstractmethod
    # def last_action(self):
    #     """
    #     Optional method to expose the last action taken, if relevant for state construction.
    #     """
    #     return None

class SignalState(StateBuilder):
    def observation_space(self, history_length=None):
        low = np.array([0.0, 0.0], dtype=np.float32)
        high = np.array([1.0, 1.0], dtype=np.float32)
        return low, high

    def build(self, mt0, mt1, action, env, signal_history=None, action_history=None):
        return np.array([mt0, mt1], dtype=np.float32)

class SignalActionState(StateBuilder):
    def observation_space(self, history_length=None):
        low = np.array([0.0, 0.0, 0], dtype=np.float32)
        high = np.array([1.0, 1.0, action_space_size], dtype=np.float32)
        return low, high

    def build(self, mt0, mt1, action, env, signal_history=None, action_history=None):
        return np.array([mt0, mt1, action], dtype=np.float32)

class SignalActionQuadrantState(StateBuilder):
    def observation_space(self, history_length=None):
        low = np.array([0.0, 0.0, 0, 0], dtype=np.float32)
        high = np.array([1.0, 1.0, action_space_size, 3], dtype=np.float32)
        return low, high

    def build(self, mt0, mt1, action, env, signal_history=None, action_history=None):
        w = env.uav.get_quadrant(env.size)
        return np.array([mt0, mt1, action, w], dtype=np.float32)

class SignalDeltaActionQuadrantState(StateBuilder):
    def observation_space(self, history_length=None):
        low = np.array([0.0, -1.0, 0, 0], dtype=np.float32)
        high = np.array([1.0, 1.0, action_space_size, 3], dtype=np.float32)
        return low, high

    def build(self, mt0, mt1, action, env, signal_history=None, action_history=None):
        w = env.uav.get_quadrant(env.size)
        return np.array([mt0, mt1 - mt0, action, w], dtype=np.float32)

class SignalDeltaState(StateBuilder):
    def observation_space(self, history_length=None):
        low = np.array([0.0, -1.0], dtype=np.float32)
        high = np.array([1.0, 1.0], dtype=np.float32)
        return low, high

    def build(self, mt0, mt1, action, env, signal_history=None, action_history=None):
        return np.array([mt1, mt1 - mt0], dtype=np.float32)

class SignalHistoryActionState(StateBuilder):
    """
    State representation with k-step history.
    Format: [mt0, action0, gradient1, action1, gradient2, action2, ..., gradient_k]
    where gradient_i = signal_i - signal_(i-1)

    This builder uses the full signal_history and action_history from the environment.
    normalize: if True, actions are scaled to [0,1] (action/40) and bounds reflect that.
    """
    def __init__(self, normalize=False):
        self.normalize = normalize

    def observation_space(self, history_length=3):
        # State: [signal_0, action_0, gradient_1, action_1, ..., gradient_k]
        # Size: 1 + 1 + (history_length - 1) * 2 + 1 = history_length * 2 + 1
        state_size = history_length * 2 + 1
        low = np.full(state_size, -np.inf, dtype=np.float32)
        low[::2] = 0.0  # signals/gradients ≥ 0 (gradients can be negative, so -inf is fine there)
        if self.normalize:
            low[1::2] = -1.0  # no-op encoded as -1.0; movement actions in [0, 1]
        else:
            low[1::2] = 0     # raw action range starts at 0

        high = np.full(state_size, 1.0, dtype=np.float32)
        if not self.normalize:
            high[1::2] = 40  # raw action range [0, 40]

        return low, high

    def build(self, mt0, mt1, action, env, signal_history=None, action_history=None):
        """
        Build state from full signal and action histories.
        signal_history: deque of signal measurements [mt0, mt1, ..., mt_k]
        action_history: deque of actions [action0, action1, ..., action_(k-1)]
        """
        if signal_history is None or action_history is None:
            # Fallback if histories aren't provided
            return np.array([mt0, mt1, action], dtype=np.float32)

        state = []
        signal_list = list(signal_history)
        action_list = list(action_history)
        # print(f'signal_history: {[round(float(x), 2) for x in signal_list]}, action_history: {[int(x) for x in action_list]}')

        # Add first signal
        state.append(float(signal_list[0]))

        # Interleave actions and gradients
        for i in range(len(action_list)):
            a = float(action_list[i])
            if self.normalize:
                # no-op (40) → -1.0 (distinct sentinel); movement (0-39) → [0, 1]
                state.append(-1.0 if a >= 40.0 else a / 39.0)
            else:
                state.append(a)
            if i + 1 < len(signal_list):
                gradient = float(signal_list[i + 1] - signal_list[i])
                state.append(gradient)

        # # Pad if necessary (in case we're at the beginning of episode)
        max_state_size = env.history_length * 2 + 1
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
        # print with fixed width on terminal for signals and int for actions
        # print(f'Built state: {[round(float(x), 2) if i%2==0 else int(x) for i, x in enumerate(state)]}')
        return np.array(state[:max_state_size], dtype=np.float32)

class SignalHistoryQuadrantState(SignalHistoryActionState):
    def observation_space(self, history_length=3):
        # State: [signal_0, action_0, gradient_1, action_1, ..., gradient_k, quadrant]
        # Size: history_length * 2 + 1 + 1 = history_length * 2 + 2
        state_size = history_length * 2 + 2
        low = np.full(state_size, -np.inf, dtype=np.float32)
        low[::2] = 0.0  # Signals and gradients have min 0 (or -1 for gradients)
        low[1::2] = 0    # Actions have min 0
        low[-1] = 0      # Quadrant has min 0

        high = np.full(state_size, 1.0, dtype=np.float32)
        if not self.normalize:
            high[1::2] = 40  # raw action range [0, 40]
            high[-1] = 3     # raw quadrant range [0, 3]

        return low, high

    def build(self, mt0, mt1, action, env, signal_history=None, action_history=None):
        base_state = super().build(mt0, mt1, action, env, signal_history, action_history)
        w = env.uav.get_quadrant(env.size)
        quad_scale = 3.0 if self.normalize else 1.0
        state = np.append(base_state, w / quad_scale)
        # print(len(state))
        return state


def build_state(cfg):
    t = cfg["type"]

    if t == "signal_only":
        return SignalState()

    if t == "signal_delta":
        return SignalDeltaState()

    if t == "signal_action_quadrant":
        return SignalActionQuadrantState()

    elif t == "signal_delta_action_quadrant":
        return SignalDeltaActionQuadrantState()

    elif t == "signal_history_action":
        history_length = cfg.get("history_length", 3)
        normalize = cfg.get("normalize", False)
        state_builder = SignalHistoryActionState(normalize=normalize)
        state_builder.history_length = history_length
        return state_builder

    elif t == "signal_history_quadrant":
        history_length = cfg.get("history_length", 3)
        normalize = cfg.get("normalize", False)
        state_builder = SignalHistoryQuadrantState(normalize=normalize)
        state_builder.history_length = history_length
        return state_builder

    raise ValueError(f"Unknown state type: {t}")


def extract_state_components(state, state_builder_type, normalize=False):
    """
    Extract prev_signal, curr_signal, and last_action from state.

    Handles different state builder types:
    - signal_only: [signal_0, signal_1] -> prev=signal_0, curr=signal_1, last_action=0
    - signal_delta: [signal_1, delta] -> prev=signal_1-delta, curr=signal_1, last_action=0
    - signal_action_quadrant: [signal_0, signal_1, action, quadrant]      -> prev=signal_0, curr=signal_1, last_action=action
    - signal_delta_action_quadrant: [signal_0, delta, action, quadrant]
      -> prev=signal_0, curr=signal_0+delta, last_action=action
    - signal_history_action: [signal_0, action_0[/40], gradient_1, action_1[/40], ..., gradient_k]
      -> prev=signal_0, curr=signal_0+sum(gradients), last_action=last_action_in_history

    normalize: must match the flag used when the state was built (signal_history_* only).
               When True, stored actions are action/40 and are denormalized here.

    Args:
        state: numpy array representing the state
        state_builder_type: string type of state builder
        normalize: whether actions were normalized during build (default False)

    Returns:
        tuple: (prev_signal, curr_signal, last_action)
    """
    if state_builder_type == "signal_only":
        prev_signal = float(state[0])
        curr_signal = float(state[1])
        last_action = 0

    elif state_builder_type == "signal_delta":
        curr_signal = float(state[0])
        delta = float(state[1])
        prev_signal = curr_signal - delta
        last_action = 0

    elif state_builder_type == "signal_action_quadrant":
        prev_signal = float(state[0])
        curr_signal = float(state[1])
        last_action = int(state[2])

    elif state_builder_type == "signal_delta_action_quadrant":
        prev_signal = float(state[0])
        delta = float(state[1])
        curr_signal = prev_signal + delta
        last_action = int(state[2])

    elif state_builder_type == "signal_history_action" or state_builder_type == "signal_history_quadrant":
        # Format: [signal_0, action_0[encoded], gradient_1, action_1[encoded], ..., gradient_k]
        # With normalize=True: no-op→-1.0, movement (0-39)→val/39; denorm inverts that.
        def _denorm(val):
            if not normalize:
                return int(val)
            return 40 if val < 0 else int(round(val * 39.0))

        curr_signal = float(state[0])
        last_action = _denorm(float(state[1]))
        # Add all gradients to get current signal
        # print(f'Components : {[round(float(x), 2) if i%2==0 else int(x) for i, x in enumerate(state)]}')
        for i in range(2, len(state), 2):
            if i < len(state):
                curr_signal += float(state[i])
            else:
                print(f'Warning: Expected gradient at index {i} but state length is {len(state)}')
        # print('last action before quadrant check:', last_action)
        if state_builder_type == "signal_history_action":
            prev_signal = curr_signal - float(state[-1])  # Last gradient in history
            last_action = _denorm(float(state[-2]))        # Last action in history
        else:
            prev_signal = curr_signal - float(state[-2])  # Last gradient before quadrant
            last_action = _denorm(float(state[-3]))        # Last action before quadrant
        # print(round(prev_signal, 4), round(curr_signal, 4), last_action)
    else:
        # Fallback: assume [prev_signal, curr_signal, action, ...]
        prev_signal = float(state[0]) if len(state) > 0 else 0.0
        curr_signal = float(state[1]) if len(state) > 1 else 0.0
        last_action = int(state[2]) if len(state) > 2 else 0

    return prev_signal, curr_signal, last_action
