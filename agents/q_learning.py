from .base import QAgent
import pickle
import numpy as np
import os
# import json
# import time

# DEBUG_LOG_PATH = "/home/nithin/code/RL/seminar/Stage_2/lab/src_finder/.cursor/debug.log"

model_cache = {}

class QLearningAgent(QAgent):
    def param_names(self):
        return ['action_dim', 'gamma', 'alpha', 'bins']

    def __init__(self,
                 action_dim,
                 gamma=0.99,
                 alpha=0.8,
                 bins=100,
                 load_if_exists=False,
                 include_quadrant=True,
                 eps_min=2e-3,
                 eps_start=1,
                 tau=4e-7,
                 path="qlearning.pkl"
        ):
        self.episode = 0
        self.eps_min = eps_min
        self.epsilon = eps_start
        self.tau = tau
        self.q = {}
        # # For debugging terminal vs time-limit handling; set from `experiment.py`.
        # self._last_truncated = False
        # self._debug_logged_episode = None
        # self._debug_logged_first_update = False
        self.action_dim = action_dim
        self.gamma = gamma
        self.alpha = alpha
        self.bins = bins  # discretization level
        self.include_quadrant = include_quadrant

        if not path.endswith('.pkl'):
            path += '.pkl'
        self.path = path
        if load_if_exists:
            self.load(load_if_exists=load_if_exists)

    def _discretize(self, state):
        if self.include_quadrant:
            mt0, mt1, ap, w = state
        else:
            mt0, mt1, ap = state
        mt0_d = int(mt0 * self.bins)
        mt1_d = int(mt1 * self.bins)
        if self.include_quadrant:
            return (mt0_d, mt1_d, int(ap), int(w))
        return (mt0_d, mt1_d, int(ap))

    def act(self, state):
        s = self._discretize(state)
        if np.random.rand() < self.epsilon or s not in self.q:
            return np.random.randint(self.action_dim)
        return max(self.q[s], key=self.q[s].get)

    def update(self, state, action, reward, next_state, done):
        s, sp = self._discretize(state), self._discretize(next_state)
        if s not in self.q:
            self.q[s] = {}
        if action not in self.q[s]:
            self.q[s][action] = 0.0

        max_next = max(self.q.get(sp, {}).values(), default=0.0)
        target = reward + (0 if done else self.gamma * max_next)
        self.q[s][action] += self.alpha * (target - self.q[s][action])
        # q_before = float(self.q[s][action])
        # td_error = float(target - q_before)
        # q_after = q_before + self.alpha * td_error

        # # #region debug q-learning TD components (first/terminal update)
        # terminal_should_log = self.episode == 0 and (bool(done) or bool(self._last_truncated))
        # first_update_should_log = self.episode == 0 and not self._debug_logged_first_update

        # if self._debug_logged_episode != self.episode:
        #     self._debug_logged_episode = self.episode
        #     self._debug_logged_first_update = False

        # if terminal_should_log or first_update_should_log:
        #     payload = {
        #         "runId": "debug-pre-fix",
        #         "hypothesisId": "H1" if terminal_should_log else "H3",
        #         "location": "agents/q_learning.py:update",
        #         "message": "q-learning update details",
        #         "data": {
        #             "episode": self.episode,
        #             "state_disc": s,
        #             "next_state_disc": sp,
        #             "state_raw": [float(x) for x in state],
        #             "next_state_raw": [float(x) for x in next_state],
        #             "action": int(action),
        #             "reward": float(reward),
        #             "done": bool(done),
        #             "last_truncated": bool(self._last_truncated),
        #             "epsilon": float(getattr(self, "epsilon", 0.0)),
        #             "alpha": float(self.alpha),
        #             "gamma": float(self.gamma),
        #             "max_next": float(max_next),
        #             "target": float(target),
        #             "q_before": float(q_before),
        #             "td_error": float(td_error),
        #             "q_after": float(q_after),
        #             "q_table_size": int(len(self.q)),
        #         },
        #         "timestamp": int(time.time() * 1000),
        #     }
        #     try:
        #         with open(DEBUG_LOG_PATH, "a") as f:
        #             f.write(json.dumps(payload) + "\n")
        #     except Exception:
        #         pass

        # self.q[s][action] = q_after

        # if first_update_should_log:
        #     self._debug_logged_first_update = True
        # # #endregion

    def save(self):
        with open(self.path, "wb") as f:
            pickle.dump(self.q, f)

    def load(self, load_if_exists=False):
        if load_if_exists:
            if not os.path.exists(self.path):
                return
        if self.path in model_cache:
            print("Model loaded from cache.")
            self.q = model_cache[self.path]
        else:
            print(f"Loading file {self.path}")
            with open(self.path, "rb") as f:
                self.q = pickle.load(f)
            model_cache[self.path] = self.q
