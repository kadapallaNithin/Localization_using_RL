from config.base import get_config
import numpy as np
EXPERIMENT_CONFIG = get_config()
EXPERIMENT_CONFIG.update({
    "agent": {
        "type": "DQN",
        "params": {
            "gamma": 0.99,
            "lr": 1e-3,
            # "lr": 3e-4,
            "batch_size": 64,
            "buffer_capacity": 50_000,
            "min_replay_size": 10_000,
            "update_every": 4,
            "target_update_freq": 100,
            "eps_start": 1.0,
            "eps_min": 0.002,
            "tau": 5e-06,
            "load_if_exists": True,
            # "path": "model.pth",
        },
    },
})
# EXPERIMENT_CONFIG["experiment"]["episodes"] = 1_0
EXPERIMENT_CONFIG["experiment"]["name"] = "dqn"
EXPERIMENT_CONFIG["environment"]["max_turn_rate"] = np.pi/3
# EXPERIMENT_CONFIG["experiment"]["print_interval"] = 100
# EXPERIMENT_CONFIG["agent"]["params"]["tau"] = 4e-2

# EXPERIMENT_CONFIG["state"]["type"] = "signal_delta"
# EXPERIMENT_CONFIG["reward"]["type"] = "improvement"
max_steps = 200
EVAL = False
if EVAL:
    max_steps = 200
    episodes = 50
    EXPERIMENT_CONFIG["agent"]["params"]["eps_min"] = 0.0
    EXPERIMENT_CONFIG["agent"]["params"]["tau"] = 1
    EXPERIMENT_CONFIG["experiment"]["render"] = True
    EXPERIMENT_CONFIG["environment"]["initial_positions_file"] = "initial_positions.json"
else:
    episodes = 100_000
    EXPERIMENT_CONFIG["agent"]["params"]["tau"] = 0.003 #5/episodes
    EXPERIMENT_CONFIG["experiment"]["parallel_envs"] = 8
EXPERIMENT_CONFIG["experiment"]["episodes"] = episodes
EXPERIMENT_CONFIG["experiment"]["max_steps"] = max_steps
EXPERIMENT_CONFIG["experiment"]["print_interval"] = 1000 if episodes > 1000 else 100
# EXPERIMENT_CONFIG["state"]["type"] = 'signal_action_quadrant'
EXPERIMENT_CONFIG["state"]["type"] = 'signal_history_quadrant_heading'
EXPERIMENT_CONFIG["state"]["history_length"] = 5
# EXPERIMENT_CONFIG["reward"] = {
#     "type": "sparse_find",
#     "params": {
#         "found_reading": 0.95,
#         "found_reward": 0.0,
#         "delay_penalty": 0.0,
#         # "delay_penalty": 0.2,
#     },
# }

