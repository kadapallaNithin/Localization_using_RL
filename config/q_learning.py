from config.base import get_config

EXPERIMENT_CONFIG = get_config()
EXPERIMENT_CONFIG.update({
    "agent": {
        "type": "Q-learning",
        "params": {
            "eps_min":2e-3,
            # "tau":4e-7, # calculated below
            "load_if_exists": True,
        },
    },
})
EXPERIMENT_CONFIG["experiment"]["episodes"] = 15_000_000
EXPERIMENT_CONFIG["experiment"]["print_interval"] = 1000
# EXPERIMENT_CONFIG["experiment"]["render"] = True
EXPERIMENT_CONFIG["state"]["type"] = 'signal_action_quadrant'
EXPERIMENT_CONFIG["reward"] = {
    "type": "sparse_find",
    "params": {
        "found_reading": 0.95,
        "found_reward": 0.0,
        "delay_penalty": 0.0,
        # "delay_penalty": 0.2,
    },
}
EXPERIMENT_CONFIG["agent"]["params"]["tau"] = 6/EXPERIMENT_CONFIG["experiment"]["episodes"]
# EXPERIMENT_CONFIG["agent"]["params"]["tau"] = 1 #6/EXPERIMENT_CONFIG["experiment"]["episodes"]


