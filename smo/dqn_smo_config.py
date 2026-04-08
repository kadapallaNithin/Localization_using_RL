from copy import deepcopy

from config.dqn import EXPERIMENT_CONFIG as DQN_CONFIG


# Start from the existing DQN configuration
EXPERIMENT_CONFIG = deepcopy(DQN_CONFIG)

# Use the sliding–mode observer on top of the noisy radiation sensor
EXPERIMENT_CONFIG["environment"]["observer"] = {
    "type": "smo",
    "params": {
        "alpha": 0.3,
        "k": 0.05,
        "dt": 1.0,
    },
}

# Give this experiment a distinct name
base_name = EXPERIMENT_CONFIG["experiment"].get("name", "DQN")
EXPERIMENT_CONFIG["experiment"]["name"] = f"{base_name}_smo"

record_readings = True  # Set to True to record readings in the environment
if record_readings:
    EXPERIMENT_CONFIG["environment"]["sensor"]["record_readings"] = True
    EXPERIMENT_CONFIG["experiment"]["name"] += "_record_readings"
    # run 2 episodes to generate readings
    EXPERIMENT_CONFIG["experiment"]["episodes"] = 1

print(EXPERIMENT_CONFIG["experiment"]["name"])