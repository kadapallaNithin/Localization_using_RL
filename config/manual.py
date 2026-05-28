EXPERIMENT_CONFIG = {
    "experiment": {
        "name": "manual",
        "seed": 42,
        "episodes": 10,
        "max_steps": 500,
        "render": True,
        "pause_interval": 0.05,
        "save_dir": "results/",
        "print_interval": 1,
        "parallel_envs": 1,
        "training": False,
    },

    "environment": {
        "size": 200,
        "source_intensity": 2.0e7,
        "physics": {
            "altitude": 10.0,
            "efficiency": 0.8,
            "dwell_time": 0.020,
        },
        "sensor": {
            "type": "radiation",
            "noise_std": 0.02,
        },
        "initial_positions_file": None,
    },

    "state": {
        "type": "signal_action_quadrant",
        "history_length": 3,
    },

    "reward": {
        "type": "sparse_find",
        "params": {
            "found_reading": 0.95,
            "found_reward": 100.0,
            "delay_penalty": 0.0,
        },
    },

    "agent": {
        "type": "Manual",
        "params": {},
    },
}
