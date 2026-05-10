def get_config():
    return {
        "experiment": {
            # "name": "dqn_signal_delta_noise002",
            "seed": 42,
            "episodes": 200, #1000_000,
            "max_steps": 200,
            "render": False,
            "pause_interval":0.0001,
            "save_dir": "results/",
            "print_interval":100,
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
                # Noise model options:
                #   none         : no noise
                #   gaussian     : additive N(0, gaussian_std)   — simple baseline
                #   poisson      : shot noise (physically correct for photon counting)
                #   multiplicative: noise ∝ signal level (RF shadowing / gain uncertainty)
                #   composite    : Poisson + additive Gaussian (most realistic)
                "noise": {
                    "model": "composite",
                    "gaussian_std": 0.005,       # electronics noise floor
                    "multiplicative_std": 0.0,   # fractional gain uncertainty (0 = off)
                },
            },
        },

        "state": {
            "type": "signal_action_quadrant",
            "history_length": 3,
            # options for type:
            # "signal_only"
            # "signal_delta"
            # "signal_action_quadrant"
            # "signal_delta_action_quadrant"
            # "signal_history_action"
        },

        "reward": {
            "type": "sparse_find",
            # options:
            # "improvement"
            # "sparse_find"
            "params": {
                "found_reading": 0.95,
                "found_reward": 100.0,
                "delay_penalty": 0.0,
                # "delay_penalty": 0.2,
            },
        },

        "agent": {
            "type": "BELIEF_MPC",
            # options:
            # "DQN"
            # "Q-learning"
            # "Grad"
            # "Uniform"
            # "BELIEF_MPC"
            "params": {},
        },
    }

