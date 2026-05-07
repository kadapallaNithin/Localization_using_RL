from copy import deepcopy

from config.base import get_config

# Switch to select student agent type: "q_learning" or "dqn"
# STUDENT_TYPE = "q_learning"  # Change to "dqn" to train DQN with Grad teacher
STUDENT_TYPE = "dqn"
EXPERIMENT_CONFIG = get_config()

# Determine student config based on switch
if STUDENT_TYPE == "dqn":
    from config.dqn import EXPERIMENT_CONFIG as DQN_CONFIG
    # student_cfg = {
    #     "type": "DQN",
    #     "params": {
    #         "gamma": 0.99,
    #         "lr": 1e-3,
    #         "batch_size": 64,
    #         "buffer_capacity": 50_000,
    #         "min_replay_size": 10_000,
    #         "update_every": 4,
    #         "target_update_freq": 100,
    #         "eps_start": 1.0,
    #         "eps_min": 0.002,
    #         "tau": 5e-06,
    #         "load_if_exists": True,
    #     },
    # }
    student_cfg = deepcopy(DQN_CONFIG)
    episodes = 1000_000
    print_interval = 1000
    state_type = "signal_history_quadrant"
    history_length = 5
else:  # q_learning
    from config.q_learning import EXPERIMENT_CONFIG as QLEARN_CONFIG
    # student_cfg = QLEARN_CONFIG
    student_cfg = {
        "type": "Q-learning",
        "params": {},
    }
    episodes = 15_000_000
    print_interval = 1000
    state_type = None  # Will use default from base
    history_length = 3

EXPERIMENT_CONFIG.update({
    "agent": {
        "type": "Supervised",
        "params": {
            # Teacher provides actions; student is updated from resulting transitions.
            "teacher_cfg": {
                "type": "Grad",
                "params": {},
            },
            # Student to learn from teacher-generated trajectories.
            "student_cfg": student_cfg,
        },
    },
})
EXPERIMENT_CONFIG["experiment"].update({
    "name": f"supervised_{STUDENT_TYPE}_{state_type}_h{history_length}",
    "episodes": episodes,
    "print_interval": print_interval,
    # "render": True,
})

# print(EXPERIMENT_CONFIG, 'supervised config')
# Update student params if q_learning
if STUDENT_TYPE == "q_learning":
    EXPERIMENT_CONFIG["agent"]["params"]["student_cfg"]["params"].update(
        QLEARN_CONFIG["agent"]["params"]
    )

# Set state type and history length
if state_type:
    EXPERIMENT_CONFIG["state"]["type"] = state_type
    EXPERIMENT_CONFIG["state"]["history_length"] = history_length

# Update epsilon schedule for student params if q_learning
if STUDENT_TYPE == "q_learning" and "tau" in EXPERIMENT_CONFIG["agent"]["params"]["student_cfg"]["params"]:
    EXPERIMENT_CONFIG["agent"]["params"]["student_cfg"]["params"]["tau"] = 6 / EXPERIMENT_CONFIG["experiment"]["episodes"]

# Debug
# EXPERIMENT_CONFIG["experiment"]["episodes"] = 3
# EXPERIMENT_CONFIG["experiment"]["render"] = True
