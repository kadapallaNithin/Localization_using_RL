# from agents.belief_mpc_pf import BeliefMPCPF
from .q_learning import QLearningAgent
from .simple_agents import GradientBasedAgent, ManualAgent, UniformSearchAgent
from .supervised import SupervisedAgent
# from agents.belief_space_mpc import BeliefSpaceMPCAgent

def build_agent(cfg, state_dim, action_dim, env_size, state_builder_type="signal_action_quadrant"):
    t = cfg["type"]
    p = cfg.get("params", {})

    # print(p)
    if t == "DQN":
        # Lazy import so `torch` isn't required unless this agent is used.
        from .dqn import DQNAgent, DQNWrapperAgent
        return DQNAgent(
            state_dim=state_dim,
            action_dim=action_dim,
            **p,
        )
    elif t == "DQNR":
        # Lazy import so `torch` isn't required unless this agent is used.
        from .dqn import DQNAgent, DQNWrapperAgent
        return DQNWrapperAgent(
            state_dim=state_dim,
            action_dim=action_dim,
            **p,
        )

    if t == "Q-learning":
        return QLearningAgent(
            action_dim=action_dim,
            **p,
        )

    if t == "Grad":
        return GradientBasedAgent(action_dim, state_builder_type=state_builder_type, **p)
    elif t == "Supervised":
        # `experiment.py` injects `params.path` for where to save/load the model.
        # For supervised, we propagate it to the *student* (the thing that actually learns).
        import copy

        # Check teacher_cfg and student_cfg presence
        if "teacher_cfg" not in p or not p["teacher_cfg"]:
            print("[SupervisedAgent Setup] WARNING: 'teacher_cfg' not configured in params!")
        if "student_cfg" not in p or not p["student_cfg"]:
            print("[SupervisedAgent Setup] WARNING: 'student_cfg' not configured in params!")

        teacher_cfg = p.get("teacher_cfg", {})
        student_cfg = p.get("student_cfg", {})

        if "path" in p:
            student_cfg = copy.deepcopy(student_cfg)
            student_cfg.setdefault("params", {})
            student_cfg["params"]["path"] = p["path"]

        return SupervisedAgent(
            state_dim,
            action_dim,
            env_size,
            teacher_cfg,
            student_cfg,
            state_builder_type=state_builder_type,
        )

    print(cfg.keys())
    if t == "Uniform":
        return UniformSearchAgent(area_size=env_size, **p)

    elif t == "ppo_lstm":
        # Lazy import so `torch` isn't required unless this agent is actually used.
        from agents.ppo_lstm import PPO_LSTM
        return PPO_LSTM(
            state_dim=state_dim,
            action_dim=action_dim,
            **cfg.get("params", {})
        )
    elif t == "BELIEF_MPC":
        return BeliefSpaceMPCAgent(
            state_dim,
            action_dim,
            env_size,
            horizon=cfg.get("horizon", 1),
        )

    elif t == "Manual":
         return ManualAgent()

    if cfg["type"] == "BELIEF_MPC_PF":
            return BeliefMPCPF(
                state_dim,
                action_dim,
                env_size,
                **p,
            )
    raise ValueError(f"Unknown agent type: {t}")
