from .base import Agent

# Supervised learning agent that uses a teacher-student framework. The teacher is used to generate actions for the student to learn from,
# while the student is the one that gets updated and saved/loaded.

class SupervisedAgent(Agent):
    def __init__(self, state_dim, action_dim, env_size, teacher_cfg, student_cfg, state_builder_type="signal_action_quadrant"):
        super().__init__()
        from .factory import build_agent

        # `teacher_cfg`/`student_cfg` may be either:
        # - an "agent config" as expected by `agents.factory.build_agent` (has `type`)
        # - a full experiment config (has top-level `agent`)
        # - empty `{}` (treated as "use student_cfg" for the teacher)
        teacher_cfg = self._normalize_agent_cfg(teacher_cfg, fallback_cfg=student_cfg)
        student_cfg = self._normalize_agent_cfg(student_cfg)

        self.teacher = build_agent(
            teacher_cfg,
            state_dim=state_dim,
            action_dim=action_dim,
            env_size=env_size,
            state_builder_type=state_builder_type,
        )
        self.student = build_agent(
            student_cfg,
            state_dim=state_dim,
            action_dim=action_dim,
            env_size=env_size,
            state_builder_type=state_builder_type,
        )

    @staticmethod
    def _normalize_agent_cfg(cfg, fallback_cfg=None):
        if cfg is None or cfg == {}:
            if fallback_cfg is None or fallback_cfg == {}:
                raise ValueError("Teacher/student agent cfg is empty; cannot build agent.")
            cfg = fallback_cfg

        # If this looks like an experiment config, unwrap the nested `agent` section.
        if isinstance(cfg, dict) and "agent" in cfg and "type" not in cfg:
            cfg = cfg["agent"]

        if not isinstance(cfg, dict) or "type" not in cfg:
            raise KeyError("Agent cfg must be a dict with a `type` field.")

        # Ensure `params` exists (factory uses `cfg.get('params', {})` but be safe).
        cfg = dict(cfg)
        cfg.setdefault("params", {})
        return cfg

    def act(self, state):
        # Teacher generates the action; student learns from the resulting transition.
        # print('teacher act')
        return self.teacher.act(state)

    def update(self, *args):
        return self.student.update(*args)

    def set_uav_position(self, pos):
        # Some wrapped agents (e.g., belief-MPC) depend on the current UAV position.
        if hasattr(self.teacher, "set_uav_position"):
            self.teacher.set_uav_position(pos)
        if hasattr(self.student, "set_uav_position"):
            self.student.set_uav_position(pos)

    def end_episode(self):
        # If the student has an epsilon schedule (e.g., Q-learning/DQN), advance it.
        if hasattr(self.student, "end_episode"):
            return self.student.end_episode()

    def save(self):
        # Delegate to student agent's save method
        return self.student.save()

    def load(self, load_if_exists=False):
        # Delegate to student agent's load method
        return self.student.load(load_if_exists=load_if_exists)

    def get_belief(self):
        # Prefer student belief if available; otherwise fall back to teacher.
        if hasattr(self.student, "get_belief"):
            return self.student.get_belief()
        if hasattr(self.teacher, "get_belief"):
            return self.teacher.get_belief()
        return None  # Not all agents implement beliefs; return None if neither does.
        # raise AttributeError("Neither student nor teacher exposes `get_belief()`.")
