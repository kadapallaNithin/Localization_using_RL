from abc import ABC, abstractmethod

class RewardFunction(ABC):
    @abstractmethod
    def compute(self, mt0, mt1, step, max_steps):
        pass

class SparseFindReward(RewardFunction):
    def __init__(self, found_reading=0.95, found_reward=100.0, delay_penalty=0.2):
        self.found_reading = found_reading
        self.found_reward = found_reward
        self.delay_penalty = delay_penalty

    def compute(self, mt0, mt1, step, max_steps):
        if mt1 > self.found_reading:
            return self.found_reward, True
        return mt1 - 1.0 - self.delay_penalty, False

class ImprovementReward(RewardFunction):
    def compute(self, mt0, mt1, step, max_steps):
        reward = (mt1 - mt0) * 10.0
        done = mt1 > 0.95
        return reward, done

def build_reward(cfg):
    t = cfg["type"]
    p = cfg.get("params", {})

    if t == "improvement":
        return ImprovementReward()

    if t == "sparse_find":
        return SparseFindReward(**p)

    raise ValueError(f"Unknown reward type: {t}")
