from abc import ABC, abstractmethod
import json
import numpy as np

class Agent(ABC):
    @abstractmethod
    def act(self, state):
        pass

    @abstractmethod
    def update(self, *args):
        pass

    @abstractmethod
    def load(self, path=None, **kwargs):
        pass

    @abstractmethod
    def save(self, path=None, **kwargs):
        pass

    def reset(self):
        pass

    def param_names(self):
        return []

    def params(self):
        params = {'name':self.__class__.__name__}
        for name in self.param_names():
            params[name] = getattr(self, name)
        return params

    def json_params(self):
        return json.dumps(self.params())

    def print_intvl(self, *args, **kwargs):
        pass

    # def maybe_learn(self, *args, **kwargs):
    #     pass

class QAgent(Agent):
    def end_episode(self):
        self.epsilon = self.eps_min + (1 - self.eps_min) * np.exp(-self.tau * self.episode)
        self.episode += 1

    def print_intvl(self, *args, **kwargs):
        print(
            # f"Ep {self.episode:5d} | "
            f"Eps {self.epsilon:1.5f}",
            *args, **kwargs)
