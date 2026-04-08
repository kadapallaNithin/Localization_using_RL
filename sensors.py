import numpy as np

class RadiationSensor:
    def __init__(self, noise_std=None):
        self.noise_std = noise_std

    def read(self, true_value):
        if self.noise_std is not None:
            true_value += np.random.normal(0, self.noise_std)
        return np.clip(true_value, 0.0, 1.0)
