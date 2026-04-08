import numpy as np


class SlidingModeObserver:
    """
    Simple sliding–mode–style observer for a scalar measurement y_k in [0, 1].

    It maintains an internal state x_hat_k that is updated using a combination
    of linear correction and a discontinuous (sign) term for robustness:

        x_hat_{k+1} = x_hat_k
                       + alpha * (y_k - x_hat_k) * dt
                       + k * sign(y_k - x_hat_k) * dt

    This acts as a robust low–pass filter on the noisy scalar measurement,
    which is suitable here because our environment state is built from the
    radiation sensor signal and its difference over time.
    """

    def __init__(self, alpha: float = 0.5, k: float = 0.05, dt: float = 1.0, x0: float = 0.0):
        self.alpha = float(alpha)
        self.k = float(k)
        self.dt = float(dt)
        self.x_hat = float(x0)

    def reset(self, x0: float | None = None) -> None:
        if x0 is None:
            x0 = 0.0
        self.x_hat = float(x0)

    def update(self, y: float) -> float:
        """Update observer state with new noisy measurement y and return estimate."""
        y = float(y)
        e = y - self.x_hat

        # Sliding-mode style correction: linear term + sign term
        self.x_hat += (self.alpha * e + self.k * np.sign(e)) * self.dt
        # Clamp to valid sensor range
        self.x_hat = float(np.clip(self.x_hat, 0.0, 1.0))
        return self.x_hat


def build_observer(cfg):
    """
    Factory to build an observer from config.

    Expected shape:
        cfg = {
            "type": "smo",
            "params": {
                "alpha": ...,
                "k": ...,
                "dt": ...,
            },
        }
    """
    if cfg is None:
        return None

    t = cfg.get("type")
    params = cfg.get("params", {})

    if t is None:
        return None

    if t.lower() == "smo":
        return SlidingModeObserver(**params)

    raise ValueError(f"Unknown observer type: {t}")

