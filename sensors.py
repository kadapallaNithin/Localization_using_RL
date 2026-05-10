import numpy as np


class RadiationSensor:
    """
    Radiation sensor with configurable noise models.

    Noise models
    ------------
    "none"           : no noise (clean physics signal)
    "gaussian"       : additive N(0, std) — simple baseline, std fixed regardless of signal level
    "poisson"        : shot noise — physically correct for photon counting; inverts p=1-exp(-λ)
                       to get expected count λ, samples Poisson(λ), maps back; noise ∝ sqrt(signal)
    "multiplicative" : log-normal / proportional noise — noise scales linearly with signal level;
                       models RF shadowing or calibration uncertainty
    "composite"      : Poisson shot noise + additive Gaussian — most realistic; models both
                       counting statistics and electronics noise floor

    Config example (in sensor dict)
    --------------------------------
        "noise": {
            "model": "composite",
            "gaussian_std": 0.005,       # electronics noise floor (composite/gaussian)
            "multiplicative_std": 0.05,  # fractional gain uncertainty (multiplicative/composite)
        }

    Backward-compatible shorthand: if "noise_std" key is present (old config), it maps to
    the "gaussian" model with that std.
    """

    # Minimum expected count to avoid log(0) issues when inverting the Poisson transform
    _MIN_LAMBDA = 1e-6

    def __init__(self, model="none", gaussian_std=0.0, multiplicative_std=0.0):
        self.model = model
        self.gaussian_std = gaussian_std
        self.multiplicative_std = multiplicative_std

    def read(self, true_value):
        p = float(true_value)
        p = self._apply_noise(p)
        return float(np.clip(p, 0.0, 1.0))

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _apply_noise(self, p):
        if self.model == "none":
            return p

        if self.model == "gaussian":
            return p + np.random.normal(0.0, self.gaussian_std)

        if self.model == "poisson":
            return self._poisson_noise(p)

        if self.model == "multiplicative":
            # Multiplicative Gaussian: p * (1 + N(0, σ))
            # Equivalent to log-normal with small σ; mean-preserving at first order
            return p * (1.0 + np.random.normal(0.0, self.multiplicative_std))

        if self.model == "composite":
            # Step 1: Poisson shot noise (counting statistics)
            p = self._poisson_noise(p)
            # Step 2: additive Gaussian (electronics noise floor)
            p += np.random.normal(0.0, self.gaussian_std)
            return p

        raise ValueError(f"Unknown noise model: {self.model!r}")

    def _poisson_noise(self, p):
        # Invert p = 1 - exp(-λ)  →  λ = -log(1 - p)
        # Clamp p away from 1.0 to keep λ finite
        p_clamped = min(p, 1.0 - 1e-9)
        lam = max(-np.log(1.0 - p_clamped), self._MIN_LAMBDA)
        # Sample actual count from Poisson distribution
        count = np.random.poisson(lam)
        # Map back: p_noisy = 1 - exp(-count)
        return 1.0 - np.exp(-float(count))

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    def __repr__(self):
        return (
            f"RadiationSensor(model={self.model!r}, "
            f"gaussian_std={self.gaussian_std}, "
            f"multiplicative_std={self.multiplicative_std})"
        )


def build_sensor(sensor_cfg: dict) -> RadiationSensor:
    """
    Build a RadiationSensor from a config dict.

    Supports two formats:

    Old format (backward-compatible):
        {"noise_std": 0.02}   →  gaussian model with std=0.02

    New format:
        {
            "noise": {
                "model": "composite",          # required
                "gaussian_std": 0.005,         # optional, default 0.0
                "multiplicative_std": 0.05,    # optional, default 0.0
            }
        }
    """
    # Backward-compat: plain noise_std key → gaussian
    if "noise_std" in sensor_cfg and "noise" not in sensor_cfg:
        std = sensor_cfg["noise_std"]
        if std is None:
            return RadiationSensor(model="none")
        return RadiationSensor(model="gaussian", gaussian_std=float(std))

    noise_cfg = sensor_cfg.get("noise", {})
    if not noise_cfg:
        return RadiationSensor(model="none")

    model = noise_cfg.get("model", "none")
    gaussian_std = float(noise_cfg.get("gaussian_std", 0.0))
    multiplicative_std = float(noise_cfg.get("multiplicative_std", 0.0))
    return RadiationSensor(
        model=model,
        gaussian_std=gaussian_std,
        multiplicative_std=multiplicative_std,
    )
