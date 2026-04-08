
from rewards import build_reward
from sensors import RadiationSensor
from smo.observer import build_observer
from state_builder import build_state
from uav_env import SimUAV, UAVEnv


def build_env(cfg):
    env_cfg = cfg["environment"]

    sensor = RadiationSensor(
        noise_std=env_cfg["sensor"].get("noise_std")
    )
    if env_cfg["sensor"].get("record_readings", False):
        readings = {
            "true_readings": [],
            "measurements": [],
            "observed_readings": []
        }
    else:    readings = None

    state_builder = build_state(cfg["state"])
    reward_fn = build_reward(cfg["reward"])

    uav = SimUAV(size=env_cfg["size"])
    
    # Get history_length from state config, default to 3
    history_length = cfg["state"].get("history_length", 3)
    return UAVEnv(
        uav=uav,
        state_builder=state_builder,
        reward_fn=reward_fn,
        sensor=sensor,
        size=env_cfg["size"],
        max_steps=cfg["experiment"]["max_steps"],
        source_intensity=env_cfg["source_intensity"],
        observer=build_observer(env_cfg.get("observer")),
        history_length=history_length,
        readings=readings,
    )
