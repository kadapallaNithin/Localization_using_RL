import os
import json
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from agents.factory import build_agent
from env import build_env
# from episode_handler import load_episode_initializations
from plot import plot_metrics
# from visualization.pf_visualizer import PFVisualizer
from pprint import pprint
from copy import deepcopy
# import time
from time import time

# DEBUG_LOG_PATH = "/home/nithin/code/RL/seminar/Stage_2/lab/src_finder/.cursor/debug.log"

def set_seed(seed):
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
    except ModuleNotFoundError:
        pass

def get_file_path(cfg, name):
    # print(f"Getting file path for: {name}", cfg["experiment"]["name"])
    return os.path.join(
        cfg["experiment"]["save_dir"],
        cfg["experiment"]["name"],
        name
    )

class Metrics:
    def __init__(self):
        self.rewards = []
        self.steps_list = []
        self.norm_dists = []
        self.dones = 0
        self.intvl_dones = 0
        self.intvl_reward = 0

    def append(self, total_reward, steps, done, norm_dist):
        self.rewards.append(total_reward)
        self.intvl_reward += total_reward
        self.steps_list.append(steps)

        if done:
            self.dones += 1
            self.intvl_dones += 1

        self.norm_dists.append(norm_dist)

    def print_intvl(self, intvl_size):
        print(
            f"AvgReward {self.intvl_reward/intvl_size:8.3f} | "
            f"Dones {self.intvl_dones}"
        )
        self.intvl_reward = self.intvl_dones = 0


    def save(self, cfg):
        metrics = self.to_dict(cfg)

        file_path = get_file_path(cfg, "metrics.json")
        with open(file_path, "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"Results saved to: {file_path}")

    def to_dict(self, cfg=None):
        metrics = {
            "config": cfg,
            "rewards": self.rewards,
            "steps": self.steps_list,
            "norm_dists": self.norm_dists,
            "dones": self.dones,
        }
        return metrics

    def plot(self, cfg):
        plot_metrics(
            self.rewards,
            self.steps_list,
            cfg["agent"]["type"],
            get_file_path(cfg, "training.png"),
            window=100,
        )

def _reset_parallel_slot(env, cfg, episode_idx, episode_inits):
    expt_cfg = cfg["experiment"]
    if episode_inits is not None:
        state, info = env.reset(options=episode_inits[episode_idx])
    else:
        state, info = env.reset(seed=expt_cfg["seed"] + episode_idx)

    start_pos = np.array(info["pos"])
    return {
        "episode": episode_idx,
        "state": state,
        "total_reward": 0.0,
        "steps": 0,
        "start_pos": start_pos,
        "prev_pos": start_pos.copy(),
        "goal_pos": np.array(env.source_pos),
        "cumulative_distance": 0.0,
    }


def _step_parallel_env(args):
    env, action = args
    return env.step(int(action))

prev_time = time()

def _run_parallel_dqn_training(cfg, agent, first_env, episode_inits=None):
    global prev_time
    expt_cfg = cfg["experiment"]
    episodes = expt_cfg["episodes"]
    max_steps = expt_cfg["max_steps"]
    parallel_envs = max(1, min(int(expt_cfg.get("parallel_envs", 1)), episodes))
    print_intvl = expt_cfg["print_interval"]

    envs = [first_env] + [build_env(cfg) for _ in range(parallel_envs - 1)]
    slots = [None] * parallel_envs
    metrics = Metrics()
    next_episode = 0
    completed = 0

    for i in range(parallel_envs):
        slots[i] = _reset_parallel_slot(envs[i], cfg, next_episode, episode_inits)
        next_episode += 1

    print(f"Using {parallel_envs} parallel DQN environments.")
    try:
        with ThreadPoolExecutor(max_workers=parallel_envs) as executor:
            while completed < episodes:
                active_indices = [i for i, slot in enumerate(slots) if slot is not None]
                states = [slots[i]["state"] for i in active_indices]
                if hasattr(agent, "act_batch"):
                    actions = agent.act_batch(states)
                else:
                    actions = [agent.act(state) for state in states]

                results = list(executor.map(
                    _step_parallel_env,
                    [(envs[i], actions[j]) for j, i in enumerate(active_indices)]
                ))
                results_by_idx = dict(zip(active_indices, results))

                transitions = []
                for j, i in enumerate(active_indices):
                    slot = slots[i]
                    next_state, reward, done, truncated, info = results_by_idx[i]

                    new_pos = np.array(info["pos"])
                    slot["cumulative_distance"] += np.linalg.norm(new_pos - slot["prev_pos"])
                    slot["prev_pos"] = new_pos
                    slot["total_reward"] += reward
                    slot["steps"] += 1

                    transitions.append((slot["state"], actions[j], reward, next_state, done))
                    slot["state"] = next_state

                if hasattr(agent, "update_batch"):
                    agent.update_batch(transitions)
                else:
                    for transition in transitions:
                        agent.update(*transition)

                for i in active_indices:
                    slot = slots[i]
                    done = bool(results_by_idx[i][2])
                    truncated = bool(results_by_idx[i][3])
                    if not (done or truncated or slot["steps"] >= max_steps):
                        continue

                    if hasattr(agent, "end_episode"):
                        agent.end_episode()

                    start_to_goal = np.linalg.norm(slot["goal_pos"] - slot["start_pos"])
                    norm_dist = slot["cumulative_distance"] / max(start_to_goal, 1e-6)
                    total_reward = slot["total_reward"]
                    steps = slot["steps"]
                    metrics.append(total_reward, steps, done, norm_dist)
                    completed += 1

                    if completed % print_intvl == 0:
                        time_taken = float(time()-prev_time)
                        agent.print_intvl(end=' | ')
                        print(
                            f"te {time_taken:5.2f} | Epi {completed:5d} | "
                            f"Reward {total_reward:7.2f} | "
                            f"Steps {steps:3d} | ",
                            end=''
                        )
                        prev_time = time()
                        metrics.print_intvl(print_intvl)

                    if next_episode < episodes:
                        slots[i] = _reset_parallel_slot(envs[i], cfg, next_episode, episode_inits)
                        next_episode += 1
                    else:
                        slots[i] = None
    finally:
        for env in envs[1:]:
            env.close()

    return metrics


def run_experiment(cfg, episode_init_csv=None, episode_inits=None):
    # ------------------ setup ------------------
    expt_cfg = cfg["experiment"]
    set_seed(expt_cfg["seed"])
    pf_viz = None
    # pf_viz = PFVisualizer(
    #     show_colorbar=expt_cfg.get("pf_viz_colorbar", True)
    # )

    # if episode_init_csv is not None:
    #     episode_inits = load_episode_initializations(episode_init_csv)

    os.makedirs(expt_cfg["save_dir"], exist_ok=True)
    dir_path = get_file_path(cfg,'')
    os.makedirs(dir_path, exist_ok=True)

    env = build_env(cfg)
    if expt_cfg.get("verbose", True):
        pprint(cfg)
    if cfg["agent"]["type"] == "Supervised":
        cfg["agent"]["params"]['student_cfg']['agent']['params'].setdefault("path", get_file_path(cfg, 'model'))
    else:
        cfg["agent"]["params"].setdefault("path", get_file_path(cfg, 'model'))
    agent = build_agent(
        cfg["agent"],
        state_dim=env.observation_space.shape[0],
        action_dim=env.action_space.n,
        env_size=cfg["environment"]["size"],
        state_builder_type=cfg["state"]["type"],
    )
    # grad_agent = build_agent(
    #     {'type':'Grad'},
    #     state_dim=env.observation_space.shape[0],
    #     action_dim=env.action_space.n,
    #     env_size=cfg["environment"]["size"],
    # )

    episodes = expt_cfg["episodes"]
    max_steps = expt_cfg["max_steps"]
    render = expt_cfg["render"]
    training = expt_cfg.get("training", True)

    if not training and hasattr(agent, "epsilon"):
        agent.epsilon = getattr(agent, "eps_min", 0.0)

    metrics = Metrics()
    print_intvl = cfg['experiment']['print_interval']


    print(f"{cfg['experiment']['name']} {episodes}")

    parallel_envs = int(expt_cfg.get("parallel_envs", 1))
    use_parallel_dqn = (
        training
        and parallel_envs > 1
        and not render
        and cfg["agent"]["type"] in {"DQN", "DQNR"}
    )
    if use_parallel_dqn:
        metrics = _run_parallel_dqn_training(cfg, agent, env, episode_inits)
        env.close()
        metrics.save(cfg)
        metrics.plot(cfg)
        agent.save()
        print("Experiment finished.\n")
        return metrics.to_dict(cfg)

    # ------------------ training loop ------------------
    for ep in range(episodes):
        # print(f"Episode {ep+1}/{episodes}")
        if episode_inits is not None:
            state, info = env.reset(options=episode_inits[ep])
        else:
            state, info = env.reset(seed=expt_cfg["seed"] + ep)
        # state, info = env.reset(options={
            
        # })

        done = truncated = False
        total_reward = 0.0
        steps = 0

        # distance tracking
        start_pos = np.array(info["pos"])
        prev_pos = start_pos.copy()
        goal_pos = np.array(env.source_pos)
        cumulative_distance = 0.0

        def _render(render_state=0):
            belief = agent.get_belief() if hasattr(agent, "get_belief") else None
            env.render(
                alpha=state[0] if len(state) > 0 else 0.3,
                state=render_state,
                pause_interval=expt_cfg.get("pause_interval", 0.01),
                name=expt_cfg["name"],
                belief_visualizer=pf_viz,
                belief=belief,
            )

        if render:
            _render()  # show initial position before first action

        while not (done or truncated):
            if hasattr(agent, "set_uav_position"):
                agent.set_uav_position(info["pos"])
            action = agent.act(state)
            next_state, reward, done, truncated, info = env.step(action)

            # metrics
            new_pos = np.array(info["pos"])
            cumulative_distance += np.linalg.norm(new_pos - prev_pos)
            prev_pos = new_pos

            if training:
                agent.update(state, action, reward, next_state, done) # or truncated

            state = next_state
            total_reward += reward
            steps += 1

            if render:
                _render(1 if done else (-1 if truncated else 0))

            if steps >= max_steps:
                break

        # # #region debug end-of-episode termination (Q-learning)
        # if ep in {0, episodes // 4, episodes // 2, 3 * episodes // 4, episodes - 1}:
        #     try:
        #         epsilon = getattr(agent, "epsilon", None)
        #         payload = {
        #             "runId": "iter2-postfix",
        #             "hypothesisId": "H1",
        #             "location": "experiment.py:end_of_episode",
        #             "message": "episode termination summary",
        #             "data": {
        #                 "episode": int(ep),
        #                 "epsilon": None if epsilon is None else float(epsilon),
        #                 "done": bool(done),
        #                 "truncated": bool(truncated),
        #                 "steps": int(steps),
        #                 "total_reward": float(total_reward),
        #                 "episodes": int(episodes),
        #                 "max_steps": int(max_steps),
        #             },
        #             "timestamp": int(time.time() * 1000),
        #         }
        #         with open(DEBUG_LOG_PATH, "a") as f:
        #             f.write(json.dumps(payload) + "\n")
        #     except Exception:
        #         pass
        # # #endregion

        if training and hasattr(agent, "end_episode"):
            agent.end_episode()

        start_to_goal = np.linalg.norm(goal_pos - start_pos)
        norm_dist = cumulative_distance / max(start_to_goal, 1e-6)
        metrics.append(total_reward, steps, done, norm_dist)
        
        if (ep + 1) % print_intvl == 0:
            agent.print_intvl(end=' | ')
            print(
                f"Epi {ep+1:5d} | "
                f"Reward {total_reward:7.2f} | "
                f"Steps {steps:3d} | ",
                end=''
            )
            metrics.print_intvl(print_intvl)


    env.close()

    metrics.save(cfg)
    metrics.plot(cfg)
    if training:
        agent.save()

    print("Experiment finished.\n")
    return metrics.to_dict(cfg)

if __name__ == "__main__":
    experiments = []
    # from config.belief_mpc_pf import EXPERIMENT_CONFIG

    from config.dqn import EXPERIMENT_CONFIG
    experiments.append((deepcopy(EXPERIMENT_CONFIG), [0.0]))

    # from config.ppo_lstm import EXPERIMENT_CONFIG
    # experiments.append((deepcopy(EXPERIMENT_CONFIG), [0.0, 0.02, 0.05]))

    # from config.base import get_config
    # EXPERIMENT_CONFIG = get_config()
    # EXPERIMENT_CONFIG['agent']['type'] = 'Grad'
    # EXPERIMENT_CONFIG["experiment"]["episodes"] = episodes = 6_000
    # # EXPERIMENT_CONFIG['state']['type'] = 'signal_history_quadrant'
    # EXPERIMENT_CONFIG["state"]["type"] = 'signal_history_quadrant'
    # EXPERIMENT_CONFIG["experiment"]["name"] = "grad"

    # from config.manual import EXPERIMENT_CONFIG
    # experiments.append((deepcopy(EXPERIMENT_CONFIG), [0.0]))

    # from config.q_learning import EXPERIMENT_CONFIG
    # experiments.append((deepcopy(EXPERIMENT_CONFIG), [0.02]))

    # from smo.dqn_smo_config import EXPERIMENT_CONFIG

    # raise Exception('check config', EXPERIMENT_CONFIG['agent']['path'])
    # from config.supervised import EXPERIMENT_CONFIG
    # experiments.append((deepcopy(EXPERIMENT_CONFIG), [0.0]))

    # print('dqn')
    # pprint(dqn_exp_cfg)
    # print('ppo')
    # pprint(PPO_EXPERIMENT_CONFIG)


    for cfg, noise_levels in experiments:
        print(noise_levels)
        # pprint(cfg)
        print(cfg['state']['type'])

        for noise in noise_levels:
            cfg["environment"]["sensor"]["noise_std"] = noise
            cfg["experiment"]["name"] += f"_{noise:.3f}".format(noise=noise)
            print(f"Running experiment: {cfg['experiment']['name']}")
            run_experiment(
                cfg,
                # episode_init_csv="episode_init.csv"
            )

    # run_experiment()

    # cfg = json.load(open('results/dqn_noise_0.0_metrics1.json'))
    # rewards = cfg['rewards']
    # steps_list = cfg['steps']
    # plot_metrics(
    #     rewards,
    #     steps_list,
    #     "DQN",
    #     "_training.png",
    #     window=100,
    # )
