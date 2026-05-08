import numpy as np
import matplotlib.pyplot as plt
from experiment import run_experiment
import json
import uuid, os
from copy import deepcopy
from config.base import get_config
from config.dqn import DQN_MODEL_PATH

run_id = None

AGENT_PARAMS = {
    "Q-learning": {
        "eps_min": 2e-3,
        "eps_start": 2e-3,
        # tau omitted: end_episode is never called in eval mode (training=False)
        "load_if_exists": os.path.exists("qlearning.pkl"),
        "path": "qlearning.pkl",
    },
    "DQN": {
        "gamma": 0.99,
        "lr": 1e-3,
        "batch_size": 64,
        "buffer_capacity": 50_000,
        "min_replay_size": 10_000,
        "update_every": 4,
        "target_update_freq": 100,
        "eps_start": 0.002,
        "eps_min": 0.002,
        # tau omitted: end_episode is never called in eval mode (training=False)
        "load_if_exists": os.path.exists(DQN_MODEL_PATH),
        "path": DQN_MODEL_PATH,
    },
    "Uniform": {},
    "Grad": {},
}


def update_run_id(new_id=None):
    global run_id
    if new_id is not None:
        run_id = new_id
    else:
        run_id = str(uuid.uuid4())[:4]
    try:
        os.makedirs(f"run/{run_id}", exist_ok=False)
    except FileExistsError:
        print("Run ID exists, generating a new one...", run_id)
        update_run_id()


def build_compare_config(agent, variable, value, episodes):
    cfg = get_config()
    cfg["experiment"].update({
        "name": f"{agent.lower().replace(' ', '_')}_{variable}_{value}",
        "episodes": episodes,
        "training": False,
        "save_dir": f"run/{run_id}/experiments",
        "print_interval": max(1, episodes),
        "render": False,
        "verbose": False,
    })
    cfg["environment"].update({
        "source_intensity": 2e7,
        "size": 200,
    })
    if variable == "source_intensity":
        cfg["environment"]["source_intensity"] = float(value)
    elif variable == "noise":
        cfg["environment"]["sensor"]["noise_std"] = float(value)
    else:
        cfg["environment"]["size"] = int(value)

    cfg["agent"] = {
        "type": agent,
        "params": deepcopy(AGENT_PARAMS.get(agent, {})),
    }
    if agent in {"Grad", "Uniform", "Q-learning"}:
        cfg["state"]["type"] = "signal_action_quadrant"
    elif agent == "DQN":
        cfg["state"]["type"] = "signal_history_quadrant"
        cfg["state"]["history_length"] = 5
        cfg["state"]["normalize"] = True   # must match training config

    return cfg


def evaluate_across_variable(variable, values, agents, episodes=30, save=True):
    results = {agent: {"steps": [], "ncd": [], "rewards":[], 'dones':[]} for agent in agents}

    os.makedirs('run', exist_ok=True)
    for value in values:
        print(f"\n=== Evaluating {variable}: {value} ===")
        for agent in agents:
            cfg = build_compare_config(agent, variable, value, episodes)
            metrics = run_experiment(cfg)
            avg_reward = float(np.mean(metrics["rewards"]))
            avg_steps = float(np.mean(metrics["steps"]))
            avg_ncd = float(np.mean(metrics["norm_dists"]))
            results[agent]["rewards"].append(avg_reward)
            results[agent]["steps"].append(avg_steps)
            results[agent]["ncd"].append(avg_ncd)
            results[agent]["dones"].append(int(metrics["dones"]))
    if save:
        with open(f"run/{run_id}/{variable}_results.json", "w") as f:
            json.dump(results, f, indent=4)

    return results


def plot_performance(name, area_sizes, results):
    snake_name = name.lower().replace(' ', '_')
    graph_params = {
        "Uniform": ['D', '#d62728'],
        "Q-learning": ['^', '#1f77b4'],
        "DQN": ['o',  '#2ca02c'],
        "Grad": ['x',  '#ff7f0e'],
    }
    # folder = 'run/' + snake_name
    # os.makedirs(folder, exist_ok=True)
    # for include_uniform in [True, False]:
    for include_uniform in [True]:
        plot_results = results
        if not include_uniform:
            plot_results = {agent: metrics for agent, metrics in results.items() if agent != "Uniform"}
            if not plot_results:
                continue
        for metric, metric_name, ylabel in [
            ("rewards", "Average Cumulative Reward", "Average Cumulative Reward"),
            ("steps", "Average Steps", "Average Steps to Reach Source"),
            ("ncd", "Normalized Distance", "Normalized Cumulative Distance"),
            ("dones", "Source Found", "Episodes Source Found"),
        ]:
            plt.figure(figsize=(6, 5))
            for (agent, metrics) in plot_results.items():
                marker, color = graph_params[agent]
                plt.plot(area_sizes, metrics[metric], marker=marker, color=color, label=agent)
            plt.title(f"{metric_name} vs {name}")
            plt.xlabel(name)
            plt.ylabel(ylabel)
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            if include_uniform:
                plt.savefig(f"run/{run_id}/{metric}_vs_{snake_name}.png")
            else:
                plt.savefig(f"run/{run_id}/no_uniform_{metric}_vs_{snake_name}.png")
            plt.close()
            # plt.show()
            # plt.clf()

if __name__ == "__main__":
    # area_sizes = []
    # source_intensities = []

    area_sizes = [100, 150, 200, 250, 300, 350, 400]
    # area_sizes = [200]
    noises = np.arange(0, 0.1, 0.025)
    # area_sizes = [200,300]
    # source_intensities = [2e7]
    source_intensities = np.arange(2e7, 5.5e7, 0.5e7)
    agents = ["DQN", "Q-learning",  "Uniform", "Grad"]
    # agents = ["Uniform"]
    # agents = ["DQN", "Uniform", "Grad"]
    # agents = ["DQN", "Grad"]
    # agents = ["Grad"]
    # for r_id in ['7c60']:
    #     run_id = 'replot'
    #     results = json.load(open(f"run/{r_id}/area_results.json", "r"))
    #     plot_performance('Area', area_sizes, results)

    #     results = json.load(open(f"run/{r_id}/source_intensity_results.json", "r"))
    #     plot_performance('Source Intensity', source_intensities, results)

    num_runs = 1
    num_episodes = 1000
    # num_episodes = 1
    save = True
    for _ in range(num_runs):
        run_id, save = 'temp', True
        # run_id = None
        update_run_id(run_id)
        if area_sizes:
            results = evaluate_across_variable('area', area_sizes, agents, episodes=num_episodes, save=save)
            plot_performance('Area Size', area_sizes, results)

        if len(noises):
            results = evaluate_across_variable('noise', noises, agents, episodes=num_episodes, save=save)
            plot_performance('Noise Level', noises, results)

        if len(source_intensities):
            results = evaluate_across_variable('source_intensity', source_intensities, agents, episodes=num_episodes, save=save)
            plot_performance('Source Intensity', source_intensities, results)
