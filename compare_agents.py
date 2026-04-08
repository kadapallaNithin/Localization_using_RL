import numpy as np
import matplotlib.pyplot as plt
from src_finder import sim_rl_model
import json
import uuid, os

run_id = None
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

def evaluate_across_variable(variable, values, agents, episodes=30, save=True):
    results = {agent: {"steps": [], "ncd": [], "rewards":[], 'dones':[]} for agent in agents}

    os.makedirs('run', exist_ok=True)
    if "DQN" in agents:
        if not os.path.exists('dqn_model.pth'):
            raise Exception('dqn_model does not exist')
    if "Q-learning" in agents:
        if not os.path.exists('qlearning.pkl'):
            raise Exception('qlearning does not exist')
    # print(f"Running {agent}...")
    for value in values:
        kwargs = {
            'source_intensity': 2e7,
            'size': 200,
        }
        if variable == 'source_intensity':
            kwargs['source_intensity'] = value
        else:
            kwargs['size'] = value
        print(f"\n=== Evaluating {variable}: {value} ===")
        rewards, steps_list, norm_dists, dones = sim_rl_model(
            models=agents,
            episodes=episodes,
            is_training=False,
            load_if_exists=True,
            plot_window=1,
            # render=True,
            folder=f"run/{variable}_{value}",
            include_quadrant=False,
            #not_found_ncd=kwargs['size']*4,
            **kwargs
        )
        for i, agent in enumerate(agents):
            avg_reward = np.mean(rewards[i])
            avg_steps = np.mean(steps_list[i])
            avg_ncd = np.mean(norm_dists[i])
            results[agent]["rewards"].append(avg_reward)
            results[agent]["steps"].append(avg_steps)
            results[agent]["ncd"].append(avg_ncd)
            results[agent]["dones"].append(dones[i])
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
    for i in range(2):
        if i == 1:
            del results['Uniform']
        for metric, metric_name, ylabel in [
            # ("rewards","Average Cumulative Reward","Average Cumulative Reward"),
            ("steps","Average Steps","Average Steps to Reach Source"),
            ("ncd","Normalized Distance","Normalized Cumulative Distance"),
            # ("dones","Dones","Number of times source found"),
            ]:
            plt.figure(figsize=(6, 5))
            for (agent, metrics) in results.items():
                marker, color = graph_params[agent]
                plt.plot(area_sizes, metrics[metric], marker=marker, color=color, label=agent)
            plt.title(f"{metric_name} vs {name}")
            plt.xlabel(name)
            plt.ylabel(ylabel)
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            if i == 1:
                plt.savefig(f"run/{run_id}/no_uniform_{metric}_vs_{snake_name}.png")
            else:
                plt.savefig(f"run/{run_id}/{metric}_vs_{snake_name}.png")
            # plt.show()
            # plt.clf()

if __name__ == "__main__":
    area_sizes = []
    source_intensities = []

    area_sizes = [100, 150, 200, 250, 300, 350, 400]
    # area_sizes = [200]
    # area_sizes = [200,300]
    # source_intensities = [2e7]
    source_intensities = np.arange(2e7, 5.5e7, 0.5e7)
    agents = ["Q-learning",  "Uniform", "Grad"] #"DQN",
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
    num_episodes = 50
    # num_episodes = 1
    save = True
    for _ in range(num_runs):
        run_id, save = 'temp', True
        # run_id = None
        update_run_id(run_id)
        if area_sizes:
            results = evaluate_across_variable('area', area_sizes, agents, episodes=num_episodes, save=save)
            plot_performance('Area Size', area_sizes, results)

        if len(source_intensities):
            results = evaluate_across_variable('source_intensity', source_intensities, agents, episodes=num_episodes)
            plot_performance('Source Intensity', source_intensities, results)
