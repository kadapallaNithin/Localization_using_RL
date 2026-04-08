import matplotlib.pyplot as plt
import numpy as np

def plot_metrics(rewards, steps, label, filename, window=100):
    # Compute moving average
    def moving_avg(x, w):
        return np.convolve(x, np.ones(w)/w, mode='valid')

    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    if window > 1:
        rewards_smooth = moving_avg(rewards, window)
        plt.plot(rewards_smooth)
    else:
        plt.scatter(range(len(rewards)), rewards)
    plt.title(f"{label} Rewards (mean {window} episodes)")
    plt.xlabel("Episode")
    plt.ylabel("Mean Reward")

    plt.subplot(1, 2, 2)
    if window > 1:
        steps_smooth = moving_avg(steps, window)
        plt.plot(steps_smooth)
    else:
        plt.scatter(range(len(steps)), steps)
    plt.title(f"{label} Steps to Source (mean {window} episodes)")
    plt.xlabel("Episode")
    plt.ylabel("Mean Steps")

    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
