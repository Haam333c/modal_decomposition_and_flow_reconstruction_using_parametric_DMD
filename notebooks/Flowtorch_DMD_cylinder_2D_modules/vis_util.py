import numpy as np
from sklearn.utils.extmath import randomized_svd
import matplotlib.pyplot as plt

def plot_snapshot_magnitudes(snapshot_dict, sampled_times_dict, Re_list):
    """
    Plots raw snapshot velocity magnitudes over time for each Reynolds number.
    """
    n_re = len(Re_list)
    fig, axes = plt.subplots(n_re, 1, figsize=(12, 3 * n_re), sharex=True)

    for i, Re in enumerate(Re_list):
        # Global snapshot magnitudes (L2 norm over space at each sampled time)
        mags = np.linalg.norm(snapshot_dict[Re], axis=0)

        # Plot
        times = np.array(sampled_times_dict[Re], dtype=float)
        ax = axes[i]
        ax.plot(times, mags)
        ax.set_ylabel("Velocity Magnitude", fontsize=12)
        ax.set_title(f"$Re$ = {Re}", fontsize=14, pad=6)
        ax.grid(True, alpha=0.6)
        ax.legend(fontsize=16)
    fig.align_ylabels(axes)
    axes[-1].set_xlabel("$t$ (Time in seconds)", fontsize=16)
    plt.suptitle("Snapshot Velocity Magnitudes For DMD Training Parameters",
                 fontsize=18, y = 0.96)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()

def compute_SVD_per_re(snapshot_dict, Re_list, n_components=100):
    """
    Compute SVD separately for each Reynolds number.

    Returns:
    - SVD_results: dict keyed by Re, each entry contains
        {
            "cumulative_energy": array,
            "residual_content": array,
            "singular_values": array
        }
    """
    SVD_results = {}
    for Re in Re_list:
        X = snapshot_dict[Re]  # snapshot matrix for this Re, shape (space_dim, n_time)
        U, s, Vh = randomized_svd(X, n_components=n_components)
        normalized_energy = s**2 / np.sum(s**2)
        cumulative_energy = np.cumsum(normalized_energy)
        residual_content = 1 - cumulative_energy
        SVD_results[Re] = {
            "cumulative_energy": cumulative_energy,
            "residual_content": residual_content,
            "singular_values": s
        }
    return SVD_results


def get_thresholds(residual_content, threshold=0.99, tau_list=None):
    if tau_list is None:
        tau_list = [1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7]
    num_modes_99 = np.searchsorted(1 - residual_content, threshold) + 1
    tau_ranks = [(tau, np.where(residual_content > tau)[0][-1] + 1) for tau in tau_list]
    return num_modes_99, tau_ranks


def plot_cumulative_energy(SVD_results, threshold=0.99):
    """
    Plot cumulative energy curves for each Re separately.
    """
    plt.figure(figsize=(10, 6))
    for Re, res in SVD_results.items():
        cumulative_energy = res["cumulative_energy"]
        num_modes_99, _ = get_thresholds(res["residual_content"], threshold)
        plt.plot(np.arange(1, len(cumulative_energy) + 1),
                 cumulative_energy, marker='o', label=f'Re={Re}')
        plt.axvline(num_modes_99, linestyle='--', alpha=0.5,
                    label=f'Re={Re}: {num_modes_99} modes')

    plt.axhline(threshold, color='red', linestyle='--',
                label=f'{int(threshold*100)}% Threshold')
    plt.title("Cumulative Energy Retained by SVD Modes (per Re)", fontsize=13)
    plt.xlabel("Number of Modes")
    plt.ylabel("Cumulative Energy")
    plt.grid(True)
    plt.legend(fontsize=9)
    plt.tight_layout()
    plt.show()

