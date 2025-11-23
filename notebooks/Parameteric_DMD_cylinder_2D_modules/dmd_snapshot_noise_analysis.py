"""
dmd_snapshot_noise_analysis.py

Utilities to add Gaussian noise to snapshot matrices and visualize
the effect on velocity magnitude over time.
"""


import numpy as np
import matplotlib.pyplot as plt
from ezyrb import POD, RBF
from pydmd import DMD, ParametricDMD

def apply_noise(data: np.ndarray, noise_level: float = 10.0) -> np.ndarray:
    """
    Adds Gaussian noise to a NumPy array.

    Parameters:
    - data: Array of shape (n_Re, space_dim, n_time)
    - noise_level: Percentage of feature-wise std deviation to scale noise

    Returns:
    - Noisy array of same shape
    """
    std = np.std(data, axis=2, keepdims=True)  # std per spatial feature across time
    noise = np.random.randn(*data.shape) * std * (noise_level * 0.01)
    return data + noise


def generate_noisy_snapshots(train_snapshots: np.ndarray,
                             noise_levels=(0, 10, 20, 40)) -> dict:
    """
    Generate noisy versions of training snapshots at specified noise levels.

    Parameters:
    - train_snapshots: ndarray of shape (n_Re, space_dim, n_time)
    - noise_levels: iterable of noise percentages

    Returns:
    - snapshot_noisy_dict: dict keyed by noise level
    """
    snapshot_noisy_dict = {}
    for level in noise_levels:
        if level > 0:
            snapshot_noisy_dict[level] = apply_noise(train_snapshots, noise_level=level)
        else:
            snapshot_noisy_dict[level] = train_snapshots.copy()
    return snapshot_noisy_dict


def plot_noisy_snapshot_magnitudes(train_snapshots: np.ndarray,
                                   snapshot_noisy_dict: dict,
                                   sampled_times_dict: dict,
                                   Re_list: list,
                                   noise_levels=(0, 10, 20, 40),
                                   param_index: int = 0):
    """
    Plot clean vs noisy snapshot magnitudes for a given parameter index.

    Parameters:
    - train_snapshots: ndarray of shape (n_Re, space_dim, n_time)
    - snapshot_noisy_dict: dict of noisy snapshots keyed by noise level
    - sampled_times_dict: dict of time vectors per Re
    - Re_list: list of Reynolds numbers
    - noise_levels: iterable of noise percentages
    - param_index: which Re index to visualize
    """
    fig, axarr = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    axarr = axarr.flatten()

    times = np.array(sampled_times_dict[Re_list[param_index]], dtype=float)

    # Clean snapshot magnitude
    clean = train_snapshots[param_index]  # shape: (space_dim, n_time)
    clean_mag = np.sqrt(np.sum(clean**2, axis=0))  # shape: (n_time,)

    for j, level in enumerate(noise_levels[:4]):
        noisy = snapshot_noisy_dict[level][param_index]  # shape: (space_dim, n_time)
        noisy_mag = np.sqrt(np.sum(noisy**2, axis=0))    # shape: (n_time,)

        axarr[j].plot(times, clean_mag, lw=2, ls="--", c="k", label="Clean Magnitude")
        axarr[j].plot(times, noisy_mag, lw=2, label="Noisy Magnitude", alpha=0.85)
        axarr[j].set_title(rf"$l={level}\%$", fontsize=16)
        axarr[j].tick_params(labelsize=12)
        axarr[j].grid(True, linestyle='--', linewidth=0.6, alpha=0.5)
        axarr[j].set_xlim(times[0], times[-1])

    fig.suptitle("Clean vs Noisy Snapshot Magnitude", fontsize=18, y=1)
    handles, labels = axarr[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2,
               bbox_to_anchor=(0.5, 1.15), fontsize=12)
    fig.supxlabel(r"$t$", y=0.02, fontsize=14)
    fig.text(-0.03, 0.5, "Velocity Magnitude", va='center',
             rotation='vertical', fontsize=14)

    plt.tight_layout()
    plt.show()




from dmd_snapshot_noise_analysis import apply_noise   # reuse your noise function


def train_parametric_dmd_with_noise(train_snapshots: np.ndarray,
                                    Re_list: list,
                                    noise_levels=(0, 10, 20, 40),
                                    pod_rank: int = 30,
                                    pod_method: str = "randomized_svd"):
    """
    Train ParametricDMD models at different noise levels.

    Parameters:
    - train_snapshots: ndarray of shape (n_Re, space_dim, n_time)
    - Re_list: list of Reynolds numbers used for training
    - noise_levels: iterable of noise percentages
    - pod_rank: rank for POD basis
    - pod_method: SVD method for POD

    Returns:
    - pdmd_models: dict {noise_level: ParametricDMD instance}
    - cached_modal_coeffs: dict {noise_level: training modal coefficients}
    - cached_roms: dict {noise_level: POD basis}
    """
    pdmd_models = {}
    cached_modal_coeffs = {}
    cached_roms = {}

    for level in noise_levels:
        print(f"\nTraining ParametricDMD with {level}% noise...")

        # Apply noise to training snapshots
        noisy_snapshots = apply_noise(train_snapshots, noise_level=level) if level > 0 else train_snapshots.copy()

        # Create POD basis, DMD instances, and interpolator
        rom = POD(rank=pod_rank, method=pod_method)
        trained_dmds = [DMD(svd_rank=-1) for _ in Re_list]
        interpolator = RBF()

        # Construct and fit ParametricDMD
        pdmd = ParametricDMD(trained_dmds, rom, interpolator)
        pdmd.fit(noisy_snapshots, np.array(Re_list).reshape(-1, 1))

        # Store model, POD basis, and training modal coefficients
        pdmd_models[level] = pdmd
        cached_modal_coeffs[level] = pdmd.training_modal_coefficients.copy()
        cached_roms[level] = rom

    print("\nNoise-level models trained:", list(pdmd_models.keys()))
    return pdmd_models, cached_modal_coeffs, cached_roms
