import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import seaborn as sns
from matplotlib.patches import Circle
from sklearn.utils.extmath import randomized_svd

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
        ax.plot(times, mags, label=f"Re={Re}", color='tab:blue')
        ax.set_ylabel("Velocity Magnitude")
        ax.set_title(f"Snapshot Magnitude Over Time — Re={Re}")
        ax.grid(True)
        ax.legend()

    axes[-1].set_xlabel("Time (s)")
    plt.suptitle("Snapshot velocity magnitudes over time across each parameter", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.show()



def compute_pod(snapshot_dict, Re_list, n_components=100):
    snapshot_matrix = np.hstack([snapshot_dict[Re] for Re in Re_list])
    U, s, Vh = randomized_svd(snapshot_matrix, n_components=n_components)
    normalized_energy = s**2 / np.sum(s**2)
    cumulative_energy = np.cumsum(normalized_energy)
    residual_content = 1 - cumulative_energy
    return cumulative_energy, residual_content

def get_thresholds(residual_content, threshold=0.99, tau_list=None):
    if tau_list is None:
        tau_list = [1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7]
    num_modes_99 = np.searchsorted(1 - residual_content, threshold) + 1
    tau_ranks = [(tau, np.where(residual_content > tau)[0][-1] + 1) for tau in tau_list]
    return num_modes_99, tau_ranks


def plot_cumulative_energy(cumulative_energy, threshold, num_modes_99):
    plt.figure(figsize=(10, 5))
    plt.plot(np.arange(1, len(cumulative_energy) + 1), cumulative_energy, marker='o', label='Cumulative Energy')
    plt.axhline(threshold, color='red', linestyle='--', label='99% Threshold')
    plt.axvline(num_modes_99, color='green', linestyle='--', label=f'{num_modes_99} Modes')
    plt.text(0.65, 0.15,
             f"Modes for 99% energy: {num_modes_99}",
             transform=plt.gca().transAxes,
             fontsize=10, color='green')

    x_margin = 10
    x_min = max(1, num_modes_99 - x_margin)
    x_max = min(len(cumulative_energy), num_modes_99 + x_margin)
    plt.xlim(x_min, x_max)
    plt.ylim(threshold - 0.05, 1.01)

    plt.title("Cumulative Energy Retained by POD Modes", fontsize=13)
    plt.xlabel("Number of Modes")
    plt.ylabel("Cumulative Energy")
    plt.grid(True)
    plt.legend(fontsize=9)
    plt.tight_layout()
    plt.show()


def plot_residual_energy(residual_content, tau_ranks):
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(np.arange(1, len(residual_content) + 1), residual_content, marker='o', color='tab:blue', label='Residual Energy')

    colors = plt.cm.viridis(np.linspace(0, 1, len(tau_ranks)))
    box_x = 0.75
    box_y = 0.95
    line_spacing = 0.06

    for i, (_, idx) in enumerate(tau_ranks):
        y_val = residual_content[idx - 1]
        ax.plot(idx, y_val, 'o', color=colors[i], markersize=8)
        ax.plot([box_x], [box_y - i * line_spacing], marker='o', color=colors[i],
                transform=ax.transAxes, markersize=6)
        ax.text(box_x + 0.02, box_y - i * line_spacing,
                f"Rank {idx}: Residual = {y_val:.1e}",
                transform=ax.transAxes,
                fontsize=10,
                verticalalignment='center',
                color='black')

    ax.set_title("Residual Information Content (1 − Cumulative Energy)", fontsize=13)
    ax.set_xlabel("Number of Modes")
    ax.set_ylabel("Residual Energy")
    ax.grid(True)
    ax.legend()
    plt.tight_layout()
    plt.show()


def plot_dmd_modal_comparison(
    pdmd_models,
    Re_list,
    sampled_times_dict,
    cached_modal_coeffs,
    Re_target,
    n_modes_to_plot=5
):
    """
    Plot comparison of DMD modal coefficients vs true coefficients
    for a given training Reynolds number across noise levels.

    Parameters
    ----------
    pdmd_models : dict
        Dictionary of ParametricDMD models keyed by noise level.
    Re_list : list or array
        List of Reynolds numbers used in training.
    sampled_times_dict : dict
        Mapping from Re -> sampled time vector.
    cached_modal_coeffs : dict or list
        Cached true modal coefficients from clean (0% noise) model.
    Re_target : int or float
        Reynolds number to inspect.
    n_modes_to_plot : int, optional
        Number of modes to plot (default = 5).
    """

    # Index of target Re in training list
    i = np.where(np.array(Re_list) == Re_target)[0][0]
    times = np.array(sampled_times_dict[Re_target], dtype=float)

    # True coefficients from clean (0% noise) model
    modal_true = cached_modal_coeffs[0][i]

    for level, model in pdmd_models.items():
        print(f"\nModal Coefficient Comparison at Re = {Re_target} (Noise Level: {level}%)")

        # Reconstruction from current noise-level model
        modal_dmd = model.training_modal_coefficients[i]
        if modal_dmd.shape != modal_true.shape:
            raise ValueError(
                f"Shape mismatch: recon={modal_dmd.shape} vs true={modal_true.shape}"
            )

        # Plot
        fig, axes = plt.subplots(n_modes_to_plot, 1,
                                 figsize=(12, 3 * n_modes_to_plot),
                                 sharex=True)
        for mode in range(n_modes_to_plot):
            ax = axes[mode]
            ax.plot(times, modal_true[mode],
                    label=f"True Mode {mode}", color="tab:blue")
            ax.plot(times, modal_dmd[mode],
                    label=f"ParametericDMD Mode {mode}",
                    linestyle="--", color="tab:orange")
            ax.set_ylabel("Amplitude")
            ax.set_title(f"Mode {mode} — Re = {Re_target}")
            ax.grid(True)
            ax.legend()

        axes[-1].set_xlabel("Time (s)")
        plt.suptitle(
            f"DMD Modal Coefficients Within The Training Window\n"
            f"Training Parameter vs True Data (Noise {level}%)",
            fontsize=16
        )
        plt.tight_layout(rect=[0, 0, 1, 0.97])
        plt.show()



def plot_dmd_modal_fft_comparison(
    pdmd_models,
    Re_list,
    cached_modal_coeffs,
    Re_target,
    n_modes_to_plot=4,
    dt=0.01
):
    """
    Plot FFT comparison of DMD modal coefficients vs true coefficients
    for a given training Reynolds number across noise levels.

    Parameters
    ----------
    pdmd_models : dict
        Dictionary of ParametricDMD models keyed by noise level.
    Re_list : list or array
        List of Reynolds numbers used in training.
    cached_modal_coeffs : dict or list
        Cached true modal coefficients from clean (0% noise) model.
    Re_target : int or float
        Reynolds number to inspect.
    n_modes_to_plot : int, optional
        Number of modes to plot (default = 4).
    dt : float, optional
        Physical timestep used for FFT frequency axis (default = 0.01).
    """

    # Index of target Re in training list
    i = np.where(np.array(Re_list) == Re_target)[0][0]

    # True coefficients from clean (0% noise) model
    modal_true = cached_modal_coeffs[0][i]   # shape (r, n_time)

    n_time = modal_true.shape[1]
    freqs = np.fft.rfftfreq(n_time, d=dt)

    for level, model in pdmd_models.items():
        print(f"\nFFT of Training Modal Coefficients at Re = {Re_target} (Noise Level: {level}%)")

        # Modal coefficients from this noise-level model
        modal_dmd = model.forecasted_modal_coefficients[i]
        if modal_dmd.shape != modal_true.shape:
            raise ValueError(
                f"Shape mismatch: recon={modal_dmd.shape} vs true={modal_true.shape}"
            )

        # Plot FFTs
        fig, axes = plt.subplots(n_modes_to_plot, 1,
                                 figsize=(10, 2.5 * n_modes_to_plot),
                                 sharex=True)
        fig.suptitle(
            f"FFT of DMD Modal Coefficients Within The Training Window\n"
            f"Training Parameter vs True Data (Noise {level}%)",
            fontsize=14
        )

        for mode in range(n_modes_to_plot):
            fft_true = np.abs(np.fft.rfft(np.asarray(modal_true[mode], dtype=np.float64)))
            fft_dmd  = np.abs(np.fft.rfft(np.asarray(modal_dmd[mode],  dtype=np.float64)))

            ax = axes[mode]
            ax.plot(freqs, fft_true, color="tab:blue", label="True FFT")
            ax.plot(freqs, fft_dmd, color="tab:orange", linestyle="--",
                    label=f"ParametericDMD FFT")
            ax.set_ylabel("Spectral Amplitude")
            ax.set_title(f"Mode {mode} — Re {Re_target}")
            ax.grid(True)
            ax.legend()

        axes[-1].set_xlabel("Frequency (Hz)")
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.show()

def plot_dmd_reconstruction_error(
    pdmd_models,
    Re_list,
    sampled_times_dict,
    snapshot_processed_dict,
    norm_scales,
    mean_flow,
    Re_target
):
    """
    Plot relative L2 reconstruction error over time for a given training Reynolds number.

    Parameters
    ----------
    pdmd_models : dict
        Dictionary of ParametricDMD models keyed by noise level.
    Re_list : list or array
        List of Reynolds numbers used in training.
    sampled_times_dict : dict
        Mapping from Re -> sampled time vector.
    snapshot_processed_dict : dict
        Mapping from Re -> processed snapshots (space_dim, n_time).
    norm_scales : dict
        Normalization scaling factors keyed by Re.
    mean_flow : ndarray
        Mean flow vector to restore physical space.
    Re_target : int or float
        Reynolds number to inspect.
    """

    # Index of target Re in training list
    i = list(Re_list).index(Re_target)

    # Time vector and true snapshots
    time_vec = np.array(sampled_times_dict[Re_target], dtype=float)
    snapshots_true = snapshot_processed_dict[Re_target]
    n_time = snapshots_true.shape[1]

    # Plot setup
    fig, ax = plt.subplots(figsize=(10, 4))

    for level, model in pdmd_models.items():
        # Forecasted modal coefficients and reconstruction
        modal_dmd = model.forecasted_modal_coefficients[i]    # (n_modes, n_time)
        snapshots_dmd = model._spatial_pod.expand(modal_dmd)  # (space_dim, n_time)

        # Compute relative L2 errors over time
        l2_rel_errors = []
        for t in range(n_time):
            U_true = snapshots_true[:, t] * norm_scales[Re_target] + mean_flow
            U_dmd  = snapshots_dmd[:, t] * norm_scales[Re_target] + mean_flow
            diff = U_true - U_dmd
            l2_err = np.linalg.norm(diff) / np.linalg.norm(U_true)
            l2_rel_errors.append(l2_err)

        ax.plot(time_vec, l2_rel_errors, lw=1.5, label=f"{level}% noise")

    # Final plot formatting
    ax.set_title(
        f"DMD Reconstruction Error For Training Parameters Within The Training Window\nRe = {Re_target}",
        fontsize=14
    )
    ax.set_xlabel("Time (s)", fontsize=12)
    ax.set_ylabel(r"Relative $L^2$ Error", fontsize=12)
    ax.grid(True, linestyle='--', linewidth=0.6, alpha=0.5)
    ax.legend(title="Noise Level", fontsize=10)
    plt.tight_layout()
    plt.show()



def plot_dmd_forecast_error(
    pdmd_models,
    Re_target,
    snapshot_future_dict,
    sampled_times_future_dict,
    norm_scales,
    mean_flow
):
    """
    Plot relative L2 reconstruction error over time in the forecast window
    for a given training Reynolds number.

    Parameters
    ----------
    pdmd_models : dict
        Dictionary of ParametricDMD models keyed by noise level.
    Re_target : int or float
        Reynolds number to inspect.
    snapshot_future_dict : dict
        Mapping from Re -> future snapshots (space_dim, n_future_times).
    sampled_times_future_dict : dict
        Mapping from Re -> future time vector.
    norm_scales : dict
        Normalization scaling factors keyed by Re.
    mean_flow : ndarray
        Mean flow vector to restore physical space.
    """

    true_future = snapshot_future_dict[Re_target]  # (space_dim, n_future_times)
    times_future = np.array(sampled_times_future_dict[Re_target], dtype=float)

    plt.figure(figsize=(10, 4))

    for level, pdmd in pdmd_models.items():
        # Forecasted snapshots for the future window
        forecasted_snapshots = pdmd.reconstructed_data[0]  # (space_dim, n_times)
        forecasted_future = forecasted_snapshots[:, :true_future.shape[1]]

        l2_rel_errors = []
        for t in range(true_future.shape[1]):
            # Restore both true and forecasted fields to physical space
            U_true = true_future[:, t] * norm_scales[Re_target] + mean_flow
            U_dmd  = forecasted_future[:, t] * norm_scales[Re_target] + mean_flow
            diff   = U_true - U_dmd

            # Relative L2 error
            l2_err = np.linalg.norm(diff) / np.linalg.norm(U_true)
            l2_rel_errors.append(l2_err)

        plt.plot(times_future, l2_rel_errors, lw=1.5, label=f"{level}% noise")

    plt.xlabel("Time (s)")
    plt.ylabel("Relative L2 Error")
    plt.title(
        f"DMD Reconstruction Error For Training Parameters Within The Forecast Window\nRe = {Re_target}"
    )
    plt.legend(title="Noise Level")
    plt.grid(True, linestyle="--", linewidth=0.6, alpha=0.5)
    plt.tight_layout()
    plt.show()


def plot_interpolated_vs_true_modal_coeffs(
    pdmd_models,
    data,
    Re_test,
    mean_flow,
    dt_phys=0.01,
    t0_phys=15.0,
    n_modes_to_plot=6,
    tol=0.05
):
    """
    Compare interpolated vs true modal coefficients for an unseen Reynolds number
    across noise levels.

    Parameters
    ----------
    pdmd_models : dict
        Dictionary of ParametricDMD models keyed by noise level.
    data : dict
        Data dictionary containing snapshot_dict and sampled_times_dict.
    Re_test : int or float
        Unseen Reynolds number to test.
    mean_flow : ndarray
        Mean flow vector used for preprocessing.
    dt_phys : float, optional
        Physical timestep (default = 0.01).
    t0_phys : float, optional
        Starting physical time (default = 15.0).
    n_modes_to_plot : int, optional
        Number of modes to plot (default = 6).
    tol : float, optional
        Time matching tolerance (default = 0.05).
    """

    # Get test snapshots and times
    snapshot_test = data["snapshot_dict"][Re_test]
    sampled_times_test = data["sampled_times_dict"][Re_test]

    for level, model in pdmd_models.items():
        # Step 1: Forecast time vector
        n_forecast = model.interpolated_modal_coefficients[0].shape[1]
        forecast_times = np.arange(n_forecast) * dt_phys + t0_phys

        # Step 2: Match forecast times to test snapshot times
        sampled_times_test_float = np.array(sampled_times_test, dtype=float)
        matched_true_indices, forecast_indices = [], []
        for i, t in enumerate(forecast_times):
            diffs = np.abs(sampled_times_test_float - t)
            if np.min(diffs) < tol:
                matched_true_indices.append(np.argmin(diffs))
                forecast_indices.append(i)

        if not matched_true_indices:
            print(f"❌ No valid matches found for noise {level}% at Re={Re_test}")
            continue
        print(f"Noise {level}%: matched {len(matched_true_indices)} time points.")

        # Step 3: Preprocess true snapshots
        snapshot_test_aligned = snapshot_test[:, matched_true_indices].copy()
        snapshot_test_aligned -= mean_flow[:, np.newaxis]
        snapshot_test_aligned /= np.linalg.norm(snapshot_test_aligned)

        # Step 4: Extract aligned modal coefficients
        interpolated_modal_coeffs = model.interpolated_modal_coefficients[0]
        true_modal_coeffs_aligned = model._spatial_pod.reduce(snapshot_test_aligned)
        aligned_times = [forecast_times[f] for f in forecast_indices]

        # Step 5: Plot each mode separately
        fig, axes = plt.subplots(n_modes_to_plot, 1,
                                 figsize=(10, 2.8 * n_modes_to_plot),
                                 sharex=True)

        for mode_idx in range(n_modes_to_plot):
            ax = axes[mode_idx]
            ax.plot(aligned_times,
                    interpolated_modal_coeffs[mode_idx, forecast_indices].real,
                    label=f"Interpolated ParametericDMD Mode {mode_idx}",
                    linewidth=2, color="tab:blue")
            ax.plot(aligned_times,
                    true_modal_coeffs_aligned[mode_idx].real,
                    linestyle=':', label=f"True Mode {mode_idx}",
                    linewidth=2, color="tab:orange")
            ax.set_ylabel("Amplitude")
            ax.set_title(f"Mode {mode_idx} — Interpolated vs True")
            ax.grid(True)
            ax.legend()

        axes[-1].set_xlabel("Physical Time [s]")
        Re_interp = model.parameters[0, 0]
        fig.suptitle(
            f"DMD Modal Coefficient Comparison Within The Forecast Window\n"
            f"Interpolated Parameter vs True Re = {Re_interp} — Noise {level}%",
            fontsize=16
        )
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.show()


def plot_fft_interpolated_vs_true_modal_coeffs(
    pdmd_models,
    snapshot_test,
    sampled_times_test,
    mean_flow,
    Re_interp,
    dt_phys=0.01,
    t0_phys=15.0,
    n_modes_to_plot=6,
    tol=0.05
):
    """
    Compare FFT of interpolated vs true modal coefficients for an unseen Reynolds number
    across noise levels.

    Parameters
    ----------
    pdmd_models : dict
        Dictionary of ParametricDMD models keyed by noise level.
    snapshot_test : ndarray
        Test snapshots at Re_test (space_dim, n_times).
    sampled_times_test : array-like
        Sampled physical times for Re_test snapshots.
    mean_flow : ndarray
        Mean flow vector used for preprocessing.
    Re_interp : float
        Interpolated Reynolds number (from model.parameters).
    dt_phys : float, optional
        Physical timestep (default = 0.01).
    t0_phys : float, optional
        Starting physical time (default = 15.0).
    n_modes_to_plot : int, optional
        Number of modes to plot (default = 6).
    tol : float, optional
        Time matching tolerance (default = 0.05).
    """

    sampled_times_test_float = np.array(sampled_times_test, dtype=float)

    for level, model in pdmd_models.items():
        # Step 1: Forecast time vector
        interpolated_modal_coeffs = model.interpolated_modal_coefficients[0]  # (r, n_forecast)
        n_forecast = interpolated_modal_coeffs.shape[1]
        forecast_times = np.arange(n_forecast) * dt_phys + t0_phys

        # Step 2: Match forecast times to test snapshot times
        matched_true_indices, forecast_indices = [], []
        for i, t in enumerate(forecast_times):
            diffs = np.abs(sampled_times_test_float - t)
            if np.min(diffs) < tol:
                matched_true_indices.append(np.argmin(diffs))
                forecast_indices.append(i)

        valid_pairs = [(f_idx, t_idx) for f_idx, t_idx in zip(forecast_indices, matched_true_indices)]
        if not valid_pairs:
            print(f"❌ No valid time matches found for noise {level}%.")
            continue
        print(f"Noise {level}%: matched {len(valid_pairs)} time points.")

        # Step 3: Filter and align snapshots using paired indices
        snapshot_test_aligned = snapshot_test[:, [t for _, t in valid_pairs]].copy()
        snapshot_test_aligned -= mean_flow[:, np.newaxis]
        snapshot_test_aligned /= np.linalg.norm(snapshot_test_aligned)

        # Step 4: Project onto training POD basis
        true_modal_coeffs_aligned = model._spatial_pod.reduce(snapshot_test_aligned)

        # Step 5: FFT comparison
        freqs = np.fft.rfftfreq(len(valid_pairs), d=dt_phys)

        fig, axes = plt.subplots(n_modes_to_plot, 1,
                                 figsize=(12, 3.2 * n_modes_to_plot),
                                 sharex=True)
        fig.suptitle(
            f"FFT of DMD Modal Coefficient Comparison Within The Forecast Window\n"
            f"Interpolated Parameter vs True Re = {Re_interp} — Noise {level}%",
            fontsize=18
        )

        for mode_idx in range(n_modes_to_plot):
            ax = axes[mode_idx]

            # Interpolated FFT
            interp_mode = interpolated_modal_coeffs[mode_idx, [f for f, _ in valid_pairs]].real
            fft_interp = np.abs(np.fft.rfft(interp_mode))
            ax.plot(freqs, fft_interp, label="Interpolated ParametericDMD FFT", linewidth=2, color="tab:blue")

            # True FFT
            true_mode = true_modal_coeffs_aligned[mode_idx].real
            fft_true = np.abs(np.fft.rfft(true_mode))
            ax.plot(freqs, fft_true, linestyle=':', label="True FFT", linewidth=2, color="tab:orange")

            ax.set_ylabel("Spectral Amplitude")
            ax.set_title(f"Mode {mode_idx} — Interpolated vs True")
            ax.grid(True)
            ax.legend()

        axes[-1].set_xlabel("Frequency [Hz]")
        plt.tight_layout(rect=[0, 0.03, 1, 0.96])
        plt.show()



def plot_future_window_error_interpolated(
    pdmd_models,
    data,
    Re_test,
    norm_scales,
    mean_flow,
    t_start=15.0,
    t_end=20.0
):
    """
    Plot relative L2 reconstruction error over time in the forecast window
    for an unseen Reynolds number (interpolated case).

    Parameters
    ----------
    pdmd_models : dict
        Dictionary of ParametricDMD models keyed by noise level.
    data : dict
        Data dictionary containing snapshot_dict and sampled_times_dict.
    Re_test : int or float
        Unseen Reynolds number to test.
    norm_scales : dict
        Normalization scaling factors keyed by Re.
    mean_flow : ndarray
        Mean flow vector to restore physical space.
    t_start : float, optional
        Start of forecast window (default = 15.0).
    t_end : float, optional
        End of forecast window (default = 20.0).
    """

    # Full test snapshots and times
    true_test = data["snapshot_dict"][Re_test]
    times_test = np.array(data["sampled_times_dict"][Re_test], dtype=float)

    # Slice to forecast window
    mask_future = (times_test >= t_start) & (times_test <= t_end)
    true_future = true_test[:, mask_future]
    times_future = times_test[mask_future]

    plt.figure(figsize=(10, 5))

    for level, pdmd in pdmd_models.items():
        # Set model parameter to unseen Re
        pdmd.parameters = np.array([[Re_test]])

        # Forecasted snapshots for the future window
        forecasted_snapshots = pdmd.reconstructed_data[0]
        forecasted_future = forecasted_snapshots[:, :true_future.shape[1]]

        # Compute relative L2 errors
        rel_errors = []
        for i in range(true_future.shape[1]):
            U_true = true_future[:, i] * norm_scales[Re_test] + mean_flow
            U_dmd  = forecasted_future[:, i] * norm_scales[Re_test] + mean_flow
            diff   = U_true - U_dmd
            l2_err = np.linalg.norm(diff) / np.linalg.norm(U_true)
            rel_errors.append(l2_err)

        plt.plot(times_future, rel_errors, lw=2, label=f"{level}% noise")

    plt.xlabel("Time (s)")
    plt.ylabel("Relative L2 Error")
    plt.title(
        f"DMD Reconstruction Error For Interpolated Parameter Within The Forecast Window\nRe = {Re_test}"
    )
    plt.legend(title="Noise Level")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.show()






































