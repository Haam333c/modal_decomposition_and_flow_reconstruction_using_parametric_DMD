import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import seaborn as sns
from matplotlib.patches import Circle
from sklearn.utils.extmath import randomized_svd


def plot_snapshot_magnitudes(snapshot_dict, sampled_times_dict, Re_list, loader_dict=None, normalize_by_inlet=True):
    """
    Plots velocity magnitude over time for each Reynolds number.
    Optionally normalizes by average inlet velocity computed from loader.
    """
    from preprocess_snapshots import compute_average_inlet_velocity

    n_re = len(Re_list)
    fig, axes = plt.subplots(n_re, 1, figsize=(12, 3 * n_re), sharex=True)

    for i, Re in enumerate(Re_list):
        mags = np.linalg.norm(snapshot_dict[Re], axis=0)

        if normalize_by_inlet and loader_dict is not None:
            U_infty = compute_average_inlet_velocity(loader_dict[Re])
            mags = mags / U_infty  
        times = np.array(sampled_times_dict[Re], dtype=float)
        ax = axes[i]
        ax.plot(times, mags, label=f"Re={Re}", color='tab:blue')
        ax.set_ylabel("Velocity Magnitude")
        ax.set_title(f"Snapshot Magnitude Over Time — Re={Re}")
        ax.grid(True)
        ax.legend()

        threshold = 0.01 * np.max(mags)
        active_indices = np.where(mags > threshold)[0]
        if len(active_indices) > 0:
            t_start = times[active_indices[0]]
            t_end = times[active_indices[-1]]
            margin = 0.5
            ax.set_xlim(t_start - margin, t_end + margin)

    axes[-1].set_xlabel("Time (s)")
    plt.suptitle("Normalized Velocity Field Over Time Across Reynolds Numbers", fontsize=16)
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

    if Re_target not in Re_list:
        raise ValueError(f"$Re$ = {Re_target} not found in Re_list")

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

        # Create subplots for each mode
        fig, axes = plt.subplots(n_modes_to_plot, 1,
                                 figsize=(12, 3 * n_modes_to_plot),
                                 sharex=True)

        for mode in range(n_modes_to_plot):
            ax = axes[mode]
            # Plot true modal coefficients
            line1, = ax.plot(times, modal_true[mode],
                             color="tab:blue", lw=1.5, label="True")
            # Plot reconstructed coefficients
            line2, = ax.plot(times, modal_dmd[mode],
                             color="tab:orange", linestyle="--", lw=1.5,
                             label="ParametricDMD")

            # Labels and titles
            ax.set_ylabel("Amplitude", fontsize=12)
            ax.set_title(f"Mode $\\Phi_{{{mode}}}$", fontsize=12, pad=6)
            ax.grid(True, alpha=0.6)

            # Legend
            ax.legend([line1, line2],
                      ["True", "$ParametricDMD$"],
                      fontsize=11, loc="center left")

            # Force x-axis to match actual data range
            ax.set_xlim(times[0], times[-1])
            ax.set_xticks(np.linspace(times[0], times[-1], 6))

        # Shared labels and layout
        fig.align_ylabels(axes)
        axes[-1].set_xlabel(r"$t$ (Time in seconds)", fontsize=14)

        # Overall title
        plt.suptitle(
            f"Modal Coefficient Dynamics $a_m(t)$— True vs $ParametricDMD$\n"
            f"$Training$ Parameter ($Re$ = {Re_target})\n (Noise {level}%)",
            fontsize=16, y=0.97
        )

        # Layout formatting
        plt.tight_layout(rect=[0, 0, 1, 0.97])
        plt.show()



def plot_dmd_fft_comparison(
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
            f"Frequency Spectrum of Modal Coefficient Dynamics $a_m(t)$ — True vs $ParametricDMD$ \n  $Training$ Parameter ($Re$ = {Re_target}) \n (Noise {level}%)",
            fontsize=14
        )

        for mode in range(n_modes_to_plot):
            fft_true = np.abs(np.fft.rfft(np.asarray(modal_true[mode], dtype=np.float64)))
            fft_dmd  = np.abs(np.fft.rfft(np.asarray(modal_dmd[mode],  dtype=np.float64)))

            ax = axes[mode]
            ax.plot(freqs, fft_true, color="tab:blue", label="True")
            ax.plot(freqs, fft_dmd, color="tab:orange", linestyle="--",
                    label=f"$ParametericDMD$")
            ax.set_ylabel("Spectral Amplitude")
            ax.set_title(f"Mode $\Phi_{{{mode}}}$")
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
    mean_flow_train,
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
    mean_flow_train : ndarray
        Mean flow vector from training set (used to restore physical space).
    Re_target : int or float
        Reynolds number to inspect.
    """

    if Re_target not in Re_list:
        raise ValueError(f"$Re$ = {Re_target} not found in Re_list")

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
            # Restore physical space using training mean flow
            U_true = snapshots_true[:, t] + mean_flow_train
            U_dmd  = snapshots_dmd[:, t] + mean_flow_train
            diff = U_true - U_dmd
            l2_err = np.linalg.norm(diff) / np.linalg.norm(U_true)
            l2_rel_errors.append(l2_err)

        ax.plot(time_vec, l2_rel_errors, lw=1.5, label=f"{level}% noise")

    # Final plot formatting
    ax.set_title(
        f"$ParametricDMD$ Reconstruction Error \n $Training$ Parameter ($Re$ = {Re_target})",
        fontsize=14
    )
    ax.set_xlabel(r"$t$ (Time in seconds)", fontsize=12)
    ax.set_ylabel(r"Relative $L^2$ Error", fontsize=12)
    ax.grid(True, linestyle='--', linewidth=0.6, alpha=0.5)
    ax.legend(title="Noise Level", fontsize=10, loc="upper right")
    plt.tight_layout()
    plt.show()




def plot_dmd_forecast_error(
    pdmd_models,
    Re_target,
    snapshot_future_dict,
    sampled_times_future_dict,
    mean_flow_train
):
    """
    Plot relative L2 reconstruction error over time in the forecast window
    for a given training Reynolds number, and show a summary table of
    average percent errors per noise level.
    """

    # True snapshots in the forecast window
    true_future = snapshot_future_dict[Re_target]  # (space_dim, n_future_times)
    times_future = np.array(sampled_times_future_dict[Re_target], dtype=float)

    # Create two rows: one for plot, one for table
    fig, (ax, ax_table) = plt.subplots(
        2, 1, figsize=(10, 7), gridspec_kw={"height_ratios": [3, 1]}
    )

    # Collect summary errors
    summary_rows = []

    for level, pdmd in pdmd_models.items():
        forecasted_snapshots = pdmd.reconstructed_data[0]
        forecasted_future = forecasted_snapshots[:, :true_future.shape[1]]

        l2_rel_errors = []
        for t in range(true_future.shape[1]):
            U_true = true_future[:, t] + mean_flow_train
            U_dmd  = forecasted_future[:, t] + mean_flow_train
            diff   = U_true - U_dmd
            l2_err = np.linalg.norm(diff) / np.linalg.norm(U_true)
            l2_rel_errors.append(l2_err)

        ax.plot(times_future, l2_rel_errors, lw=1.5, label=f"{level}% noise")

        avg_err_pct = 100 * np.mean(l2_rel_errors)
        summary_rows.append([f"{level}%", f"{avg_err_pct:.2f}%"])

    # Plot formatting
    ax.set_xlabel(r"$t$ (Time in seconds)", fontsize=12)
    ax.set_ylabel(r"Relative $L^2$ Error", fontsize=12)
    ax.set_title(
        f"$ParametricDMD$ Forecast Reconstruction Error \n $Training$ Parameter ($Re$ = {Re_target}) \n (Noise {level}%)",
        fontsize=14
    )
    ax.legend(title="Noise Level", fontsize=10, loc="upper right")
    ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.5)

    # Table axis
    ax_table.axis("off")
    table = ax_table.table(
        cellText=summary_rows,
        colLabels=["Noise Level", "Avg. % Error"],
        cellLoc="center",
        loc="center"
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11)

    plt.tight_layout()
    plt.show()



def plot_dmd_modal_comparison_interp_vs_true(
    pdmd_models,
    snapshot_test,
    loader_test,
    Re_test,
    dt_phys=0.01,
    t0_phys=15.0,
    n_modes_to_plot=6,
    time_window=(3.0, 20.0)
):
    """
    Compare interpolated vs true modal coefficients for an unseen Reynolds number
    across noise levels, using the same time matching logic as the single‑model version.
    """

    # Step 0: Extract sampled_times_test from loader within time_window
    sampled_times_test = [float(t) for t in loader_test.write_times
                          if time_window[0] <= float(t) <= time_window[1]]
    sampled_times_test_float = np.array(sampled_times_test, dtype=float)

    for level, model in pdmd_models.items():
        # Step 1: Forecast time vector (interpolated coefficients)
        interpolated_modal_coeffs = model.interpolated_modal_coefficients[0]
        n_forecast = interpolated_modal_coeffs.shape[1]
        forecast_times = np.arange(n_forecast) * dt_phys + t0_phys

        # Step 2: Slice true snapshots to forecast window
        mask = (sampled_times_test_float >= t0_phys) & (sampled_times_test_float <= forecast_times[-1])
        snapshot_test_window = snapshot_test[:, mask]
        times_test_window = sampled_times_test_float[mask]

        # Step 3: Align lengths (safe even if mask is empty)
        min_len = min(snapshot_test_window.shape[1], n_forecast) if snapshot_test_window.shape[1] > 0 else 0
        snapshot_test_aligned = snapshot_test_window[:, :min_len]
        forecast_times_aligned = forecast_times[:min_len]
        interp_modal_aligned = interpolated_modal_coeffs[:, :min_len]

        # Step 4: Project onto training POD basis (skip if no data)
        if min_len > 0:
            true_modal_coeffs_aligned = model._spatial_pod.reduce(snapshot_test_aligned)

            Re_interp = model.parameters[0, 0]

            # Step 5: Plot modal comparison
            fig, axes = plt.subplots(n_modes_to_plot, 1,
                                     figsize=(12, 2.8 * n_modes_to_plot),
                                     sharex=True)
            fig.suptitle(
                f"Modal Coefficient Dynamics $a_m(t)$ — True vs Interpolated $ParametricDMD$\n"
                f"$Unseen^*$ Parameter (Re={Re_interp}) \n (Noise {level}%)",
                fontsize=16, y=0.97
            )

            for mode_idx in range(n_modes_to_plot):
                ax = axes[mode_idx]
                line1, = ax.plot(forecast_times_aligned,
                                 true_modal_coeffs_aligned[mode_idx].real,
                                 color="tab:blue", lw=1.5, label="True")
                line2, = ax.plot(forecast_times_aligned,
                                 interp_modal_aligned[mode_idx].real,
                                 color="tab:orange", linestyle="--", lw=1.5,
                                 label="$Interpolated ParametricDMD$")

                ax.set_ylabel("Amplitude", fontsize=12)
                ax.set_title(f"Mode $\\Phi_{{{mode_idx}}}$", fontsize=12, pad=6)
                ax.grid(True, alpha=0.6)
                ax.legend([line1, line2], ["True", "Interpolated $ParametricDMD$"],
                          fontsize=11, loc="center left")

                # Force x-axis to match actual data range
                ax.set_xlim(forecast_times_aligned[0], forecast_times_aligned[-1])
                ax.set_xticks(np.linspace(forecast_times_aligned[0], forecast_times_aligned[-1], 6))

            # Shared labels and layout
            fig.align_ylabels(axes)
            axes[-1].set_xlabel("$t$ (Time in seconds)", fontsize=14)

            plt.tight_layout(rect=[0, 0, 1, 0.97])
            plt.show()






def plot_fft_modal_comparison_interp_vs_true(
    pdmd_models,
    snapshot_test,
    loader_test,
    Re_test,
    dt_phys=0.01,
    t0_phys=15.0,
    n_modes_to_plot=6,
    time_window=(3.0, 20.0)
):
    """
    Compare FFT of interpolated vs true modal coefficients for an unseen Reynolds number
    across noise levels, using the same time matching logic as the modal comparison function.
    """

    # Step 0: Extract sampled_times_test from loader within time_window
    sampled_times_test = [float(t) for t in loader_test.write_times
                          if time_window[0] <= float(t) <= time_window[1]]
    sampled_times_test_float = np.array(sampled_times_test, dtype=float)

    for level, model in pdmd_models.items():
        # Step 1: Forecast time vector (interpolated coefficients)
        interpolated_modal_coeffs = model.interpolated_modal_coefficients[0]
        n_forecast = interpolated_modal_coeffs.shape[1]
        forecast_times = np.arange(n_forecast) * dt_phys + t0_phys

        # Step 2: Slice true snapshots to forecast window
        mask = (sampled_times_test_float >= t0_phys) & (sampled_times_test_float <= forecast_times[-1])
        snapshot_test_window = snapshot_test[:, mask]
        times_test_window = sampled_times_test_float[mask]

        # Step 3: Align lengths
        min_len = min(snapshot_test_window.shape[1], n_forecast) if snapshot_test_window.shape[1] > 0 else 0
        snapshot_test_aligned = snapshot_test_window[:, :min_len]
        forecast_times_aligned = forecast_times[:min_len]
        interp_modal_aligned = interpolated_modal_coeffs[:, :min_len]

        # Step 4: Project onto training POD basis (skip if no data)
        if min_len > 0:
            true_modal_coeffs_aligned = model._spatial_pod.reduce(snapshot_test_aligned)

            # Step 5: FFT comparison
            freqs = np.fft.rfftfreq(min_len, d=dt_phys)

            fig, axes = plt.subplots(n_modes_to_plot, 1,
                                     figsize=(12, 3.0 * n_modes_to_plot),
                                     sharex=True)
            fig.suptitle(
                f"Frequency Spectrum of Modal Coefficient Dynamics $a_m(t)$ — True vs Interpolated $ParametricDMD$\n"
                f"$Unseen^*$ Parameter (Re={Re_test}) \n (Noise {level}%)",
                fontsize=16, y=0.97
            )

            for mode_idx in range(n_modes_to_plot):
                ax = axes[mode_idx]

                # True FFT
                true_mode = true_modal_coeffs_aligned[mode_idx].real
                fft_true = np.abs(np.fft.rfft(true_mode))
                ax.plot(freqs, fft_true, label="True", linewidth=2, color="tab:blue")

                # Interpolated FFT
                interp_mode = interp_modal_aligned[mode_idx].real
                fft_interp = np.abs(np.fft.rfft(interp_mode))
                ax.plot(freqs, fft_interp, linestyle='--', label="Interpolated $ParametericDMD$", linewidth=2, color="tab:orange")

                ax.set_ylabel("Spectral Amplitude")
                ax.set_title(f"Mode $\\Phi_{{{mode_idx}}}$")
                ax.grid(True)
                ax.legend(loc="upper right", fontsize=10)

            axes[-1].set_xlabel("Frequency [Hz]")
            fig.align_ylabels(axes)
            plt.tight_layout(rect=[0, 0, 1, 0.96])
            plt.show()




def plot_interp_reconstruction_error(snapshot_test,
                                     times_test,
                                     pdmd_models,
                                     dt_phys,
                                     Re_test,
                                     mean_flow_test,
                                     t0_phys=15.0,
                                     time_window=(15.0, 20.0)):
    """
    Plot relative L2 reconstruction error for ParametricDMD at a test Reynolds number,
    across multiple noise levels, with a summary table of average % errors.
    """

    # True CFD snapshots and times
    t_vec_true = np.array(times_test, dtype=float)
    X_true = snapshot_test

    # Create two rows: one for plot, one for table
    fig, (ax, ax_table) = plt.subplots(
        2, 1, figsize=(10, 7), gridspec_kw={"height_ratios": [3, 1]}
    )

    summary_rows = []

    for level, pdmd in pdmd_models.items():
        # Forecast times shifted to start at t0_phys
        X_recon_raw = pdmd.reconstructed_data[0].real
        n_forecast = X_recon_raw.shape[1]
        t_vec_forecast = np.arange(n_forecast) * dt_phys + t0_phys

        # Slice true snapshots to forecast window
        mask = (t_vec_true >= time_window[0]) & (t_vec_true <= t_vec_forecast[-1])
        X_true_window = X_true[:, mask]
        t_true_window = t_vec_true[mask]

        # Align lengths
        min_len = min(X_true_window.shape[1], n_forecast)
        X_true_aligned = X_true_window[:, :min_len]
        X_recon_aligned = X_recon_raw[:, :min_len] + mean_flow_test[:, None]
        time_phys = t_vec_forecast[:min_len]

        # Compute errors
        abs_error_pdmd = np.linalg.norm(X_true_aligned - X_recon_aligned, axis=0)
        rel_error_pdmd = abs_error_pdmd / np.linalg.norm(X_true_aligned, axis=0)

        # Plot curve
        ax.plot(time_phys, rel_error_pdmd, lw=2, label=f"{level}% noise")

        # Collect average error for table
        avg_err_pct = 100 * np.mean(rel_error_pdmd)
        summary_rows.append([f"{level}%", f"{avg_err_pct:.2f}%"])

    # Plot formatting
    ax.set_xlabel(r"$t$ (Time in seconds)", fontsize=12)
    ax.set_ylabel(r"Relative $L^2$ Error", fontsize=12)
    ax.set_title(
        f"$ParametricDMD$ Interpolation Reconstruction Error\n$Unseen^*$ Parameter (Re={Re_test})",
        fontsize=14
    )
    ax.legend(title="Noise Level", fontsize=10, loc="upper right")
    ax.grid(True, linestyle="--", alpha=0.6)

    # Table axis
    ax_table.axis("off")
    table = ax_table.table(
        cellText=summary_rows,
        colLabels=["Noise Level", "Avg. % Error"],
        cellLoc="center",
        loc="center"
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11)

    plt.tight_layout()
    plt.show()







































