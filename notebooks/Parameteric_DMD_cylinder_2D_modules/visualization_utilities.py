import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from matplotlib.patches import Circle
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.ticker import MaxNLocator
from matplotlib.ticker import FixedLocator

import seaborn as sns

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
        ax.plot(times, mags)
        ax.set_ylabel("Velocity Magnitude", fontsize=12)
        ax.set_title(f"$Re$ = {Re}", fontsize=14, pad=6)
        ax.grid(True, alpha=0.6)
        ax.legend(fontsize=16)
    fig.align_ylabels(axes)
    axes[-1].set_xlabel("Time ($t$ in seconds)", fontsize=16)
    plt.suptitle("Snapshot Velocity Magnitudes For DMD Training Parameters",
                 fontsize=18, y = 0.96)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
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

    plt.title("Cumulative Energy Retained by POD Modes", fontsize=16, pad=10)
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

    ax.set_title("Residual Content (Cumulative Energy)", fontsize=16, pad=10)
    ax.set_xlabel("Number of Modes")
    ax.set_ylabel("Residual Energy")
    ax.grid(True)
    ax.legend()
    plt.tight_layout()
    plt.show()


def plot_dmd_modal_comparison(pdmd, Re_list, sampled_times_dict,
                              Re_value, n_modes_to_plot=5):
    """
    Plots a comparison of true vs ParametricDMD modal coefficients
    for a given Reynolds number (non-noise pipeline).

    Parameters:
    - pdmd: fitted ParametricDMD object
    - Re_list: list/array of Reynolds numbers used in training
    - sampled_times_dict: dict of physical time steps per Reynolds number
    - Re_value: Reynolds number to inspect (e.g., 200)
    - n_modes_to_plot: number of dominant modes to visualize (default: 5)
    """
    if Re_value not in Re_list:
        raise ValueError(f"Re = {Re_value} not found in Re_list")

    index = np.where(Re_list == Re_value)[0][0]
    times = np.array(sampled_times_dict[Re_value], dtype=float)

    # Modal coefficients from the clean pipeline
    modal_true = pdmd.training_modal_coefficients[index]
    modal_dmd = pdmd._dmd[index].reconstructed_data[:, :modal_true.shape[1]]

    fig, axes = plt.subplots(n_modes_to_plot, 1, figsize=(12, 3 * n_modes_to_plot), sharex=True)

    for mode in range(n_modes_to_plot):
        ax = axes[mode]
        ax.plot(times, modal_true[mode], label=f"True", color='tab:blue')
        ax.plot(times, modal_dmd[mode], label=f"ParametricDMD", linestyle='--', color='tab:orange')
        ax.set_ylabel(f"Amplitude", fontsize=14)
        ax.set_title(f"Mode $\Phi_{{{mode}}}$", fontsize=14)
        ax.grid(True)
        ax.legend(fontsize=13)
    fig.align_ylabels(axes)
    axes[-1].set_xlabel("Time ($t$ in seconds)", fontsize=15)
    plt.suptitle(f"Modal Coefficients Comparison — ParametericDMD vs True \n($Re$ = {Re_value})",fontsize=18, y=0.97)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.show()



def plot_dmd_fft_comparison(pdmd, Re_list, Re_target, n_plot=4, dt=0.01):
    """
    Plot FFTs of modal coefficients: True vs ParametricDMD reconstruction over training window.

    Parameters:
    - pdmd: ParametricDMD object containing training and reconstruction data
    - Re_list: list or array of Reynolds numbers used in training
    - Re_target: specific Reynolds number to compare
    - n_plot: number of modes to plot
    - dt: time step size used during training
    """

    # Locate index for target Re
    i = np.where(Re_list == Re_target)[0][0]

    # Extract modal coefficients
    modal_true = pdmd.training_modal_coefficients[i]  # shape: (n_modes, n_time)
    modal_dmd = pdmd._dmd[i].reconstructed_data[:, :pdmd._time_instants]  # shape: (n_modes, n_time)

    # Frequency axis
    n_time = modal_true.shape[1]
    freqs = np.fft.rfftfreq(n_time, d=dt)

    # Plot
    fig, axes = plt.subplots(n_plot, 1, figsize=(10, 2.5 * n_plot), sharex=True)
    fig.suptitle(f"Frequency Spectrum of Modal Coefficients — ParametericDMD vs True \n($Re$ = {Re_target})", fontsize=16,y=0.96)

    for mode in range(n_plot):
        modal_true_clean = np.asarray(modal_true[mode], dtype=np.float64)
        modal_dmd_clean = np.asarray(modal_dmd[mode], dtype=np.float64)
        fft_true = np.abs(np.fft.rfft(modal_true_clean))
        fft_dmd = np.abs(np.fft.rfft(modal_dmd_clean))

        ax = axes[mode]
        line1, = ax.plot(freqs, fft_true, color="tab:blue")
        line2, = ax.plot(freqs, fft_dmd, color="tab:orange", linestyle="--")
        ax.set_ylabel(f"Spectral Amplitude", fontsize=12)
        ax.grid(True)

        # Subplot title
        ax.set_title(f"Mode $\Phi_{{{mode}}}$", fontsize=12)

        # Stacked legend labels
        label1 = "True"
        label2 = "ParametricDMD"
        ax.legend([line1, line2], [label1, label2])
    fig.align_ylabels(axes)
    axes[-1].set_xlabel("Frequency (Hz)", fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()


def plot_flow_comparison_dmd_vs_true(
    Re_target, 
    Re_list,
    t_start,
    t_end,
    granularity,
    rom,
    pdmd,
    mean_flow,
    norm_scales,
    sampled_times_dict,
    snapshot_processed_dict,
    masked_coords_dict,
    num_points_dict,
    cmap='icefire'
):
    # Colormap setup
    if isinstance(cmap, str):
        cmap = sns.color_palette(cmap, as_cmap=True)

    # Time setup
    i = np.where(Re_list == Re_target)[0][0]
    time_vec = np.array(sampled_times_dict[Re_target], dtype=float)
    selected_times = np.arange(t_start, t_end + 1e-6, granularity)
    t_indices = [np.where(np.isclose(time_vec, t, atol=1e-6))[0][0] for t in selected_times]

    # Spatial setup
    U_basis = rom.modes
    coords = masked_coords_dict[Re_target]
    num_points = num_points_dict[Re_target]
    tri = mtri.Triangulation(coords[:, 0], coords[:, 1])

    # DMD coefficients
    modal_dmd = pdmd._dmd[i].reconstructed_data[:, :pdmd.training_modal_coefficients[i].shape[1]]
    mean = mean_flow
    norm_scale = norm_scales[Re_target]

    # Precompute fields
    fields_per_row = []
    for t_idx in t_indices:
        coeff_t = modal_dmd[:, t_idx]
        U_dmd = U_basis @ coeff_t * norm_scale + mean
        U_true = snapshot_processed_dict[Re_target][:, t_idx] * norm_scale + mean

        ux_dmd, uy_dmd = np.real(U_dmd[:num_points]), np.real(U_dmd[num_points:])
        ux_true, uy_true = np.real(U_true[:num_points]), np.real(U_true[num_points:])

        mag_dmd = np.sqrt(ux_dmd**2 + uy_dmd**2)
        mag_true = np.sqrt(ux_true**2 + uy_true**2)
        residual = mag_true - mag_dmd

        fields_per_row.append((mag_true, mag_dmd, residual, time_vec[t_idx]))

    # Global scaling
    mag_true_all = np.concatenate([f[0] for f in fields_per_row])
    mag_dmd_all = np.concatenate([f[1] for f in fields_per_row])
    global_min = min(mag_true_all.min(), mag_dmd_all.min())
    global_max = max(mag_true_all.max(), mag_dmd_all.max())
    fixed_ticks_main = np.linspace(global_min, global_max, 10)
    sm_shared = ScalarMappable(norm=Normalize(vmin=global_min, vmax=global_max), cmap=cmap)
    sm_shared.set_array([])

    residual_all = np.concatenate([np.abs(f[2]) for f in fields_per_row])
    resid_min, resid_max = residual_all.min(), residual_all.max()
    fixed_ticks_resid = np.linspace(resid_min, resid_max, 10)
    sm_resid = ScalarMappable(norm=Normalize(vmin=resid_min, vmax=resid_max), cmap=cmap)
    sm_resid.set_array([])

    # Plotting
    n_rows = len(fields_per_row)
    fig, axes = plt.subplots(n_rows, 3, figsize=(14, 3 * n_rows))
    fig.suptitle(f"Flow Comparison — ParametricDMD vs True \n($Re$ = {Re_target})", fontsize=18, y=0.94)

    if n_rows == 1:
        axes = np.expand_dims(axes, axis=0)

    for row, (mag_true, mag_dmd, residual, t_val) in enumerate(fields_per_row):
        # True
        ax_true = axes[row, 0]
        ax_true.tricontourf(tri, mag_true, levels=200, cmap=cmap, vmin=global_min, vmax=global_max)
        ax_true.add_patch(Circle((0.2, 0.2), 0.05, color='black', zorder=10))
        ax_true.set_aspect('equal'); ax_true.set_xticks([]); ax_true.set_yticks([])
        for spine in ax_true.spines.values(): spine.set_visible(False)
        fig.colorbar(sm_shared, ax=ax_true, shrink=0.75, pad=0.02, ticks=fixed_ticks_main)
        ax_true.set_ylabel(f"$t$ = {t_val:.2f} s", fontsize=15, rotation=90, labelpad=25)

        # DMD
        ax_dmd = axes[row, 1]
        ax_dmd.tricontourf(tri, mag_dmd, levels=200, cmap=cmap, vmin=global_min, vmax=global_max)
        ax_dmd.add_patch(Circle((0.2, 0.2), 0.05, color='black', zorder=10))
        ax_dmd.set_aspect('equal'); ax_dmd.set_xticks([]); ax_dmd.set_yticks([])
        for spine in ax_dmd.spines.values(): spine.set_visible(False)
        fig.colorbar(sm_shared, ax=ax_dmd, shrink=0.75, pad=0.02, ticks=fixed_ticks_main)

        # Residual
        ax_res = axes[row, 2]
        ax_res.tricontourf(tri, residual, levels=200, cmap=cmap, vmin=resid_min, vmax=resid_max)
        ax_res.add_patch(Circle((0.2, 0.2), 0.05, color='black', zorder=10))
        ax_res.set_aspect('equal'); ax_res.set_xticks([]); ax_res.set_yticks([])
        for spine in ax_res.spines.values(): spine.set_visible(False)
        fig.colorbar(sm_resid, ax=ax_res, shrink=0.75, pad=0.02, ticks=fixed_ticks_resid)

    # Column titles only once
    axes[0, 0].set_title("True Magnitude", fontsize=15, pad=12)
    axes[0, 1].set_title("ParametricDMD", fontsize=15, pad=12)
    axes[0, 2].set_title(r"Residual $(U_{\mathrm{True}} - U_{\mathrm{ParametricDMD}})$", fontsize=15, pad=12)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95], h_pad=1)
    plt.show()




def plot_dmd_modal_comparison_interp_vs_true(
    pdmd,
    snapshot_test,
    loader_test,
    mean_flow,
    Re_test,
    dt_phys=0.01,
    t0_phys=15.0,
    n_modes_to_plot=6,
    match_tolerance=0.05,
    time_window=(3.0, 20.0)
):
    """
    Plot DMD modal coefficient comparison between interpolated and true data at a given test Reynolds number.

    Parameters:
    - pdmd: Fitted ParametricDMD object
    - snapshot_test: Snapshot matrix for Re_test (shape: space_dim x n_time)
    - loader_test: FOAMDataloader object used to access metadata
    - mean_flow: Mean flow vector used during training (shape: space_dim,)
    - Re_test: Reynolds number of the test data
    - dt_phys: Time step used in forecast (default: 0.01)
    - t0_phys: Starting time of forecast (default: 15.0)
    - n_modes_to_plot: Number of modes to visualize (default: 6)
    - match_tolerance: Time matching tolerance in seconds (default: 0.05)
    - time_window: Tuple specifying the time range to extract from loader_test (default: (3.0, 20.0))
    """

    # Step 0: Extract sampled_times_test from loader
    sampled_times_test = [t for t in loader_test.write_times if time_window[0] <= float(t) <= time_window[1]]
    sampled_times_test_float = np.array(sampled_times_test, dtype=float)

    # Step 1: Reconstruct forecast time vector
    interpolated_modal_coeffs = pdmd.interpolated_modal_coefficients[0]
    n_forecast = interpolated_modal_coeffs.shape[1]
    forecast_times = np.arange(n_forecast) * dt_phys + t0_phys

    # Step 2: Match forecast times to test snapshot times
    matched_true_indices = []
    forecast_indices = []

    for i, t in enumerate(forecast_times):
        diffs = np.abs(sampled_times_test_float - t)
        min_diff = np.min(diffs)
        if min_diff < match_tolerance:
            matched_true_indices.append(np.argmin(diffs))
            forecast_indices.append(i)

    if not matched_true_indices:
        print("❌ No valid time matches found.")
        return

    # Step 3: Preprocess test snapshots
    snapshot_test_aligned = snapshot_test[:, [t for _, t in zip(forecast_indices, matched_true_indices)]].copy()
    snapshot_test_aligned -= mean_flow[:, np.newaxis]
    snapshot_test_aligned /= np.linalg.norm(snapshot_test_aligned)

    # Step 4: Project onto training POD basis
    true_modal_coeffs_aligned = pdmd._spatial_pod.reduce(snapshot_test_aligned)
    aligned_times = [forecast_times[f] for f in forecast_indices]
    Re_interp = pdmd.parameters[0, 0]

    # Step 5: Plot modal comparison
    fig, axes = plt.subplots(n_modes_to_plot, 1, figsize=(10, 2.8 * n_modes_to_plot), sharex=True)
    fig.suptitle(f"Modal Coefficient Comparison — ParametericDMD Interpolation vs True \n($Re$ = {Re_interp})", fontsize=16, y=0.96)

    for mode_idx in range(n_modes_to_plot):
        ax = axes[mode_idx]
        ax.plot(aligned_times, true_modal_coeffs_aligned[mode_idx].real,
                linestyle=':', label=f"True", linewidth=2, color="tab:orange")
        ax.plot(aligned_times, interpolated_modal_coeffs[mode_idx, forecast_indices].real,
                label=f"Interpolated ParameteriDMD", linewidth=2, color="tab:blue")
        
        ax.set_ylabel(f"Amplitude", fontsize=13)
        ax.set_title(f"Mode $\Phi_{{{mode_idx}}}$", fontsize =13)
        ax.grid(True)
        ax.legend(fontsize = 12)
    fig.align_ylabels(axes)
    axes[-1].set_xlabel("Time ($t$ in seconds)", fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()



def plot_fft_modal_comparison_interp_vs_true(
    pdmd,
    snapshot_test,
    loader_test,
    mean_flow,
    Re_test,
    dt_phys=0.01,
    t0_phys=15.0,
    n_modes_to_plot=6,
    match_tolerance=0.05,
    time_window=(3.0, 20.0)
):
    """
    Plot FFT comparison of interpolated vs true DMD modal coefficients for a test Reynolds number.
    """

    # Step 0: Extract sampled_times_test from loader
    sampled_times_test = [t for t in loader_test.write_times if time_window[0] <= float(t) <= time_window[1]]
    sampled_times_test_float = np.array(sampled_times_test, dtype=float)

    # Step 1: Reconstruct forecast time vector
    interpolated_modal_coeffs = pdmd.interpolated_modal_coefficients[0]
    n_forecast = interpolated_modal_coeffs.shape[1]
    forecast_times = np.arange(n_forecast) * dt_phys + t0_phys

    # Step 2: Match forecast times to test snapshot times
    matched_true_indices = []
    forecast_indices = []
    for i, t in enumerate(forecast_times):
        diffs = np.abs(sampled_times_test_float - t)
        min_diff = np.min(diffs)
        if min_diff < match_tolerance:
            matched_true_indices.append(np.argmin(diffs))
            forecast_indices.append(i)

    valid_pairs = [(f_idx, t_idx) for f_idx, t_idx in zip(forecast_indices, matched_true_indices)]
    if not valid_pairs:
        print("❌ No valid time matches found.")
        return

    # Step 3: Align and normalize snapshots
    snapshot_test_aligned = snapshot_test[:, [t for _, t in valid_pairs]].copy()
    snapshot_test_aligned -= mean_flow[:, np.newaxis]
    snapshot_test_aligned /= np.linalg.norm(snapshot_test_aligned)

    # Step 4: Project onto POD basis
    true_modal_coeffs_aligned = pdmd._spatial_pod.reduce(snapshot_test_aligned)

    # Step 5: FFT comparison
    freqs = np.fft.rfftfreq(len(valid_pairs), d=dt_phys)
    fig, axes = plt.subplots(n_modes_to_plot, 1, figsize=(10, 2.5 * n_modes_to_plot), sharex=True)
    fig.suptitle(
        f"Frequency Spectrum of Modal Coefficients — ParametricDMD Interpolation vs True\n($Re$ = {Re_test})",
        fontsize=16, y=0.96
    )

    for mode_idx in range(n_modes_to_plot):
        ax = axes[mode_idx]
        interp_mode = interpolated_modal_coeffs[mode_idx, [f for f, _ in valid_pairs]].real
        fft_interp = np.abs(np.fft.rfft(interp_mode))

        true_mode = true_modal_coeffs_aligned[mode_idx].real
        fft_true = np.abs(np.fft.rfft(true_mode))

        line1, = ax.plot(freqs, fft_true, color="tab:blue")
        line2, = ax.plot(freqs, fft_interp, color="tab:orange", linestyle="--")

        ax.set_ylabel("Spectral Amplitude", fontsize=12)
        ax.set_title(f"Mode $\\Phi_{{{mode_idx}}}$", fontsize=12)
        ax.grid(True)

        # Legend
        ax.legend([line1, line2], ["True", "Interpolated ParametricDMD"], fontsize=11, frameon=True)

    fig.align_ylabels(axes)
    axes[-1].set_xlabel("Frequency (Hz)", fontsize=14)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()





def plot_flow_comparison_interpolated_dmd_vs_true(
    pdmd,
    snapshot_test,
    sampled_times_test,
    loader_test,
    mask_test,
    num_points_test,
    norm_scale,
    mean_flow,
    Re_test,
    t_start,
    t_end,
    granularity,
    dt_phys=0.01,
    cmap="jet"
):
    # Reconstruct forecast time vector
    forecast_times_physical = 15.0 + (np.array(pdmd.dmd_timesteps) - 500) * dt_phys

    # Filter forecast times
    step = max(1, int(granularity / dt_phys))
    filtered_times_forecast = [t for t in forecast_times_physical if t_start <= t <= t_end][::step]

    if len(filtered_times_forecast) == 0:
        print("⚠️ No forecast times found — expanding window by 0.5s.")
        t_start -= 0.25
        t_end += 0.25
        filtered_times_forecast = [t for t in forecast_times_physical if t_start <= t <= t_end][::step]

    sampled_times_test_float = np.array(sampled_times_test, dtype=float)

    # Match indices
    matched_true_indices = []
    for t in filtered_times_forecast:
        diffs = np.abs(sampled_times_test_float - t)
        min_diff = np.min(diffs)
        matched_true_indices.append(np.argmin(diffs) if min_diff < 1e-2 else None)

    forecast_indices = []
    for t in filtered_times_forecast:
        idx = np.where(np.isclose(forecast_times_physical, t, atol=1e-6))[0]
        forecast_indices.append(idx[0] if len(idx) > 0 else None)

    # Spatial setup
    coords_test = loader_test.vertices[mask_test.numpy(), :]
    triang_test = mtri.Triangulation(coords_test[:, 0], coords_test[:, 1])

    forecasted_snapshots = pdmd.reconstructed_data[0]

    # Plotting
    num_rows = len(filtered_times_forecast)
    fig, axes = plt.subplots(num_rows, 3, figsize=(14, 3 * num_rows))
    fig.suptitle(f"Flow Comparison — ParametricDMD Interpolation vs True \n($Re$ = {Re_test})",
                 fontsize=18, y=0.94)

    if num_rows == 1:
        axes = np.expand_dims(axes, axis=0)

    # Compute global min/max for consistent scaling
    mag_true_all = np.sqrt(np.real(snapshot_test[:num_points_test])**2 +
                        np.real(snapshot_test[num_points_test:])**2)
    mag_forecast_all = np.sqrt(np.real(forecasted_snapshots[:num_points_test])**2 +
                            np.real(forecasted_snapshots[num_points_test:])**2)
    global_min = min(mag_true_all.min(), mag_forecast_all.min())
    global_max = max(mag_true_all.max(), mag_forecast_all.max())


    fixed_ticks_main = np.linspace(global_min, global_max, 10)


    sm_shared = ScalarMappable(norm=Normalize(vmin=global_min, vmax=global_max), cmap=cmap)
    sm_shared.set_array([])

    residual_all = []
    for row in range(num_rows):
        true_idx = matched_true_indices[row]
        forecast_idx = forecast_indices[row]
        if true_idx is None or forecast_idx is None:
            continue

        u_x_true = snapshot_test[:num_points_test, true_idx]
        u_y_true = snapshot_test[num_points_test:, true_idx]
        mag_true = np.sqrt(u_x_true**2 + u_y_true**2).real

        U_raw = forecasted_snapshots[:, forecast_idx]
        U_forecast = U_raw * norm_scale + mean_flow
        u_x_forecast = U_forecast[:num_points_test]
        u_y_forecast = U_forecast[num_points_test:]
        mag_forecast = np.sqrt(u_x_forecast**2 + u_y_forecast**2).real

        error_field = (mag_true - mag_forecast).real
        residual_all.append(np.abs(error_field))

    resid_min = min(r.min() for r in residual_all)
    resid_max = max(r.max() for r in residual_all)
    fixed_ticks_resid = np.linspace(resid_min, resid_max, 10)
    sm_resid = ScalarMappable(norm=Normalize(vmin=resid_min, vmax=resid_max), cmap=cmap)
    sm_resid.set_array([])


    for row in range(num_rows):
        t = filtered_times_forecast[row]
        true_idx = matched_true_indices[row]
        forecast_idx = forecast_indices[row]
        if true_idx is None or forecast_idx is None:
            continue

        # True field
        u_x_true = snapshot_test[:num_points_test, true_idx]
        u_y_true = snapshot_test[num_points_test:, true_idx]
        mag_true = np.sqrt(u_x_true**2 + u_y_true**2).real

        # Forecast field
        U_raw = forecasted_snapshots[:, forecast_idx]
        U_forecast = U_raw * norm_scale + mean_flow
        u_x_forecast = U_forecast[:num_points_test]
        u_y_forecast = U_forecast[num_points_test:]
        mag_forecast = np.sqrt(u_x_forecast**2 + u_y_forecast**2).real

        # Error field
        error_field = (mag_true - mag_forecast).real

        # True plot
        ax_true = axes[row, 0]
        contour_true = ax_true.tricontourf(triang_test, mag_true, levels=200, cmap=cmap,
                                           vmin=global_min, vmax=global_max)
        ax_true.add_patch(Circle((0.2, 0.2), 0.05, color='black', zorder=10))
        ax_true.set_aspect('equal')
        ax_true.set_xticks([]); ax_true.set_yticks([])
        for spine in ax_true.spines.values():
            spine.set_visible(False)
        fig.colorbar(sm_shared, ax=ax_true, shrink=0.75, pad=0.02, ticks=fixed_ticks_main)

        # Forecast plot
        ax_forecast = axes[row, 1]
        contour_forecast = ax_forecast.tricontourf(triang_test, mag_forecast, levels=200, cmap=cmap,
                                                   vmin=global_min, vmax=global_max)
        ax_forecast.add_patch(Circle((0.2, 0.2), 0.05, color='black', zorder=10))
        ax_forecast.set_aspect('equal')
        ax_forecast.set_xticks([]); ax_forecast.set_yticks([])
        for spine in ax_forecast.spines.values():
            spine.set_visible(False)
        fig.colorbar(sm_shared, ax=ax_forecast, shrink=0.75, pad=0.02, ticks=fixed_ticks_main)

        # Residual plot
    
        ax_error = axes[row, 2]
        contour_error = ax_error.tricontourf(triang_test, np.abs(error_field), levels=200, cmap=cmap,
                                            vmin=resid_min, vmax=resid_max)
        ax_error.add_patch(Circle((0.2, 0.2), 0.05, color='black', zorder=10))
        ax_error.set_aspect('equal')
        ax_error.set_xticks([]); ax_error.set_yticks([])
        for spine in ax_error.spines.values():
            spine.set_visible(False)

        fig.colorbar(sm_resid, ax=ax_error, shrink=0.75, pad=0.02, ticks=fixed_ticks_resid)

        
        # Add vertical time label for the row
        ax_true.set_ylabel(f"$t$ = {t:.2f} s", fontsize=15, rotation=90, labelpad=25)

    # Add column titles only once (top row)
    axes[0, 0].set_title("True Magnitude", fontsize=15, pad=12)
    axes[0, 1].set_title("Interpolated ParametricDMD", fontsize=15, pad=12)
    axes[0, 2].set_title(r"Residual $(U_{\mathrm{True}} - U_{\mathrm{Interpolated \ ParametricDMD}})$",
                         fontsize=15, pad=12)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95], h_pad=1)
    plt.show()




