import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from matplotlib.patches import Circle
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.ticker import MaxNLocator, FixedLocator

import seaborn as sns
from sklearn.utils.extmath import randomized_svd

import matplotlib as mpl
mpl.rcParams['mathtext.fontset'] = 'cm'   # Computer Modern






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
    plt.suptitle("Snapshot Velocity Magnitudes for DMD Training Parameters",
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
                              Re_value, U_ref_dict, L_ref, n_modes_to_plot=5):
    """
    Plots a comparison of true vs ParametricDMD modal coefficients
    for a given Reynolds number, using non-dimensional time.
    """
    if Re_value not in Re_list:
        raise ValueError(f"$Re$ = {Re_value} not found in Re_list")

    # Index of target Re in training list
    index = np.where(np.array(Re_list) == Re_value)[0][0]
    times = np.array(sampled_times_dict[Re_value], dtype=float)

    # Non-dimensionalize time
    U_ref = U_ref_dict[Re_value]
    time_nd = (times - times[0]) * U_ref / L_ref

    # True vs reconstructed modal coefficients
    modal_true = pdmd.training_modal_coefficients[index]
    modal_dmd = pdmd._dmd[index].reconstructed_data[:, :modal_true.shape[1]]

    # Create subplots
    fig, axes = plt.subplots(n_modes_to_plot, 1,
                             figsize=(12, 2.8 * n_modes_to_plot),
                             sharex=True)

    for mode in range(n_modes_to_plot):
        ax = axes[mode]
        ax.plot(time_nd, modal_true[mode],
                color="tab:blue", lw=1.8, label="True")
        ax.plot(time_nd, modal_dmd[mode],
                color="tab:orange", lw=1.8, linestyle="--", label="ParametricDMD")

        ax.set_ylabel("Amplitude", fontsize=12)
        ax.set_title(f"Mode $\\Phi_{{{mode}}}$", fontsize=13, pad=6)
        ax.grid(alpha=0.6)
        ax.legend(fontsize=11, loc="upper right")
        ax.set_xlim(time_nd[0], time_nd[-1])
        ax.set_xticks(np.linspace(time_nd[0], time_nd[-1], 6))

    axes[-1].set_xlabel(r"$t^* = t U_{ref} / L_{ref}$", fontsize=13)


    fig.align_ylabels(axes)
    plt.suptitle(
        f"Modal Coefficient Dynamics — "
        f"True vs ParametricDMD \n Training Parameter $(Re={Re_value})$",
        fontsize=16, y=0.97
    )
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()



def plot_dmd_fft_comparison(pdmd, Re_list, Re_target, L_ref, nu,
                            n_plot=4, dt=0.01, ref_St=0.2):
    """
    Plot FFTs of modal coefficients: True vs ParametricDMD reconstruction,
    using nondimensional frequency (Strouhal number).

    Parameters
    ----------
    pdmd : ParametricDMD
        Object containing training and reconstruction data.
    Re_list : list or array
        Reynolds numbers used in training.
    Re_target : int
        Specific Reynolds number to compare.
    L_ref : float
        Reference length (e.g., cylinder diameter).
    nu : float
        Kinematic viscosity.
    n_plot : int
        Number of modes to plot.
    dt : float
        Time step size used during training.
    ref_St : float, optional
        Reference Strouhal number for vortex shedding (default = 0.2).
    """

    # Locate index for target Re
    i = np.where(Re_list == Re_target)[0][0]

    # Extract modal coefficients
    modal_true = pdmd.training_modal_coefficients[i]  # shape: (n_modes, n_time)
    modal_dmd = pdmd._dmd[i].reconstructed_data[:, :pdmd._time_instants]

    # Frequency axis (dimensional)
    n_time = modal_true.shape[1]
    freqs = np.fft.rfftfreq(n_time, d=dt)

    # Compute reference velocity for this Re
    U_ref = Re_target * nu / L_ref

    # Convert to Strouhal number
    St = freqs * L_ref / U_ref

    # Plot
    fig, axes = plt.subplots(n_plot, 1, figsize=(10, 2.5 * n_plot), sharex=True)
    fig.suptitle(
        f"Frequency Spectrum of Modal Coefficient Dynamics "
        f"— True vs ParametricDMD \n Training Parameter $(Re = {Re_target})$",
        fontsize=16, y=0.96
    )

    for mode in range(n_plot):
        modal_true_clean = np.asarray(modal_true[mode], dtype=np.float64)
        modal_dmd_clean = np.asarray(modal_dmd[mode], dtype=np.float64)
        fft_true = np.abs(np.fft.rfft(modal_true_clean))
        fft_dmd = np.abs(np.fft.rfft(modal_dmd_clean))

        ax = axes[mode]
        line1, = ax.plot(St, fft_true, color="tab:blue")
        line2, = ax.plot(St, fft_dmd, color="tab:orange", linestyle="--")
        ax.set_ylabel("Spectral Amplitude", fontsize=12)
        ax.grid(True)
        ax.set_title(f"Mode $\\Phi_{{{mode}}}$", fontsize=12)
        ax.legend([line1, line2], ["True", "ParametricDMD"])

    fig.align_ylabels(axes)
    axes[-1].set_xlabel(r"$St = f L_{ref} / U_{ref}$", fontsize=13)

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
    sampled_times_dict,
    snapshot_processed_dict,
    masked_coords_dict,
    num_points_dict,
    L_ref=0.1,
    cylinderX=0.2,
    cylinderY=0.2,
    radius=0.05,
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
    coords = masked_coords_dict[Re_target] / L_ref   
    num_points = num_points_dict[Re_target]
    tri = mtri.Triangulation(coords[:, 0], coords[:, 1])

    # DMD coefficients
    modal_dmd = pdmd._dmd[i].reconstructed_data[:, :pdmd.training_modal_coefficients[i].shape[1]]

    # Precompute fields
    fields_per_row = []
    for t_idx in t_indices:
        coeff_t = modal_dmd[:, t_idx]
        U_dmd = U_basis @ coeff_t + mean_flow   
        U_true = snapshot_processed_dict[Re_target][:, t_idx] + mean_flow  

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
    fig.suptitle(f"Flow Comparison — True vs ParametricDMD\n Training Parameter $(Re = {Re_target})$", fontsize=18, y=0.96)

    if n_rows == 1:
        axes = np.expand_dims(axes, axis=0)

    for row, (mag_true, mag_dmd, residual, t_val) in enumerate(fields_per_row):
        # True
        ax_true = axes[row, 0]
        ax_true.tricontourf(tri, mag_true, levels=200, cmap=cmap, vmin=global_min, vmax=global_max)
        ax_true.add_patch(Circle((cylinderX/L_ref, cylinderY/L_ref), radius/L_ref, color="black", zorder=10))
        ax_true.set_aspect("equal")
        fig.colorbar(sm_shared, ax=ax_true, shrink=1, pad=0.02, ticks=fixed_ticks_main)
        ax_true.text(-0.30, 0.5, f"$t$ = {t_val:.2f} s",
                     transform=ax_true.transAxes,
                     fontsize=16, rotation=90,
                     va="center", ha="center",
                     bbox=dict(facecolor="white", alpha=0.7, edgecolor="none", pad=2))

        # DMD
        ax_dmd = axes[row, 1]
        ax_dmd.tricontourf(tri, mag_dmd, levels=200, cmap=cmap, vmin=global_min, vmax=global_max)
        ax_dmd.add_patch(Circle((cylinderX/L_ref, cylinderY/L_ref), radius/L_ref, color="black", zorder=10))
        ax_dmd.set_aspect("equal")
        fig.colorbar(sm_shared, ax=ax_dmd, shrink=1, pad=0.02, ticks=fixed_ticks_main)

        # Residual
        ax_res = axes[row, 2]
        ax_res.tricontourf(tri, residual, levels=200, cmap=cmap, vmin=resid_min, vmax=resid_max)
        ax_res.add_patch(Circle((cylinderX/L_ref, cylinderY/L_ref), radius/L_ref, color="black", zorder=10))
        ax_res.set_aspect("equal")
        fig.colorbar(sm_resid, ax=ax_res, shrink=1, pad=0.02, ticks=fixed_ticks_resid)

        # Non-dimensional axis labels
        for ax in [ax_true, ax_dmd, ax_res]:
            ax.set_xlabel(r"$x/L_{ref}$", fontsize=12)
            ax.set_ylabel(r"$y/L_{ref}$", fontsize=12)
            ax.tick_params(labelbottom=True, labelleft=True)

    # Column titles only once
    axes[0, 0].set_title("True Magnitude", fontsize=15, pad=12,
                         bbox=dict(facecolor="white", alpha=0.7, edgecolor="none", pad=2))
    axes[0, 1].set_title("ParametricDMD", fontsize=15, pad=12,
                         bbox=dict(facecolor="white", alpha=0.7, edgecolor="none", pad=2))
    axes[0, 2].set_title(r"Residual $(U_{\text{True}} - U_{\text{ParametricDMD}})$", fontsize=15, pad=12,
                         bbox=dict(facecolor="white", alpha=0.7, edgecolor="none", pad=2))

    plt.tight_layout(rect=[0, 0.03, 1, 0.95], h_pad=2.0)
    plt.show()



def plot_dmd_reconstruction_error(
    Re_target,
    Re_list,
    rom,
    pdmd,
    mean_flow_train,
    sampled_times_dict,
    snapshot_processed_dict,
    L_ref,
    nu,
    color='tab:orange'
):
    """
    Plot relative L2 reconstruction error for a training Reynolds number using DMD.
    Uses training mean flow (mean-only preprocessing, no normalization).
    Time axis is nondimensionalized: t* = t U_ref / L_ref.
    """

    # Index of target Re in training list
    i = np.where(Re_list == Re_target)[0][0]
    time_vec = np.array(sampled_times_dict[Re_target], dtype=float)

    # Reference velocity for nondimensionalization
    U_ref = Re_target * nu / L_ref
    time_star = (time_vec - time_vec[0]) * U_ref / L_ref   # nondimensional time

    # True and reconstructed snapshots (aligned in training window)
    X_true = snapshot_processed_dict[Re_target] + mean_flow_train[:, None]
    modal_dmd = pdmd._dmd[i].reconstructed_data[:, :pdmd.training_modal_coefficients[i].shape[1]]
    X_recon = rom.modes @ modal_dmd + mean_flow_train[:, None]

    # Compute relative error over time
    abs_error = np.linalg.norm(X_true - X_recon, axis=0)
    rel_error = abs_error / np.linalg.norm(X_true, axis=0)

    # Plot with nondimensional time
    fig, ax = plt.subplots(figsize=(8,4))
    ax.plot(time_star, rel_error, lw=2, color=color)
    ax.set_xlabel(r"$t^* = t U_{ref} / L_{ref}$", fontsize=13)
    ax.set_ylabel(r"Relative $L^2$ Error", fontsize=13)
    ax.set_title(f"ParametricDMD Reconstruction Error\n Training Parameter $(Re = {Re_target})$", fontsize=15)
    ax.grid(True, alpha=0.6)
    ax.legend()
    ax.set_xlim(time_star.min(), time_star.max())
    ax.margins(x=0)
    ax.set_xticks(np.linspace(time_star.min(), time_star.max(), 6))
    plt.tight_layout()
    plt.show()




def plot_dmd_forecast_error(
    Re_target,
    Re_list,
    rom,
    pdmd,
    mean_flow_train,
    sampled_times_dict,
    snapshot_processed_dict,
    L_ref,
    nu,
    color='tab:blue'
):
    """
    Plot relative L2 forecast error for a training Reynolds number using ParametricDMD.
    Uses training mean flow (mean-only preprocessing, no normalization).
    Time axis is nondimensionalized: t* = t U_ref / L_ref.
    """

    # Index of target Re in training list
    i = np.where(Re_list == Re_target)[0][0]
    time_vec = np.array(sampled_times_dict[Re_target], dtype=float)

    # Reference velocity for nondimensionalization
    U_ref = Re_target * nu / L_ref
    time_star = (time_vec - time_vec[0]) * U_ref / L_ref   # nondimensional time

    # True snapshots
    X_true = snapshot_processed_dict[Re_target] + mean_flow_train[:, None]

    # Forecasted modal coefficients (from pdmd)
    modal_forecast = pdmd.forecasted_modal_coefficients[i]
    X_forecast = rom.modes @ modal_forecast + mean_flow_train[:, None]

    # Compute relative error over time
    abs_error = np.linalg.norm(X_true - X_forecast, axis=0)
    rel_error = abs_error / np.linalg.norm(X_true, axis=0)

    # Plot with nondimensional time
    fig, ax = plt.subplots(figsize=(8,4))
    ax.plot(time_star, rel_error, lw=2, color=color)
    ax.set_xlabel(r"$t^* = t U_{ref} / L_{ref}$", fontsize=13)
    ax.set_ylabel(r"Relative $L^2$ Error", fontsize=13)
    ax.set_title(f"ParametricDMD Forecast Reconstruction Error\n Training Parameter $(Re = {Re_target})$", fontsize=15)
    ax.grid(True, alpha=0.6)
    ax.legend()
    ax.set_xlim(time_star.min(), time_star.max())
    ax.margins(x=0)
    ax.set_xticks(np.linspace(time_star.min(), time_star.max(), 6))
    plt.tight_layout()
    plt.show()


def plot_dmd_modal_comparison_interp_vs_true(
    pdmd,
    snapshot_test,
    loader_test,
    Re_test,
    L_ref,
    nu,
    dt_phys=0.01,
    t0_phys=15.0,
    n_modes_to_plot=6,
    time_window=(3.0, 20.0)
):
    """
    Compare DMD modal coefficients between interpolated and true data
    at a given test Reynolds number, using nondimensional time.
    """

    # Extract sampled times from loader within the chosen window
    sampled_times_test = [float(t) for t in loader_test.write_times
                          if time_window[0] <= float(t) <= time_window[1]]
    sampled_times_test_float = np.array(sampled_times_test, dtype=float)

    # Forecast time vector from interpolated coefficients
    interpolated_modal_coeffs = pdmd.interpolated_modal_coefficients[0]
    n_forecast = interpolated_modal_coeffs.shape[1]
    forecast_times = np.arange(n_forecast) * dt_phys + t0_phys

    # Nondimensionalize time using U_ref = Re * nu / L_ref
    U_ref = Re_test * nu / L_ref
    forecast_times_star = (forecast_times - forecast_times[0]) * U_ref / L_ref
    sampled_times_test_star = (sampled_times_test_float - forecast_times[0]) * U_ref / L_ref

    
    # Slice true snapshots to match forecast window
    mask = (sampled_times_test_float >= t0_phys) & (sampled_times_test_float <= forecast_times[-1])
    snapshot_test_window = snapshot_test[:, mask]
    times_test_window_star = sampled_times_test_star[mask]

    # Align lengths of true and forecast data
    min_len = min(snapshot_test_window.shape[1], n_forecast)
    snapshot_test_aligned = snapshot_test_window[:, :min_len]
    forecast_times_star_aligned = forecast_times_star[:min_len]
    interp_modal_aligned = interpolated_modal_coeffs[:, :min_len]

    # Project true snapshots onto training POD basis
    true_modal_coeffs_aligned = pdmd._spatial_pod.reduce(snapshot_test_aligned)
    Re_interp = pdmd.parameters[0, 0]

    # Plot modal coefficient comparison
    fig, axes = plt.subplots(n_modes_to_plot, 1,
                             figsize=(12, 2.8 * n_modes_to_plot),
                             sharex=True)
    fig.suptitle(f"Modal Coefficient Dynamics — True vs Interpolated ParametricDMD\n"
                 f"$Unseen^*$ Parameter $(Re = {Re_interp})$",
                 fontsize=16, y=0.97)

    for mode_idx in range(n_modes_to_plot):
        ax = axes[mode_idx]
        line1, = ax.plot(forecast_times_star_aligned,
                         true_modal_coeffs_aligned[mode_idx].real,
                         color="tab:blue", lw=1.5, label="True")
        line2, = ax.plot(forecast_times_star_aligned,
                         interp_modal_aligned[mode_idx].real,
                         color="tab:orange", linestyle="--", lw=1.5, label="Interpolated ParametricDMD")

        ax.set_ylabel("Amplitude", fontsize=12)
        ax.set_title(f"Mode $\\Phi_{{{mode_idx}}}$", fontsize=12, pad=6)
        ax.grid(True, alpha=0.6)
        ax.legend([line1, line2], ["True", "Interpolated ParametricDMD"], fontsize=11, loc="center left")
        ax.set_xlim(forecast_times_star_aligned[0], forecast_times_star_aligned[-1])
        ax.set_xticks(np.linspace(forecast_times_star_aligned[0], forecast_times_star_aligned[-1], 6))

    fig.align_ylabels(axes)
    axes[-1].set_xlabel(r"$t^* = t U_{ref} / L_{ref}$", fontsize=14)

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.show()





def plot_fft_modal_comparison_interp_vs_true(
    pdmd,
    snapshot_test,
    loader_test,
    Re_test,
    L_ref,
    nu,
    dt_phys=0.01,
    t0_phys=15.0,
    n_modes_to_plot=6,
    match_tolerance=0.05,
    time_window=(3.0, 20.0)
):
    """
    Plot FFT comparison of interpolated vs true DMD modal coefficients
    at a test Reynolds number, using nondimensional frequency (Strouhal number).
    """

    # Extract sampled times from loader
    sampled_times_test = [t for t in loader_test.write_times if time_window[0] <= float(t) <= time_window[1]]
    sampled_times_test_float = np.array(sampled_times_test, dtype=float)

    # Forecast time vector
    interpolated_modal_coeffs = pdmd.interpolated_modal_coefficients[0]
    n_forecast = interpolated_modal_coeffs.shape[1]
    forecast_times = np.arange(n_forecast) * dt_phys + t0_phys

    # Match forecast times to test snapshot times
    matched_true_indices, forecast_indices = [], []
    for i, t in enumerate(forecast_times):
        diffs = np.abs(sampled_times_test_float - t)
        if np.min(diffs) < match_tolerance:
            matched_true_indices.append(np.argmin(diffs))
            forecast_indices.append(i)

    valid_pairs = [(f_idx, t_idx) for f_idx, t_idx in zip(forecast_indices, matched_true_indices)]
    if not valid_pairs:
        print("No valid time matches found.")
        return

    # Align snapshots
    snapshot_test_aligned = snapshot_test[:, [t for _, t in valid_pairs]].copy()

    # Project onto POD basis
    true_modal_coeffs_aligned = pdmd._spatial_pod.reduce(snapshot_test_aligned)

    # FFT frequencies in Hz
    freqs = np.fft.rfftfreq(len(valid_pairs), d=dt_phys)

    # Convert to Strouhal number
    U_ref = Re_test * nu / L_ref
    freqs_star = freqs * L_ref / U_ref

    # Plot FFT comparison
    fig, axes = plt.subplots(n_modes_to_plot, 1, figsize=(10, 2.5 * n_modes_to_plot), sharex=True)
    fig.suptitle(
        f"Frequency Spectrum of Modal Coefficient Dynamics — True vs Interpolated ParametricDMD\n"
        f"$Unseen^*$ Parameter $(Re = {Re_test})$",
        fontsize=16, y=0.96
    )

    for mode_idx in range(n_modes_to_plot):
        ax = axes[mode_idx]
        interp_mode = interpolated_modal_coeffs[mode_idx, [f for f, _ in valid_pairs]].real
        fft_interp = np.abs(np.fft.rfft(interp_mode))

        true_mode = true_modal_coeffs_aligned[mode_idx].real
        fft_true = np.abs(np.fft.rfft(true_mode))

        line1, = ax.plot(freqs_star, fft_true, color="tab:blue")
        line2, = ax.plot(freqs_star, fft_interp, color="tab:orange", linestyle="--")

        ax.set_ylabel("Spectral Amplitude", fontsize=12)
        ax.set_title(f"Mode $\\Phi_{{{mode_idx}}}$", fontsize=12)
        ax.grid(True)
        ax.legend([line1, line2], ["True", "Interpolated ParametricDMD"], fontsize=11, frameon=True)

    fig.align_ylabels(axes)
    axes[-1].set_xlabel(r"$St = f L_{ref} / U_{ref}$", fontsize=14)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()





def plot_flow_comparison_interpolated_dmd_vs_true(
    pdmd,
    snapshot_test,
    sampled_times_test,
    loader_test,
    mask_test,
    num_points_test,
    mean_flow_test,   
    Re_test,
    t_start,
    t_end,
    granularity,
    dt_phys=0.01,
    L_ref=0.1,
    cylinderX=0.2,
    cylinderY=0.2,
    radius=0.05,
    cmap="jet"
):
    """
    Plot comparison of true vs interpolated ParametricDMD reconstructions for a given test Reynolds number.
    """

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

    # Spatial setup (non-dimensionalized, keep / L_ref)
    coords_test = loader_test.vertices[mask_test.numpy(), :] / L_ref
    triang_test = mtri.Triangulation(coords_test[:, 0], coords_test[:, 1])

    # Interpolated reconstruction for the test Re
    pdmd.parameters = np.array([[Re_test]])
    interp_snapshots = pdmd.reconstructed_data[0]

    # Plotting
    num_rows = len(filtered_times_forecast)
    fig, axes = plt.subplots(num_rows, 3, figsize=(14, 3 * num_rows))
    fig.suptitle(f"Flow Comparison — True vs Interpolated ParametricDMD\n $Unseen^*$ Parameter $(Re = {Re_test})$",
                 fontsize=18, y=0.96)

    if num_rows == 1:
        axes = np.expand_dims(axes, axis=0)

    # Compute global min/max for consistent scaling
    mag_true_all = np.sqrt(np.real(snapshot_test[:num_points_test])**2 +
                           np.real(snapshot_test[num_points_test:])**2)
    mag_forecast_all = np.sqrt(np.real(interp_snapshots[:num_points_test])**2 +
                               np.real(interp_snapshots[num_points_test:])**2)
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

        U_raw = interp_snapshots[:, forecast_idx]
        U_forecast = U_raw + mean_flow_test
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

        # Forecast field (interpolated reconstruction + test mean)
        U_raw = interp_snapshots[:, forecast_idx]
        U_forecast = U_raw + mean_flow_test
        u_x_forecast = U_forecast[:num_points_test]
        u_y_forecast = U_forecast[num_points_test:]
        mag_forecast = np.sqrt(u_x_forecast**2 + u_y_forecast**2).real

        # Error field
        error_field = (mag_true - mag_forecast).real

        # True plot
        ax_true = axes[row, 0]
        ax_true.tricontourf(triang_test, mag_true, levels=200, cmap=cmap,
                            vmin=global_min, vmax=global_max)
        ax_true.add_patch(Circle((cylinderX/L_ref, cylinderY/L_ref), radius/L_ref, color="black", zorder=10))
        ax_true.set_aspect("equal")
        fig.colorbar(sm_shared, ax=ax_true, shrink=1, pad=0.02, ticks=fixed_ticks_main)
        ax_true.text(-0.30, 0.5, f"$t$ = {t:.2f} s",
                     transform=ax_true.transAxes,
                     fontsize=16, rotation=90,
                     va="center", ha="center",
                     bbox=dict(facecolor="white", alpha=0.7, edgecolor="none", pad=2))

        # Forecast plot
        ax_forecast = axes[row, 1]
        ax_forecast.tricontourf(triang_test, mag_forecast, levels=200, cmap=cmap,
                                vmin=global_min, vmax=global_max)
        ax_forecast.add_patch(Circle((cylinderX/L_ref, cylinderY/L_ref), radius/L_ref, color="black", zorder=10))
        ax_forecast.set_aspect("equal")
        fig.colorbar(sm_shared, ax=ax_forecast, shrink=1, pad=0.02, ticks=fixed_ticks_main)

        # Residual plot
        ax_error = axes[row, 2]
        ax_error.tricontourf(triang_test, np.abs(error_field), levels=200, cmap=cmap,
                             vmin=resid_min, vmax=resid_max)
        ax_error.add_patch(Circle((cylinderX/L_ref, cylinderY/L_ref), radius/L_ref, color="black", zorder=10))
        ax_error.set_aspect("equal")
        fig.colorbar(sm_resid, ax=ax_error, shrink=1, pad=0.02, ticks=fixed_ticks_resid)

        # Non-dimensional axis labels
        for ax in [ax_true, ax_forecast, ax_error]:
            ax.set_xlabel(r"$x/L_{ref}$", fontsize=12)
            ax.set_ylabel(r"$y/L_{ref}$", fontsize=12)
            ax.tick_params(labelbottom=True, labelleft=True)

    # Column titles only once
    axes[0, 0].set_title("True Magnitude", fontsize=15, pad=12,
                         bbox=dict(facecolor="white", alpha=0.7, edgecolor="none", pad=2))
    axes[0, 1].set_title("Interpolated ParametricDMD", fontsize=15, pad=12,
                         bbox=dict(facecolor="white", alpha=0.7, edgecolor="none", pad=2))
    axes[0, 2].set_title(r"Residual $(U_{True} - U_{Interpolated \; ParametricDMD})$",
                         fontsize=15, pad=12,
                         bbox=dict(facecolor="white", alpha=0.7, edgecolor="none", pad=2))

    # Final layout adjustments
    plt.tight_layout(rect=[0, 0.03, 1, 0.95], h_pad=2.0)
    plt.show()


def plot_interp_reconstruction_error(snapshot_test,
                                     times_test,
                                     pdmd,
                                     dt_phys,
                                     Re_test,
                                     mean_flow_test,
                                     L_ref,
                                     nu,
                                     t0_phys=15.0,
                                     time_window=(15.0, 20.0)):
    """
    Plot relative L2 reconstruction error for ParametricDMD at a test Reynolds number.
    Time axis is nondimensionalized: t* = (t - t0) U_ref / L_ref.
    """

    # True CFD snapshots and times
    t_vec_true = np.array(times_test, dtype=float)
    X_true = snapshot_test

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

    # Nondimensionalize time starting at zero
    U_ref = Re_test * nu / L_ref
    time_star = (time_phys - time_phys[0]) * U_ref / L_ref

    # Compute errors
    abs_error_pdmd = np.linalg.norm(X_true_aligned - X_recon_aligned, axis=0)
    rel_error_pdmd = abs_error_pdmd / np.linalg.norm(X_true_aligned, axis=0)
    perc_error_pdmd = rel_error_pdmd * 100.0

    # Total percentage error (mean over time)
    total_error = perc_error_pdmd.mean()

    # Plot
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(time_star, rel_error_pdmd, lw=2, color="tab:orange")
    ax.set_xlabel(r"$t^* = t U_{ref} / L_{ref}$", fontsize=13)
    ax.set_ylabel(r"Relative $L^2$ Error", fontsize=13)
    ax.set_title(r"ParametricDMD Interpolation Reconstruction Error"
                 f"\n $Unseen^*$ Parameter $(Re = {Re_test})$",
                 fontsize=15)
    ax.grid(True, alpha=0.6)

    # Place percentage error text where legend would be
    ax.text(0.98, 0.95, f"Mean Error = {total_error:.2f}%",
            transform=ax.transAxes, fontsize=12,
            ha="right", va="top",
            bbox=dict(facecolor="white", alpha=0.7, edgecolor="none"))

    ax.set_xlim(time_star[0], time_star[-1])
    ax.set_xticks(np.linspace(time_star[0], time_star[-1], 6))

    plt.tight_layout()
    plt.show()


