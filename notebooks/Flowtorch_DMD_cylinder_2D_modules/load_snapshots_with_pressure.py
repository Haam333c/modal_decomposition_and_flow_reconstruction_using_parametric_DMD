"""
Extended snapshot loader for velocity + pressure fields.

This block modifies the workflow so that both velocity (Ux, Uy)
and pressure (p) snapshots are included in the training and test
datasets. The stacked snapshot matrices can then be used in DMD/PROM
to reconstruct both velocity and pressure fields consistently.
"""

import numpy as np
from pathlib import Path

def load_masked_matrix(loader, mask_indices, times, include_pressure=False):
    """
    Load masked snapshots for selected times.

    Parameters:
    - loader: FOAMDataloader object that reads simulation data.
    - mask_indices: integer indices of masked points.
    - times: list of time steps to load.
    - include_pressure: if True, also load pressure field.

    Returns:
    - data_matrix: stacked array with shape ((2 + P) * num_points, num_times),
                   where P=1 if pressure included, else 0.
                   Layout: [Ux; Uy; p] if include_pressure=True
    - num_points: number of spatial points selected by mask.
    """
    num_points = len(mask_indices)
    num_times = len(times)
    n_fields = 2 + (1 if include_pressure else 0)  # 2 for Ux, Uy; +1 for p
    data_matrix = np.zeros((n_fields * num_points, num_times), dtype=np.float64)

    for i, time in enumerate(times):
        # Load velocity snapshot (U field)
        snapshot_U = loader.load_snapshot("U", time).numpy()  # shape: (n_points, 3)
        ux_masked = snapshot_U[mask_indices, 0]
        uy_masked = snapshot_U[mask_indices, 1]

        # Fill velocity components
        data_matrix[:num_points, i] = ux_masked
        data_matrix[num_points:2*num_points, i] = uy_masked

        # Optionally load pressure snapshot
        if include_pressure:
            snapshot_p = loader.load_snapshot("p", time).numpy()  # shape: (n_points,)
            p_masked = snapshot_p[mask_indices]
            data_matrix[2*num_points:, i] = p_masked

    return data_matrix, num_points


def load_all_snapshots(Re_list, base_path, mask_box, FOAMDataloader,
                       training_window=(10.0, 15.0), future_window=(15.0, 20.0),
                       sampling_step=1, include_pressure=False):
    """
    Load training and future snapshots for all Reynolds numbers in Re_list.

    Parameters:
    - Re_list: list of Reynolds numbers for training.
    - base_path: base folder containing simulation data.
    - mask_box: function to generate spatial mask.
    - FOAMDataloader: class to load OpenFOAM data.
    - training_window: time window for training snapshots.
    - future_window: time window for forecast snapshots.
    - sampling_step: downsampling step for time indices.
    - include_pressure: if True, also load pressure field.

    Returns:
    - data: dictionary containing snapshot matrices and metadata.
    """
    snapshot_dict = {}
    sampled_times_dict = {}
    snapshot_future_dict = {}
    sampled_times_future_dict = {}
    mask_dict = {}
    num_points_dict = {}
    loader_dict = {}
    masked_coords_dict = {}

    # Precompute mask once using the first Re (same mesh assumed)
    folder0 = Path(base_path).expanduser() / f"cylinder_2D_Re{Re_list[0]}"
    loader0 = FOAMDataloader(str(folder0))
    vertices0 = loader0.vertices[:, :2]
    global_mask = mask_box(vertices0, lower=[0.1, -1], upper=[0.75, 1])
    mask_indices = np.where(global_mask.numpy())[0]
    global_masked_coords = vertices0[mask_indices]
    num_points_global = len(mask_indices)

    for Re in Re_list:
        folder = Path(base_path).expanduser() / f"cylinder_2D_Re{Re}"
        loader = FOAMDataloader(str(folder))
        times = loader.write_times

        # Training window
        sampled_times = [t for t in times if training_window[0] <= float(t) <= training_window[1]][::sampling_step]
        data_matrix, num_points = load_masked_matrix(loader, mask_indices, sampled_times, include_pressure=include_pressure)

        snapshot_dict[Re] = data_matrix
        sampled_times_dict[Re] = sampled_times
        mask_dict[Re] = global_mask
        num_points_dict[Re] = num_points
        loader_dict[Re] = loader
        masked_coords_dict[Re] = global_masked_coords

        # Future window
        sampled_future_times = [t for t in times if future_window[0] <= float(t) <= future_window[1]][::sampling_step]
        if sampled_future_times:
            future_data_matrix, _ = load_masked_matrix(loader, mask_indices, sampled_future_times, include_pressure=include_pressure)
            snapshot_future_dict[Re] = future_data_matrix
            sampled_times_future_dict[Re] = sampled_future_times

        print(f"Re={Re}: Masked points = {num_points_global}, include_pressure={include_pressure}")

    print("All snapshots loaded.")
    return {
        "snapshot_dict": snapshot_dict,
        "sampled_times_dict": sampled_times_dict,
        "mask_dict": mask_dict,
        "num_points_dict": num_points_dict,
        "loader_dict": loader_dict,
        "masked_coords_dict": masked_coords_dict,
        "snapshot_future_dict": snapshot_future_dict,
        "sampled_times_future_dict": sampled_times_future_dict
    }


def load_test_parameter(Re, path, mask_box, FOAMDataloader,
                        forecast_window=(15.0, 20.0),
                        sampling_step=1, include_pressure=False):
    """
    Load test Reynolds number data aligned ONLY with the forecast window.

    Parameters:
    - Re: test Reynolds number.
    - path: folder path for test case.
    - mask_box: function to generate spatial mask.
    - FOAMDataloader: class to load OpenFOAM data.
    - forecast_window: time window for forecast snapshots.
    - sampling_step: downsampling step.
    - include_pressure: if True, also load pressure field.

    Returns:
    - test_data: dictionary with forecast snapshots and metadata.
    """
    loader = FOAMDataloader(path)
    times = loader.write_times
    vertices = loader.vertices[:, :2]

    # Global mask
    mask = mask_box(vertices, lower=[0.1, -1], upper=[0.75, 1])
    mask_indices = np.where(mask.numpy())[0]
    masked_coords = vertices[mask_indices]

    # Forecast window only
    sampled_times_forecast = [t for t in times if forecast_window[0] <= float(t) <= forecast_window[1]][::sampling_step]
    snapshot_forecast, num_points = load_masked_matrix(loader, mask_indices, sampled_times_forecast, include_pressure=include_pressure)

    print(f"Test Re={Re}: forecast snapshot shape = {snapshot_forecast.shape}, include_pressure={include_pressure}")

    return {
        "snapshot_forecast": snapshot_forecast,
        "sampled_times_forecast": sampled_times_forecast,
        "mask": mask,
        "num_points": num_points,
        "loader": loader,
        "masked_coords": masked_coords
    }
