"""
load_snapshots.py

This file helps load velocity data from OpenFOAM simulations.
It applies a spatial mask, selects time windows for training and future prediction, and returns the data in a structured format.

Functions:
- load_masked_matrix: Loads velocity snapshots for selected times and masked region.
- load_all_snapshots: Loads data for each Reynolds number, applies the mask, and returns training and future snapshots.
- load_test_snapshots: Loads data for a single Reynolds number not used in training (for testing or comparison).
"""

import numpy as np
import torch as pt
from pathlib import Path

def load_masked_matrix(loader, mask_indices, times):
    """
    Loads velocity snapshots for selected times and masked region (faster version).

    Parameters:
    - loader: FOAMDataloader object that reads simulation data.
    - mask_indices: Precomputed integer indices of masked points.
    - times: List of time steps to load.

    Returns:
    - data_matrix: Array with shape (2*num_points, num_times), containing u and v velocities.
    - num_points: Number of spatial points selected by the mask.
    """
    num_points = len(mask_indices)
    num_times = len(times)
    data_matrix = np.zeros((2 * num_points, num_times), dtype=np.float64)

    for i, time in enumerate(times):
        snapshot = loader.load_snapshot("U", time).numpy()  # shape: (n_points, 3)
        ux_masked = snapshot[mask_indices, 0]
        uy_masked = snapshot[mask_indices, 1]
        data_matrix[:num_points, i] = ux_masked
        data_matrix[num_points:, i] = uy_masked

    return data_matrix, num_points


def load_all_snapshots(Re_list, base_path, mask_box, FOAMDataloader,
                       training_window=(10.0, 15.0), future_window=(15.0, 20.0),
                       sampling_step=1):
    snapshot_dict = {}
    sampled_times_dict = {}
    mask_dict = {}
    num_points_dict = {}
    loader_dict = {}
    masked_coords_dict = {}
    snapshot_future_dict = {}
    sampled_times_future_dict = {}

    # Precompute mask once using the first Re (assuming same mesh for all cases)
    folder0 = Path(base_path).expanduser() / f"cylinder_2D_Re{Re_list[0]}"
    loader0 = FOAMDataloader(str(folder0))
    vertices0 = loader0.vertices[:, :2]
    global_mask = mask_box(vertices0, lower=[0.1, -1], upper=[0.75, 1])
    mask_indices = np.where(global_mask.numpy())[0]   # precompute indices once
    global_masked_coords = vertices0[mask_indices]
    num_points_global = len(mask_indices)

    for Re in Re_list:
        folder = Path(base_path).expanduser() / f"cylinder_2D_Re{Re}"
        loader = FOAMDataloader(str(folder))
        times = loader.write_times

        sampled_times = [t for t in times if training_window[0] <= float(t) <= training_window[1]][::sampling_step]

        print(f"Re={Re}: Masked points = {num_points_global}")

        data_matrix, num_points = load_masked_matrix(loader, mask_indices, sampled_times)

        snapshot_dict[Re] = data_matrix
        sampled_times_dict[Re] = sampled_times
        mask_dict[Re] = global_mask
        num_points_dict[Re] = num_points
        loader_dict[Re] = loader
        masked_coords_dict[Re] = global_masked_coords

        sampled_future_times = [t for t in times if future_window[0] <= float(t) <= future_window[1]][::sampling_step]
        if sampled_future_times:
            future_data_matrix, _ = load_masked_matrix(loader, mask_indices, sampled_future_times)
            snapshot_future_dict[Re] = future_data_matrix
            sampled_times_future_dict[Re] = sampled_future_times

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


def load_test_parameter(Re, path, mask_box, FOAMDataloader, time_window=(3.0, 20.0), sampling_step=1):
    loader = FOAMDataloader(path)
    times = loader.write_times

    sampled_times = [t for t in times if time_window[0] <= float(t) <= time_window[1]][::sampling_step]
    vertices = loader.vertices[:, :2]
    mask = mask_box(vertices, lower=[0.1, -1], upper=[0.75, 1])
    mask_indices = np.where(mask.numpy())[0]   # precompute indices once
    masked_coords = vertices[mask_indices]

    snapshot, num_points = load_masked_matrix(loader, mask_indices, sampled_times)

    print(f"Test Re={Re}: Loaded snapshot shape = {snapshot.shape}")
    return snapshot, num_points, sampled_times, mask, masked_coords, loader

def add_test_parameter(data, Re, path, mask_box, FOAMDataloader,
                       time_window=(3.0, 20.0), sampling_step=1):
    """
    Adds a test Reynolds number to the existing data dictionaries.

    Parameters:
    - data: dictionary returned by load_all_snapshots
    - Re: Reynolds number to load (e.g. 150)
    - path: full path to the simulation folder
    - mask_box: function to create spatial mask
    - FOAMDataloader: class that reads OpenFOAM data
    - time_window: time range to load (start, end)
    - sampling_step: how often to sample time steps

    Updates data in place with keys for the test Re.
    """
    snapshot, num_points, sampled_times, mask, masked_coords, loader = load_test_parameter(
        Re=Re,
        path=path,
        mask_box=mask_box,
        FOAMDataloader=FOAMDataloader,
        time_window=time_window,
        sampling_step=sampling_step
    )

    data["snapshot_dict"][Re] = snapshot
    data["sampled_times_dict"][Re] = sampled_times
    data["mask_dict"][Re] = mask
    data["num_points_dict"][Re] = num_points
    data["loader_dict"][Re] = loader
    data["masked_coords_dict"][Re] = masked_coords

    return data