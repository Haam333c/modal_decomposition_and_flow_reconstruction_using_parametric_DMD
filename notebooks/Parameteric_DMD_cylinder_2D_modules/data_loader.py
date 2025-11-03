"""
load_snapshots.py

This file helps load velocity data from OpenFOAM simulations.
It applies a spatial mask, selects time windows for training and future prediction, and returns the data in a structured format.

Functions:
- load_masked_matrix: Loads velocity snapshots for selected times and masked region.
- load_all_snapshots: Loads data for each Reynolds number, applies the mask, and returns training and future snapshots.
- load_test_snapshots: Loads data for a single Reynolds number not used in training (for testing or comparison).
"""

import torch as pt
from pathlib import Path

def load_masked_matrix(loader, mask, times):
    """
    Loads velocity snapshots for selected times and masked region.

    Parameters:
    - loader: FOAMDataloader object that reads simulation data.
    - mask: Boolean tensor that selects part of the domain.
    - times: List of time steps to load.

    Returns:
    - data_matrix: Array with shape (2*num_points, num_times), containing u and v velocities.
    - num_points: Number of spatial points selected by the mask.
    """
    num_points = mask.sum().item()
    num_times = len(times)
    data_matrix = pt.zeros((2 * num_points, num_times), dtype=pt.float64)

    for i, time in enumerate(times):
        snapshot = loader.load_snapshot("U", time)
        ux_masked = pt.masked_select(snapshot[:, 0], mask)
        uy_masked = pt.masked_select(snapshot[:, 1], mask)
        data_matrix[:num_points, i] = ux_masked
        data_matrix[num_points:, i] = uy_masked

    return data_matrix.numpy(), num_points

def load_all_snapshots(Re_list, base_path, mask_box, FOAMDataloader, training_window=(10.0, 15.0), future_window=(15.0, 20.0), sampling_step=1):
    """
    Loads velocity data for each Reynolds number.

    For each case:
    - Picks time steps for training and future prediction.
    - Applies a spatial mask to focus on a region.
    - Loads velocity snapshots for those times.

    Parameters:
    - Re_list: List of Reynolds numbers to load.
    - base_path: Folder where simulation data is stored.
    - mask_box: Function that creates a spatial mask.
    - FOAMDataloader: Class that reads OpenFOAM data.
    - training_window: Time range for training data (start, end).
    - future_window: Time range for future data (start, end).
    - sampling_step: How often to sample time steps.

    Returns:
    - A dictionary with:
        - snapshot_dict: Training velocity data
        - sampled_times_dict: Training time steps
        - mask_dict: Spatial masks
        - num_points_dict: Number of masked points
        - loader_dict: FOAMDataloader objects
        - masked_coords_dict: Coordinates of masked points
        - snapshot_future_dict: Future velocity data
        - sampled_times_future_dict: Future time steps
    """
    snapshot_dict = {}
    sampled_times_dict = {}
    mask_dict = {}
    num_points_dict = {}
    loader_dict = {}
    masked_coords_dict = {}
    snapshot_future_dict = {}
    sampled_times_future_dict = {}

    for Re in Re_list:
        folder = Path(base_path).expanduser() / f"cylinder_2D_Re{Re}"
        loader = FOAMDataloader(str(folder))
        times = loader.write_times

        sampled_times = [t for t in times if training_window[0] <= float(t) <= training_window[1]][::sampling_step]

        vertices = loader.vertices[:, :2]
        mask = mask_box(vertices, lower=[0.1, -1], upper=[0.75, 1])
        masked_coords = vertices[mask.numpy()]
        print(f"Re={Re}: Masked points = {mask.sum().item()}")

        data_matrix, num_points = load_masked_matrix(loader, mask, sampled_times)

        snapshot_dict[Re] = data_matrix
        sampled_times_dict[Re] = sampled_times
        mask_dict[Re] = mask
        num_points_dict[Re] = num_points
        loader_dict[Re] = loader
        masked_coords_dict[Re] = masked_coords

        sampled_future_times = [t for t in times if future_window[0] <= float(t) <= future_window[1]][::sampling_step]
        if sampled_future_times:
            future_data_matrix, _ = load_masked_matrix(loader, mask, sampled_future_times)
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
    """
    Loads velocity snapshots for a single test Reynolds number not used in training.

    This is useful for testing or comparing true data against interpolated results.

    Parameters:
    - Re: Reynolds number to load.
    - path: Full path to the simulation folder for this Re.
    - mask_box: Function to create a spatial mask.
    - FOAMDataloader: Class that reads OpenFOAM data.
    - time_window: Time range to load (start, end).
    - sampling_step: How often to sample time steps.

    Returns:
    - snapshot: Velocity data array (2*num_points, num_times)
    - num_points: Number of masked spatial points
    - sampled_times: List of time steps used
    - mask: Boolean tensor used for masking
    - masked_coords: Coordinates of masked points
    - loader: FOAMDataloader instance used to access metadata
    """
    loader = FOAMDataloader(path)
    times = loader.write_times

    sampled_times = [t for t in times if time_window[0] <= float(t) <= time_window[1]][::sampling_step]
    vertices = loader.vertices[:, :2]
    mask = mask_box(vertices, lower=[0.1, -1], upper=[0.75, 1])
    masked_coords = vertices[mask.numpy()]

    snapshot, num_points = load_masked_matrix(loader, mask, sampled_times)

    print(f"Test Re={Re}: Loaded snapshot shape = {snapshot.shape}")
    return snapshot, num_points, sampled_times, mask, masked_coords, loader

