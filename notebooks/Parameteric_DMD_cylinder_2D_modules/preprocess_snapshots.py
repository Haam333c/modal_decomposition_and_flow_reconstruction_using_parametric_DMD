import numpy as np

def preprocess_snapshots(snapshot_dict, Re_list, Re_test=None):
    """
    Preprocesses velocity snapshots by subtracting the mean flow and normalizing each block.

    Steps:
    1. Stack all training snapshots across Reynolds numbers and time.
    2. Compute mean flow from the stacked data.
    3. Subtract mean flow and normalize each snapshot block.
    4. Stack normalized snapshots into a 3D array.
    5. Optionally process a test Reynolds number with the same mean flow.

    Parameters:
    - snapshot_dict: Dictionary of raw velocity snapshots per Reynolds number.
    - Re_list: List of Reynolds numbers used for training.
    - Re_test: Optional test Reynolds number to preprocess with same mean flow.

    Returns:
    - train_snapshots: Array of shape (n_Re, space_dim, n_time)
    - mean_flow: Mean flow vector of shape (space_dim,)
    - snapshot_processed_dict: Dictionary of normalized snapshots per Re (includes test if provided)
    - norm_scales: Dictionary of normalization scales per Re (includes test if provided)
    """
    # Step 1: Stack all training snapshots across Re and time
    all_training_snapshots = np.concatenate([
        snapshot_dict[Re].T for Re in Re_list
    ], axis=0)  # shape: (n_total_snapshots, space_dim)

    # Step 2: Compute mean flow
    mean_flow = np.mean(all_training_snapshots, axis=0)  # shape: (space_dim,)
    print("Mean flow computed from training window Shape:", mean_flow.shape)

    # Step 3: Subtract mean and normalize training blocks
    snapshot_processed_dict = {}
    norm_scales = {}
    for Re in Re_list:
        snapshots = snapshot_dict[Re].copy()
        snapshots -= mean_flow[:, np.newaxis]
        norm = np.linalg.norm(snapshots)
        snapshots /= norm
        snapshot_processed_dict[Re] = snapshots
        norm_scales[Re] = norm

    print("Mean flow subtracted and snapshots normalized for each snapshot matrix.")

    # Step 4: Stack normalized training snapshots
    train_snapshots = np.array([snapshot_processed_dict[Re] for Re in Re_list])
    print("train_snapshots shape:", train_snapshots.shape)

    # Step 5: Optionally process test Re
    if Re_test is not None:
        snapshots = snapshot_dict[Re_test].copy()
        snapshots -= mean_flow[:, np.newaxis]
        norm = np.linalg.norm(snapshots)
        snapshots /= norm
        snapshot_processed_dict[Re_test] = snapshots
        norm_scales[Re_test] = norm
        print(f"Test Re={Re_test}: processed shape = {snapshots.shape}, norm scale = {norm:.4f}")

    return train_snapshots, mean_flow, snapshot_processed_dict, norm_scales


def compute_average_inlet_velocity(loader, tol=1e-3, time=None):
    """
    Computes the average inlet velocity magnitude at a given time by automatically
    detecting the inlet as the face with the minimum x-coordinate.

    Parameters:
    - loader: FOAMDataloader instance for a given Re case
    - tol: tolerance for identifying inlet points near the minimum x-coordinate
    - time: time step to extract velocity from (default: first available)

    Returns:
    - U_infty: average velocity magnitude at the inlet
    """
    if time is None:
        time = loader.write_times[0]  # use first available time if not specified

    snapshot = loader.load_snapshot("U", time)  # shape: (n_points, 3)
    vertices = loader.vertices[:, :2]  # shape: (n_points, 2)

    if hasattr(vertices, "detach"):
        vertices = vertices.detach().cpu().numpy()

    inlet_x = np.min(vertices[:, 0])
    inlet_mask = np.abs(vertices[:, 0] - inlet_x) < tol
    if not inlet_mask.any():
        raise ValueError(f"No inlet points found at x ≈ {inlet_x:.6f} (tol={tol})")

    u_inlet = snapshot[inlet_mask, 0]
    v_inlet = snapshot[inlet_mask, 1]
    if hasattr(u_inlet, "detach"):
        u_inlet = u_inlet.detach().cpu().numpy()
        v_inlet = v_inlet.detach().cpu().numpy()

    U_infty = np.mean(np.sqrt(u_inlet**2 + v_inlet**2))
    return U_infty