import numpy as np

def preprocess_snapshots(snapshot_dict, Re_list, snapshot_test, times_test,
                         train_window=(10.0, 15.0)):
    """
    Preprocess velocity snapshots by subtracting the global training mean flow only.
    No normalization is applied.

    Steps:
    1. Stack all training snapshots across Reynolds numbers and time (within train_window).
    2. Compute mean flow from the stacked training data.
    3. Subtract mean flow from each training snapshot block.
    4. Stack mean-subtracted training snapshots into a 3D array.
    5. Align test snapshots to train_window, subtract training mean,
       and also compute its own mean flow separately.

    Parameters
    ----------
    snaplot__error_interpolatedpshot_dict : dict
        Raw velocity snapshots per Reynolds number (training).
    Re_list : list
        Reynolds numbers used for training.
    snapshot_test : ndarray
        Raw velocity snapshots for test Re (space_dim, n_time).
    times_test : array-like
        Time vector for test snapshots.
    train_window : tuple
        Time window for training snapshots (start, end).

    Returns
    -------
    train_snapshots : ndarray
        Array of shape (n_Re, space_dim, n_time) with mean-subtracted training snapshots.
    mean_flow_train : ndarray
        Global mean flow vector from training data (space_dim,).
    snapshot_processed_dict : dict
        Dictionary of mean-subtracted snapshots per Re.
    mean_flow_test : ndarray
        Mean flow vector of test Re (space_dim,).
    snapshot_test_processed : ndarray
        Mean-subtracted test snapshots aligned to train_window.
    """

    # Step 1: Stack all training snapshots
    all_training_snapshots = np.concatenate([snapshot_dict[Re].T for Re in Re_list], axis=0)
    mean_flow_train = np.mean(all_training_snapshots, axis=0)

    # Step 2: Subtract training mean from training blocks
    snapshot_processed_dict = {}
    for Re in Re_list:
        snapshots = snapshot_dict[Re].copy()
        snapshots -= mean_flow_train[:, None]
        snapshot_processed_dict[Re] = snapshots

    train_snapshots = np.array([snapshot_processed_dict[Re] for Re in Re_list])

    # Step 3: Align test snapshots to training window
    times_test = np.array(times_test, dtype=float)  # ensure numeric array
    mask = (times_test >= train_window[0]) & (times_test <= train_window[1])
    snapshot_test_window = snapshot_test[:, mask]

    # Compute test mean separately
    mean_flow_test = np.mean(snapshot_test_window, axis=1)

    # Subtract training mean for ROM consistency
    snapshot_test_processed = snapshot_test_window - mean_flow_train[:, None]


    return train_snapshots, mean_flow_train, snapshot_processed_dict, mean_flow_test, snapshot_test_processed


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
