import numpy as np

def preprocess_snapshots(snapshot_dict, Re_list, Re_test=None, test_data=None):
    """
    Preprocesses velocity snapshots by subtracting the mean flow and normalizing each block.

    Parameters:
    - snapshot_dict: Dictionary of raw velocity snapshots per Reynolds number (training set).
    - Re_list: List of Reynolds numbers used for training.
    - Re_test: Optional test Reynolds number identifier.
    - test_data: Optional dictionary containing test snapshots (from load_test_parameter).

    Returns:
    - train_snapshots: Array of shape (n_Re, space_dim, n_time)
    - mean_flow: Mean flow vector of shape (space_dim,)
    - snapshot_processed_dict: Dictionary of normalized snapshots per Re (includes test if provided)
    - norm_scales: Dictionary of normalization scales per Re (includes test if provided)
    """
    # Step 1: Stack all training snapshots across Re and time
    all_training_snapshots = np.concatenate([
        snapshot_dict[Re].T for Re in Re_list
    ], axis=0)

    # Step 2: Compute mean flow
    mean_flow = np.mean(all_training_snapshots, axis=0)
    print("Mean flow computed from training window Shape:", mean_flow.shape)

    # Step 3: Process training blocks
    snapshot_processed_dict, norm_scales = {}, {}
    for Re in Re_list:
        snapshots = snapshot_dict[Re].copy()
        snapshots -= mean_flow[:, np.newaxis]
        norm = np.linalg.norm(snapshots)
        snapshots /= norm
        snapshot_processed_dict[Re] = snapshots
        norm_scales[Re] = norm

    train_snapshots = np.array([snapshot_processed_dict[Re] for Re in Re_list])
    print("train_snapshots shape:", train_snapshots.shape)

    # Step 4: Process test Re if provided
    if Re_test is not None:
        if test_data is not None and "snapshot_forecast" in test_data:
            # Use separately loaded test_data
            snapshots = test_data["snapshot_forecast"].copy()
        elif Re_test in snapshot_dict:
            # Use test snapshots already merged into snapshot_dict
            snapshots = snapshot_dict[Re_test].copy()
        else:
            raise KeyError(f"Test Re={Re_test} not found in snapshot_dict and no test_data provided.")

        snapshots -= mean_flow[:, np.newaxis]
        norm = np.linalg.norm(snapshots)
        snapshots /= norm
        snapshot_processed_dict[Re_test] = snapshots
        norm_scales[Re_test] = norm
        print(f"Test Re={Re_test}: processed shape = {snapshots.shape}, norm scale = {norm:.4f}")

    return train_snapshots, mean_flow, snapshot_processed_dict, norm_scales
