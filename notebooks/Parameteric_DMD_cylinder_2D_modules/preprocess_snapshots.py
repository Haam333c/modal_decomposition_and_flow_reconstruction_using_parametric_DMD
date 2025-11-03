import numpy as np

def preprocess_snapshots(snapshot_dict, Re_list):
    """
    Preprocesses velocity snapshots by subtracting the mean flow and normalizing each block.

    Steps:
    1. Stack all training snapshots across Reynolds numbers and time.
    2. Compute mean flow from the stacked data.
    3. Subtract mean flow and normalize each snapshot block.
    4. Stack normalized snapshots into a 3D array.

    Parameters:
    - snapshot_dict: Dictionary of raw velocity snapshots per Reynolds number.
    - Re_list: List of Reynolds numbers used for training.

    Returns:
    - train_snapshots: Array of shape (n_Re, space_dim, n_time)
    - mean_flow: Mean flow vector of shape (space_dim,)
    - snapshot_processed_dict: Dictionary of normalized snapshots per Re
    - norm_scales: Dictionary of normalization scales per Re
    """
    # Step 1: Stack all training snapshots across Re and time
    all_training_snapshots = np.concatenate([
        snapshot_dict[Re].T for Re in Re_list
    ], axis=0)  # shape: (n_total_snapshots, space_dim)

    # Step 2: Compute mean flow
    mean_flow = np.mean(all_training_snapshots, axis=0)  # shape: (space_dim,)
    print("Mean flow computed from training window Shape:", mean_flow.shape)

    # Step 3: Subtract mean and normalize
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

    # Step 4: Stack normalized snapshots
    train_snapshots = np.array([
        snapshot_processed_dict[Re] for Re in Re_list
    ])
    print("train_snapshots shape:", train_snapshots.shape)

    return train_snapshots, mean_flow, snapshot_processed_dict, norm_scales
