import numpy as np
from ezyrb import POD, RBF
from pydmd import DMD, ParametricDMD

def prepare_noisy_data(train_snapshots, noise_levels, pipeline_type="snapshot", pod_rank=30):
    """
    Prepares noisy data for ParametricDMD training based on selected pipeline.

    Parameters:
    - train_snapshots: Clean training snapshot array of shape (n_Re, space_dim, n_time)
    - noise_levels: List of noise levels to inject (e.g., [0, 10, 20])
    - pipeline_type: 'snapshot' or 'pod'
    - pod_rank: POD rank for basis construction

    Returns:
    - rom: POD basis object (fitted only for 'pod' pipeline)
    - coeffs_clean: Clean modal coefficients (only for 'pod' pipeline)
    - coeffs_noisy_dict: Dict of noisy modal coefficients (only for 'pod' pipeline)
    - snapshot_noisy_dict: Dict of noisy snapshots (only for 'snapshot' pipeline)
    """
    n_parameters = train_snapshots.shape[0]
    space_dim = train_snapshots.shape[1]
    n_time_steps = train_snapshots.shape[2]

    # Reshape snapshots to 2D matrix for POD
    train_snapshots_2d = train_snapshots.transpose(1, 0, 2).reshape(space_dim, -1)

    # Create POD basis (only fit if using 'pod' pipeline)
    rom = POD(rank=pod_rank, method='randomized_svd')
    coeffs_clean = None
    coeffs_noisy_dict = {}
    snapshot_noisy_dict = {}

    if pipeline_type == "pod":
        # Fit POD basis
        rom.fit(train_snapshots_2d)

        # Project to modal coefficients
        coeffs_clean = rom.transform(train_snapshots_2d)  # shape: (n_modes, total_snapshots)
        coeffs_clean = coeffs_clean.reshape(coeffs_clean.shape[0], n_parameters, n_time_steps)

        # Inject noise into modal coefficients
        for level in noise_levels:
            if level == 0:
                coeffs_noisy = coeffs_clean.copy()
            else:
                std = np.std(coeffs_clean, axis=(1, 2), keepdims=True)
                noise = np.random.randn(*coeffs_clean.shape) * std * (level * 0.01)
                coeffs_noisy = coeffs_clean + noise
            coeffs_noisy_dict[level] = coeffs_noisy

    elif pipeline_type == "snapshot":
        # Inject noise into raw snapshots
        for level in noise_levels:
            if level == 0:
                noisy = train_snapshots.copy()
            else:
                reshaped = train_snapshots.transpose(1, 0, 2).reshape(space_dim, -1)
                std = np.std(reshaped, axis=1, keepdims=True)
                noise = np.random.randn(*reshaped.shape) * std * (level * 0.01)
                noisy_reshaped = reshaped + noise
                noisy = noisy_reshaped.reshape(space_dim, n_parameters, n_time_steps).transpose(1, 0, 2)
            snapshot_noisy_dict[level] = noisy

    else:
        raise ValueError("pipeline_type must be 'snapshot' or 'pod'")

    return rom, coeffs_clean, coeffs_noisy_dict, snapshot_noisy_dict




def train_parametric_dmd(
    noise_levels,
    noise_analysis_type,
    snapshot_noisy_dict,
    coeffs_noisy_dict,
    rom,
    Re_list,
    n_time_steps,
    sampled_times_dict
):
    """
    Trains ParametricDMD models for each noise level using pydmd.DMD.

    Parameters:
    - noise_levels: List of noise levels (e.g., [0, 10, 20])
    - noise_analysis_type: 'snapshot' or 'pod'
    - snapshot_noisy_dict: Dict of noisy snapshots (used if 'snapshot')
    - coeffs_noisy_dict: Dict of noisy modal coefficients (used if 'pod')
    - rom: POD basis object (fitted only for 'pod' pipeline)
    - Re_list: List of training parameters (Reynolds numbers)
    - n_time_steps: Number of time steps per Re
    - sampled_times_dict: Dict of physical time vectors per Re

    Returns:
    - pdmd_models: Dict of trained ParametricDMD models per noise level
    - cached_dmd_lists: Dict of DMD instances per noise level
    - cached_modal_coeffs: Dict of modal coefficient arrays per noise level
    - rom: POD basis object (fitted in 'pod', unfitted in 'snapshot')
    """
    pdmd_models = {}
    cached_dmd_lists = {}
    cached_modal_coeffs = {}

    for level in noise_levels:
        print(f"\nTraining ParametricDMD with {level}% noise...")

        if noise_analysis_type == "snapshot":
            noisy_snapshots = snapshot_noisy_dict[level]  # shape: (n_Re, space_dim, n_time_steps)

            trained_dmds = [DMD(svd_rank=-1) for _ in Re_list]
            interpolator = RBF()

            # Use the provided (unfitted) rom — ParametricDMD will fit it internally
            pdmd = ParametricDMD(trained_dmds, rom, interpolator)
            pdmd.fit(noisy_snapshots, np.array(Re_list).reshape(-1, 1))

            pdmd_models[level] = pdmd
            cached_dmd_lists[level] = pdmd._dmd.copy()
            cached_modal_coeffs[level] = pdmd.training_modal_coefficients.copy()

        elif noise_analysis_type == "pod":
            coeffs_noisy = coeffs_noisy_dict[level]  # shape: (n_modes, n_Re, n_time_steps)

            trained_dmds = []
            modal_coeffs_by_param = []

            for i, Re in enumerate(Re_list):
                coeffs_i = coeffs_noisy[:, i, :]
                dmd = DMD(svd_rank=-1)
                dmd.fit(coeffs_i)
                trained_dmds.append(dmd)
                modal_coeffs_by_param.append(dmd.dynamics)

            interpolator = RBF()
            pdmd = ParametricDMD(trained_dmds, rom, interpolator)

            pdmd._training_parameters = np.array(Re_list).reshape(-1, 1)
            pdmd._training_modal_coefficients = modal_coeffs_by_param
            pdmd._ntrain = len(Re_list)
            interpolator.fit(pdmd._training_parameters, pdmd._training_modal_coefficients)

            time_vec = np.array(sampled_times_dict[Re_list[0]], dtype=float)
            pdmd.original_time["t0"] = time_vec[0]
            pdmd.original_time["dt"] = time_vec[1] - time_vec[0]
            pdmd.original_time["tend"] = time_vec[-1]
            pdmd.dmd_time = pdmd.original_time.copy()

            pdmd_models[level] = pdmd
            cached_dmd_lists[level] = trained_dmds
            cached_modal_coeffs[level] = modal_coeffs_by_param

        else:
            raise ValueError("noise_analysis_type must be 'snapshot' or 'pod'")

    return pdmd_models, cached_dmd_lists, cached_modal_coeffs, rom
