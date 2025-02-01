import numpy as np

def _check_seed(seed: int):
    if seed is not None:
        np.random.seed(seed)

def shift_time_series(original_samples: np.ndarray, num_samples: int, n_timestamps=10, seed=None):
    _check_seed(seed=seed)
    
    indices = np.random.choice(len(original_samples), num_samples, replace=True)
    selected_samples = original_samples[indices].copy()
    
    shifts = np.random.randint(1, n_timestamps + 1, num_samples)
    
    for i in range(num_samples):
        selected_samples[i] = np.roll(selected_samples[i], shifts[i])
        selected_samples[i][:shifts[i]] = 0
    return selected_samples, indices

def add_noise(original_samples: np.ndarray, num_samples: int, noise_intensity=0.01, seed=None):
    _check_seed(seed=seed)
    
    indices = np.random.choice(len(original_samples), num_samples, replace=True)
    selected_samples = original_samples[indices].copy()
    
    noise = np.random.normal(0, noise_intensity, selected_samples.shape)
    return selected_samples + noise, indices

def mask_time_series(original_samples: np.ndarray, num_samples: int, mask_length=100, seed=None):
    _check_seed(seed=seed)
    
    indices = np.random.choice(len(original_samples), num_samples, replace=True)
    selected_samples = original_samples[indices].copy()
    
    starts = np.random.randint(0, selected_samples.shape[1] - mask_length, num_samples)
    
    for i in range(num_samples):
        selected_samples[i, starts[i]:starts[i] + mask_length] = 0
    return selected_samples, indices

def augment_timeseries(
    X, y,
    num_shift=100, shift_params={},
    num_noise=100, noise_params={},
    num_mask=100, mask_params={}
):
    augmented_X = [X]
    augmented_y = [y]
    
    if num_shift > 0:
        shift_aug, shift_indices = shift_time_series(X, num_shift, **shift_params)
        augmented_X.append(shift_aug)
        augmented_y.append(y[shift_indices])
    
    if num_noise > 0:
        noise_aug, noise_indices = add_noise(X, num_noise, **noise_params)
        augmented_X.append(noise_aug)
        augmented_y.append(y[noise_indices])
    
    if num_mask > 0:
        mask_aug, mask_indices = mask_time_series(X, num_mask, **mask_params)
        augmented_X.append(mask_aug)
        augmented_y.append(y[mask_indices])

    return np.vstack(augmented_X), np.hstack(augmented_y)
