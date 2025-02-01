import numpy as np
from scipy.stats import kurtosis, skew

def calculate_summary_features(arr, n_windows=20):
    features = []

    cur_arr = np.array_split(arr, n_windows, axis=1)
    for window in cur_arr:
        features.append(np.mean(window, axis=1))
        features.append(np.mean(np.abs(window), axis=1))
        features.append(np.std(window, axis=1))
        features.append(np.sqrt(np.mean(np.square(window), axis=1)))
        features.append(kurtosis(window, axis=1, fisher=True))
        features.append(skew(window, axis=1))

    features = np.column_stack(features)    
    return features
