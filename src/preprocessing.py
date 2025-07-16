import numpy as np


def reshape_dataset(data, id_col, target_col, feature_cols):
    features = []
    target = []
    for cur_id in np.unique(data[id_col].to_numpy()):
        cur_id_data = data[data[id_col].to_numpy() == cur_id]
        target.append(np.mean(cur_id_data[target_col].to_numpy()).astype("int"))
        features.append(cur_id_data[feature_cols].to_numpy())

    features = pad_sequences(features)
    return np.array(features), np.array(target).reshape(-1, 1)


def pad_sequences(arrays, pad_value=0):
    max_length = max(arr.shape[0] for arr in arrays)
    padded_arrays = [
        np.pad(
            arr,
            ((0, max_length - arr.shape[0]), (0, 0)),
            mode="constant",
            constant_values=pad_value,
        )
        for arr in arrays
    ]
    return np.stack(padded_arrays)


def process_data(data, id_col, user_col, video_col, target_col, feature_cols):
    data = data.copy()

    num_videos = len(np.unique(data[video_col]))
    data[id_col] = (num_videos * data[user_col] + data[video_col]).astype("int")
    data = data[[id_col] + [user_col] + feature_cols + [target_col]]

    user_mapping = {}
    for id in np.unique(data[id_col]):
        user_mapping[id] = id // num_videos

    features, target = reshape_dataset(data, id_col, target_col, feature_cols)
    return user_mapping, features, target
