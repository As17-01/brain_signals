import pathlib
import sys

import hydra
import numpy as np
import pandas as pd
import random
import omegaconf
import torch
from hydra_slayer import Registry
from loguru import logger
from omegaconf import DictConfig
from torch.utils.data import DataLoader, TensorDataset

sys.path.append("../")

import src.utils
import src.preprocessing
import src.training
import src.models


@hydra.main(config_path="configs", config_name="config", version_base="1.2")
def main(cfg: DictConfig) -> None:
    load_dir = pathlib.Path(cfg.load_dir)
    load_path = load_dir / "EEG_data.csv"
    data = pd.read_csv(load_path)
    logger.info(f"{len(data)} rows successfully loaded from: {load_path}")

    user_mapping, features, target = src.preprocessing.process_data(data, cfg.id_col, cfg.user_col, cfg.video_col, cfg.target_col, cfg.feature_cols)
    logger.info(f"Features shape: {features.shape}; Target shape: {target.shape}")

    device = src.utils.get_default_device()
    logger.info(f"Current device is {device}")

    registry = Registry()
    registry.add_from_module(src.models, prefix="src.models.")
    

    for i in range(cfg.num_folds):
        logger.info(f"Fold {i + 1} / {cfg.num_folds}")

        # For reproducibility
        np.random.seed(cfg.seed + i)
        random.seed(cfg.seed + i)
        torch.manual_seed(cfg.seed + i)

        unique_users = np.unique(list(user_mapping.values()))
        train_users = np.random.choice(unique_users, 7, replace=False)
        train_index = [index for index, user in user_mapping.items() if user in train_users]
        test_index = [index for index, user in user_mapping.items() if user not in train_users]

        train_features, train_target = torch.from_numpy(features[train_index]), torch.from_numpy(target[train_index])
        test_features, test_target = torch.from_numpy(features[test_index]), torch.from_numpy(target[test_index])

        train_features.to(device), train_target.to(device)
        test_features.to(device), test_target.to(device)

        train_data = TensorDataset(train_features, train_target)
        test_data = TensorDataset(test_features, test_target)

        train_dataloader = DataLoader(train_data, num_workers=2, batch_size=cfg.batch_size, shuffle=True)
        test_dataloader = DataLoader(test_data, num_workers=2, batch_size=cfg.batch_size, shuffle=False)

        model = registry.get_from_params(**cfg["model"])
        model.to(device)

        logger.info("Start training...")
        optimizer = torch.optim.Adam(model.parameters(), cfg.learning_rate)
        history_train, history_test = src.training.fit(cfg.num_epochs, model, train_dataloader, test_dataloader, optimizer)
        print(history_train)
        print(history_test)


if __name__ == "__main__":
    main()