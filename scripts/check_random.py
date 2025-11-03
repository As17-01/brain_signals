import os
import pathlib
import sys

import hydra
import numpy as np
import pandas as pd
import random
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


def compose_metrics(history_list):
    num_epochs = len(history_list[0])
    metrics = []
    for epoch in range(num_epochs):
        epoch_metrics = {}
        epoch_metrics["Epoch"] = epoch
        epoch_metrics["LOSS MEAN"] = np.round(
            np.mean([history[epoch]["LOSS"] for history in history_list]), 4
        )
        epoch_metrics["AUC MEAN"] = np.round(
            np.mean([history[epoch]["AUC"] for history in history_list]), 4
        )
        epoch_metrics["ACC MEAN"] = np.round(
            np.mean([history[epoch]["ACC"] for history in history_list]), 4
        )

        epoch_metrics["LOSS STD"] = np.round(
            np.std([history[epoch]["LOSS"] for history in history_list]), 4
        )
        epoch_metrics["AUC STD"] = np.round(
            np.std([history[epoch]["AUC"] for history in history_list]), 4
        )
        epoch_metrics["ACC STD"] = np.round(
            np.std([history[epoch]["ACC"] for history in history_list]), 4
        )

        metrics.append(epoch_metrics)
    return metrics


def save_results(metrics, model_name, type):
    result_dir = pathlib.Path(f"../results/{model_name}")
    result_dir.mkdir(exist_ok=True, parents=True)

    filename = result_dir / f"{type}.txt"
    if filename.exists():
        os.remove(filename)

    with open(filename, "w") as f:
        for epoch, _ in enumerate(metrics):
            f.write(
                f"Epoch {(epoch + 1)}: ACC = {metrics[epoch]['ACC MEAN']} std: {metrics[epoch]['ACC STD']}; LOSS = {metrics[epoch]['LOSS MEAN']} std: {metrics[epoch]['LOSS STD']}; AUC = {metrics[epoch]['AUC MEAN']} std: {metrics[epoch]['AUC STD']}\n"
            )


@hydra.main(config_path="configs", config_name="config", version_base="1.2")
def main(cfg: DictConfig) -> None:
    load_dir = pathlib.Path(cfg.load_dir)
    load_path = load_dir / "EEG_data.csv"
    data = pd.read_csv(load_path)
    logger.info(f"{len(data)} rows successfully loaded from: {load_path}")

    user_mapping, features, target = src.preprocessing.process_data(
        data, cfg.id_col, cfg.user_col, cfg.video_col, cfg.target_col, cfg.feature_cols
    )
    logger.info(f"Features shape: {features.shape}; Target shape: {target.shape}")

    device = src.utils.get_default_device()
    logger.info(f"Current device is {device}")

    registry = Registry()
    registry.add_from_module(src.models, prefix="src.models.")

    history_train_list = []
    history_test_list = []

    a = []
    # TODO: Check for unique splits. For now it is checked manually
    for i in range(cfg.num_folds):
        logger.info(f"Fold {i + 1} / {cfg.num_folds}")

        # For reproducibility
        np.random.seed(cfg.seed + i)
        random.seed(cfg.seed + i)
        torch.manual_seed(cfg.seed + i)

        unique_users = np.unique(list(user_mapping.values()))
        train_users = np.random.choice(unique_users, 7, replace=False)
        a.append(set(train_users))
        print(len(a), len(np.unique(a)))


if __name__ == "__main__":
    main()
