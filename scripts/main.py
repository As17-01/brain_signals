import pathlib
import sys

import hydra
import numpy as np
import pandas as pd
import omegaconf
import torch
from hydra_slayer import Registry
from loguru import logger
from omegaconf import DictConfig
from torch.utils.data import DataLoader

sys.path.append("../")

import src.utils
import src.preprocessing


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

    batch_size = cfg.batch_size
    lr = cfg.learning_rate
    num_epochs = cfg.num_epochs

    # cfg_dct = omegaconf.OmegaConf.to_container(cfg, resolve=True)
    # registry = Registry()
    # registry.add_from_module(src.datasets, prefix="src.datasets.")
    # registry.add_from_module(src.models, prefix="src.models.")

    # train_cfg = cfg_dct["dataset"].copy()
    # train_cfg["annotation_file"] = load_dir / "train.csv"
    # train_data = registry.get_from_params(**train_cfg)

    # valid_cfg = cfg_dct["dataset"].copy()
    # valid_cfg["annotation_file"] = load_dir / "valid.csv"
    # valid_data = registry.get_from_params(**valid_cfg)

    # train_dataloader = DataLoader(train_data, num_workers=num_workers, batch_size=batch_size, shuffle=True)
    # valid_dataloader = DataLoader(valid_data, num_workers=num_workers, batch_size=batch_size, shuffle=False)

    # train_dataloader = src.utils.DeviceDataLoader(train_dataloader, device)
    # valid_dataloader = src.utils.DeviceDataLoader(valid_dataloader, device)

    # model = registry.get_from_params(**cfg_dct["model"])
    # src.utils.to_device(model, device)

    # with torch.no_grad():
    #     model.eval()
    #     history = [src.utils.evaluate(model, valid_dataloader)]

    # history += src.utils.fit(num_epochs, lr, model, train_dataloader, valid_dataloader)


if __name__ == "__main__":
    main()