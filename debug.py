import matplotlib.pyplot as plt
import pytorch_lightning as pl
import torch
import yaml

from model.model import get_model
from data_loader import (
    TrainDataModule,
    get_all_test_dataloaders,
    get_normal_test_dataloader,
    get_train_dataloader,
)


# with open("./configs/ganomaly_config.yaml", "r") as f:
with open("./configs/ganomaly_config.yaml", "r") as f:
    config = yaml.safe_load(f)


train_data_module = TrainDataModule(
    split_dir=config["split_dir"],
    target_size=config["target_size"],
    batch_size=config["batch_size"],
    debug=True,
)

dl = train_data_module.train_dataloader()

for b in dl:
    print(b[0].shape)
    break
