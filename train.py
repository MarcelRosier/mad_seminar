import matplotlib.pyplot as plt
import pytorch_lightning as pl
import torch
import yaml

from model.model import get_model
from data_loader import TrainDataModule, get_all_test_dataloaders

# with open("./configs/ganomaly_config.yaml", "r") as f:
with open("./configs/ganomaly_config.yaml", "r") as f:
    config = yaml.safe_load(f)

print(config)
# Reproducibility
pl.seed_everything(config["seed"])

train_data_module = TrainDataModule(
    split_dir=config["split_dir"],
    target_size=config["target_size"],
    batch_size=config["batch_size"],
)

# Init model
model = get_model(config)

# Use tensorboard logger and CSV logger
trainer = pl.Trainer(
    max_epochs=config["num_epochs"],
    logger=[
        pl.loggers.TensorBoardLogger(save_dir="./"),
        pl.loggers.CSVLogger(save_dir="./"),
    ],
    accelerator="gpu",
    devices=1,
)

trainer.fit(model, datamodule=train_data_module)
