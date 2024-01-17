import matplotlib.pyplot as plt
import pytorch_lightning as pl
import torch
import yaml
import wandb
from pytorch_lightning.loggers import WandbLogger

from model.model import get_model, Ganomaly
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
# model = get_model(config)

# model = Ganomaly(
#     batch_size=32,
#     input_size=(128, 128),
#     latent_vec_size=100,
#     wadv=1,
#     wcon=50,
#     wenc=1,
#     lr=0.0002,
#     beta1=0.5,
#     beta2=0.999,
# )

##! Custom
model = Ganomaly(
    batch_size=config["batch_size"],
    input_size=(64, 64),
    latent_vec_size=100,
    wadv=1,
    wcon=50,
    wenc=1,
    lr=config["lr"],
    beta1=0.5,
    beta2=0.999,
)

# from torchsummary import summary

# summary(model.net.to("cuda"), (1, 128, 128))

trainer = pl.Trainer(
    max_epochs=config["num_epochs"],
    logger=[
        pl.loggers.TensorBoardLogger(save_dir="./"),
        WandbLogger(project="GANomaly"),
    ],
    accelerator="gpu",
    devices=1,
    callbacks=model.configure_callbacks(),  # Include your callbacks
)

trainer.fit(model, datamodule=train_data_module)
