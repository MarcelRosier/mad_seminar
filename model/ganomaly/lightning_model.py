from __future__ import annotations
from typing import Dict

"""GANomaly: Semi-Supervised Anomaly Detection via Adversarial Training.

https://arxiv.org/abs/1805.06725
"""

# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


import logging

import torch
from pytorch_lightning.callbacks import Callback, EarlyStopping
from pytorch_lightning.utilities.types import STEP_OUTPUT  # ,EPOCH_OUTPUT
from torch import Tensor, optim

import pytorch_lightning as pl

from model.ganomaly.loss import DiscriminatorLoss, GeneratorLoss

from model.ganomaly.torch_model import GanomalyModel
from model.ganomaly.custom_torch_model import CustomGanomalyModel

logger = logging.getLogger(__name__)

EPOCH_OUTPUT = Dict  # prbly incorrect


class Ganomaly(pl.LightningModule):
    """PL Lightning Module for the GANomaly Algorithm.

    Args:
        batch_size (int): Batch size.
        input_size (tuple[int, int]): Input dimension.
        n_features (int): Number of features layers in the CNNs.
        latent_vec_size (int): Size of autoencoder latent vector.
        extra_layers (int, optional): Number of extra layers for encoder/decoder. Defaults to 0.
        add_final_conv_layer (bool, optional): Add convolution layer at the end. Defaults to True.
        wadv (int, optional): Weight for adversarial loss. Defaults to 1.
        wcon (int, optional): Image regeneration weight. Defaults to 50.
        wenc (int, optional): Latent vector encoder weight. Defaults to 1.
    """

    def __init__(
        self,
        batch_size: int = 32,
        input_size: tuple[int, int] = (128, 128),
        n_features: int = 32,
        latent_vec_size: int = 100,
        extra_layers: int = 0,
        add_final_conv_layer: bool = True,
        wadv: int = 1,
        wcon: int = 50,
        wenc: int = 1,
        lr: float = 0.0002,
        beta1: float = 0.5,
        beta2: float = 0.999,
        kernel_size: int = 4,
    ) -> None:
        super().__init__()

        # self.net = CustomGanomalyModel(
        #     input_size=input_size, latent_vec_size=latent_vec_size, channels_start=64
        # )
        self.net: GanomalyModel = GanomalyModel(
            input_size=input_size,
            num_input_channels=1,
            n_features=n_features,
            latent_vec_size=latent_vec_size,
            extra_layers=extra_layers,
            add_final_conv_layer=add_final_conv_layer,
            kernel_size=kernel_size,
        )
        self.real_label = torch.ones(size=(batch_size,), dtype=torch.float32)
        self.fake_label = torch.zeros(size=(batch_size,), dtype=torch.float32)

        self.min_scores: Tensor = torch.tensor(
            float("inf"), dtype=torch.float32
        )  # pylint: disable=not-callable
        self.max_scores: Tensor = torch.tensor(
            float("-inf"), dtype=torch.float32
        )  # pylint: disable=not-callable

        self.generator_loss = GeneratorLoss(wadv, wcon, wenc, self.log)
        self.discriminator_loss = DiscriminatorLoss()

        # TODO: LR should be part of optimizer in config.yaml! Since ganomaly has custom
        #   optimizer this is to be addressed later.
        self.learning_rate = lr
        self.beta1 = beta1
        self.beta2 = beta2

        # self.automatic_optimization = False  # Turn off automatic optimization

    def configure_optimizers(self) -> list[optim.Optimizer]:
        """Configures optimizers for each decoder.

        Note:
            This method is used for the existing CLI.
            When PL CLI is introduced, configure optimizers method will be
                deprecated, and optimizers will be configured from either
                config.yaml file or from CLI.

        Returns:
            Optimizer: Adam optimizer for each decoder
        """
        optimizer_d = optim.Adam(
            self.net.discriminator.parameters(),
            lr=self.learning_rate,
            betas=(self.beta1, self.beta2),
        )
        optimizer_g = optim.Adam(
            self.net.generator.parameters(),
            lr=self.learning_rate,
            betas=(self.beta1, self.beta2),
        )
        scheduler_d = torch.optim.lr_scheduler.ExponentialLR(
            optimizer_d, gamma=0.995, verbose=True
        )
        scheduler_g = torch.optim.lr_scheduler.ExponentialLR(
            optimizer_g, gamma=0.995, verbose=True
        )

        return [optimizer_d, optimizer_g], [scheduler_d, scheduler_g]

    def training_step(
        self, batch: dict[str, str | Tensor], batch_idx: int, optimizer_idx: int
    ) -> STEP_OUTPUT:  # pylint: disable=arguments-differ
        """Training step.

        Args:
            batch (dict[str, str | Tensor]): Input batch containing images.
            batch_idx (int): Batch index.
            optimizer_idx (int): Optimizer which is being called for current training step.

        Returns:
            STEP_OUTPUT: Loss
        """
        del batch_idx  # `batch_idx` variables is not used.

        # forward pass
        padded, fake, latent_i, latent_o = self.net(batch)
        pred_real, _ = self.net.discriminator(padded)

        if optimizer_idx == 0:  # Discriminator
            pred_fake, _ = self.net.discriminator(fake.detach())
            loss = self.discriminator_loss(pred_real, pred_fake)
        else:  # Generator
            scores = torch.mean(torch.pow((latent_i - latent_o), 2), dim=1).view(-1)
            self.log(
                "train_anomaly_score_G",
                scores.mean().item(),
                on_step=True,
                on_epoch=False,
            )
            pred_fake, _ = self.net.discriminator(fake)
            loss = self.generator_loss(
                latent_i, latent_o, padded, fake, pred_real, pred_fake
            )

        loss_name = "D_loss" if optimizer_idx == 0 else "G_loss"
        self.log(loss_name, loss.item(), on_epoch=True, prog_bar=True, logger=True)
        return {"loss": loss}

    def _reset_min_max(self) -> None:
        """Resets min_max scores."""
        self.min_scores = torch.tensor(
            float("inf"), dtype=torch.float32
        )  # pylint: disable=not-callable
        self.max_scores = torch.tensor(
            float("-inf"), dtype=torch.float32
        )  # pylint: disable=not-callable

    def validation_step(
        self, batch: dict[str, str | Tensor], *args, **kwargs
    ) -> STEP_OUTPUT:
        """Update min and max scores from the current step.

        Args:
            batch (dict[str, str | Tensor]): Predicted difference between z and z_hat.

        Returns:
            (STEP_OUTPUT): Output predictions.
        """
        _, fake, latent_i, latent_o = self.net(batch)
        scores = torch.mean(torch.pow((latent_i - latent_o), 2), dim=1).view(-1)
        self.max_scores = max(self.max_scores, torch.max(scores))
        self.min_scores = min(self.min_scores, torch.min(scores))

        self.log(
            "val_anomaly_score", scores.mean().item(), on_step=True, on_epoch=False
        )

        # return batch
        return {"scores": scores}

    def _normalize(self, scores: Tensor) -> Tensor:
        """Normalize the scores based on min/max of entire dataset.

        Args:
            scores (Tensor): Un-normalized scores.

        Returns:
            Tensor: Normalized scores.
        """
        scores = (scores - self.min_scores.to(scores.device)) / (
            self.max_scores.to(scores.device) - self.min_scores.to(scores.device)
        )
        return scores

    def detect_anomaly(self, x: Tensor):
        _, fake, latent_i, latent_o = self.net(x)
        anomaly_score = torch.mean(torch.pow((latent_i - latent_o), 2), dim=1).view(-1)
        return {
            "reconstruction": fake,
            "anomaly_map": torch.abs(x - fake),
            "latent_i": latent_i,
            "latent_o": latent_o,
            "anomaly_score": anomaly_score,
        }

    # def configure_callbacks(self) -> list[Callback]:
    #     """Configure model-specific callbacks."""
    #     early_stopping = EarlyStopping(
    #         monitor="val_anomaly_score",  # Change this to the metric you want to monitor
    #         patience=10,  # Number of epochs with no improvement after which training will be stopped
    #         mode="min",  # "min" if the monitored quantity should be minimized, "max" otherwise
    #     )
    #     return [early_stopping]
