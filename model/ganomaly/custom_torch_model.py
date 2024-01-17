"""Torch models defining encoder, decoder, Generator and Discriminator.

Custom implementation to attempt to get shit working for an MRI usecase
"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn

from model.ganomaly.utils import pad_nextpow2


class Encoder(nn.Module):
    """Encoder Network.

    Args:
        input_size (tuple[int, int]): Size of input image
        latent_vec_size (int): Size of latent vector z
        num_input_channels (int): Number of input channels in the image
        n_features (int): Number of features per convolution layer
        extra_layers (int): Number of extra layers since the network uses only a single encoder layer by default.
            Defaults to 0.
    """

    def __init__(
        self,
        input_size: tuple[int, int],
        latent_vec_size: int,
        num_input_channels: int,
        n_features: int,
    ) -> None:
        super().__init__()

        self.layers = nn.Sequential(
            nn.Conv2d(num_input_channels, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(
                self.calculate_conv_output_size(num_input_channels, (64, 64)),
                latent_vec_size,
            ),
        )

    def conv_block_stride(self, channels_in, channels_out):
        block = nn.Sequential(
            nn.Conv2d(channels_in, channels_out, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(channels_out),
            nn.ReLU(),
        )
        return block

    def conv_block_maxpool(self, channels_in, channels_out):
        pass

    def calculate_conv_output_size(self, input_channels, input_size):
        # Function to calculate the size of the linear layer input after convolutional layers
        with torch.no_grad():
            x = torch.zeros((1, input_channels, *input_size))
            conv_output = self.layers[:5](x)  # why :5 !?
            return conv_output.view(1, -1).size(1)

    def forward(self, input_tensor: Tensor) -> Tensor:
        """Return latent vectors."""

        output = self.layers(input_tensor)

        return output


class Decoder(nn.Module):
    """Decoder Network.

    Args:
        input_size (tuple[int, int]): Size of input image
        latent_vec_size (int): Size of latent vector z
        num_input_channels (int): Number of input channels in the image
        n_features (int): Number of features per convolution layer
        extra_layers (int): Number of extra layers since the network uses only a single encoder layer by default.
            Defaults to 0.
    """

    def __init__(
        self,
        latent_vec_size: int,
    ) -> None:
        super().__init__()

        self.layers = nn.Sequential(
            nn.Linear(latent_vec_size, self.calculate_conv_input_size((64, 64))),
            nn.BatchNorm1d(self.calculate_conv_input_size((64, 64))),
            nn.ReLU(),
            nn.Unflatten(1, (128, 4, 4)),
        )

    def calculate_conv_input_size(self):
        # Function to calculate the size of the linear layer input before transposed convolutions
        with torch.no_grad():
            x = torch.zeros((1, 128, 4, 4))
            conv_input = self.decoder_layers[:2](x.view(1, -1))
            return conv_input.size(1)

    def forward(self, input_tensor: Tensor) -> Tensor:
        """Return generated image."""

        output = self.layers(input_tensor)
        # scale to [-1,1]
        output = nn.Tanh(output)
        return output


class Discriminator(nn.Module):
    """Discriminator.

        Made of only one encoder layer which takes x and x_hat to produce a score.

    Args:
        input_size (tuple[int, int]): Input image size.
        num_input_channels (int): Number of image channels.
        n_features (int): Number of feature maps in each convolution layer.
        extra_layers (int, optional): Add extra intermediate layers. Defaults to 0.
    """

    def __init__(
        self,
        input_size: tuple[int, int],
        num_input_channels: int,
    ) -> None:
        super().__init__()
        encoder = Encoder(
            input_size=input_size,
            latent_vec_size=1,
            num_input_channels=num_input_channels,
        )
        layers = []
        for block in encoder.children():
            if isinstance(block, nn.Sequential):
                layers.extend(list(block.children()))
            else:
                layers.append(block)

        self.features = nn.Sequential(*layers[:-1])
        self.classifier = nn.Sequential(layers[-1])
        self.classifier.add_module("Sigmoid", nn.Sigmoid())

    def forward(self, input_tensor: Tensor) -> tuple[Tensor, Tensor]:
        """Return class of object and features."""
        features = self.features(input_tensor)
        classifier = self.classifier(features)
        classifier = classifier.view(-1, 1).squeeze(1)
        return classifier, features


class Generator(nn.Module):
    """Generator model.

    Made of an encoder-decoder-encoder architecture.

    Args:
        input_size (tuple[int, int]): Size of input data.
        latent_vec_size (int): Dimension of latent vector produced between the first encoder-decoder.
        num_input_channels (int): Number of channels in input image.
        n_features (int): Number of feature maps in each convolution layer.
        extra_layers (int, optional): Extra intermediate layers in the encoder/decoder. Defaults to 0.
        add_final_conv_layer (bool, optional): Add a final convolution layer in the decoder. Defaults to True.
    """

    def __init__(
        self,
        input_size: tuple[int, int],
        latent_vec_size: int,
        num_input_channels: int,
    ) -> None:
        super().__init__()
        self.encoder1 = Encoder(
            input_size,
            latent_vec_size,
            num_input_channels,
        )
        self.decoder = Decoder(
            input_size,
            latent_vec_size,
            num_input_channels,
        )
        self.encoder2 = Encoder(
            input_size,
            latent_vec_size,
            num_input_channels,
        )

    def forward(self, input_tensor: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Return generated image and the latent vectors."""
        latent_i = self.encoder1(input_tensor)
        gen_image = self.decoder(latent_i)
        latent_o = self.encoder2(gen_image)
        return gen_image, latent_i, latent_o


class GanomalyModel(nn.Module):
    """Ganomaly Model.

    Args:
        input_size (tuple[int, int]): Input dimension.
        num_input_channels (int): Number of input channels.
        n_features (int): Number of features layers in the CNNs.
        latent_vec_size (int): Size of autoencoder latent vector.
        extra_layers (int, optional): Number of extra layers for encoder/decoder. Defaults to 0.
        add_final_conv_layer (bool, optional): Add convolution layer at the end. Defaults to True.
    """

    def __init__(
        self,
        input_size: tuple[int, int],
        num_input_channels: int,
        latent_vec_size: int,
    ) -> None:
        super().__init__()
        self.generator: Generator = Generator(
            input_size=input_size,
            latent_vec_size=latent_vec_size,
            num_input_channels=num_input_channels,
        )
        self.discriminator: Discriminator = Discriminator(
            input_size=input_size,
            num_input_channels=num_input_channels,
        )

    def forward(self, batch: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor] | Tensor:
        """Get scores for batch.

        Args:
            batch (Tensor): Images

        Returns:
            idk bro
        """
        padded_batch = pad_nextpow2(batch)
        fake, latent_i, latent_o = self.generator(padded_batch)

        return padded_batch, fake, latent_i, latent_o
        # if self.training:
        #     return padded_batch, fake, latent_i, latent_o
        # # convert nx1x1 to n
        # return torch.mean(torch.pow((latent_i - latent_o), 2), dim=1).view(-1)
