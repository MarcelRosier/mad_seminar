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
        channels_start: int = 32,
    ) -> None:
        super().__init__()

        self.layers = nn.Sequential(
            nn.Conv2d(1, channels_start, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(channels_start),
            nn.ReLU(),
            nn.Conv2d(
                channels_start, channels_start * 2, kernel_size=3, stride=2, padding=1
            ),
            nn.BatchNorm2d(channels_start * 2),
            nn.ReLU(),
            nn.Conv2d(
                channels_start * 2,
                channels_start * 4,
                kernel_size=3,
                stride=2,
                padding=1,
            ),
            nn.BatchNorm2d(channels_start * 4),
            nn.ReLU(),
        )

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(
                self.calculate_conv_output_size(input_size),
                latent_vec_size,
            ),
        )

    # def conv_block_stride(self, channels_in, channels_out):
    #     block = nn.Sequential(
    #         nn.Conv2d(channels_in, channels_out, kernel_size=3, stride=2, padding=1),
    #         nn.BatchNorm2d(channels_out),
    #         nn.ReLU(),
    #     )
    #     return block

    # def conv_block_maxpool(self, channels_in, channels_out):
    #     pass

    def calculate_conv_output_size(self, input_size):
        # Function to calculate the size of the linear layer input after convolutional layers
        with torch.no_grad():
            x = torch.zeros((1, 1, *input_size))
            conv_output = self.layers(x)  # why :5 !?
            self.conv_output_shape = conv_output.shape[1:]
            return conv_output.view(1, -1).size(1)

    def forward(self, input_tensor: Tensor) -> Tensor:
        """Return latent vectors."""

        output = self.layers(input_tensor)
        latent = self.fc(output)

        return latent


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
        self, latent_vec_size: int, conv_input_shape: int, channels_start: int = 32
    ) -> None:
        super().__init__()

        conv_input_size = (
            conv_input_shape[0] * conv_input_shape[1] * conv_input_shape[2]
        )
        self.layers = nn.Sequential(
            nn.Linear(latent_vec_size, conv_input_size),
            nn.BatchNorm1d(conv_input_size),
            nn.ReLU(),
            nn.Unflatten(1, conv_input_shape),
        )

        self.convs = nn.Sequential(
            nn.Sequential(
                nn.ConvTranspose2d(
                    channels_start * 4,
                    channels_start * 2,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    output_padding=1,
                ),
                nn.BatchNorm2d(channels_start * 2),
                nn.ReLU(),
                nn.ConvTranspose2d(
                    channels_start * 2,
                    channels_start,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    output_padding=1,
                ),
                nn.BatchNorm2d(channels_start),
                nn.ReLU(),
                nn.ConvTranspose2d(
                    channels_start,
                    1,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    output_padding=1,
                ),
            )
        )

        self.tanh = nn.Tanh()

    def forward(self, input_tensor: Tensor) -> Tensor:
        """Return generated image."""

        output = self.layers(input_tensor)
        output = self.convs(output)
        # scale to [-1,1]
        output = self.tanh(output)
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
        channels_start: int = 32,
    ) -> None:
        super().__init__()
        encoder = Encoder(
            input_size=input_size, latent_vec_size=1, channels_start=channels_start
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
        channels_start: int = 32,
    ) -> None:
        super().__init__()

        self.encoder1 = Encoder(
            input_size=input_size,
            latent_vec_size=latent_vec_size,
            channels_start=channels_start,
        )
        self.decoder = Decoder(
            latent_vec_size=latent_vec_size,
            conv_input_shape=self.encoder1.conv_output_shape,
            channels_start=channels_start,
        )
        self.encoder2 = Encoder(
            input_size=input_size,
            latent_vec_size=latent_vec_size,
            channels_start=channels_start,
        )

    def forward(self, input_tensor: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Return generated image and the latent vectors."""
        latent_i = self.encoder1(input_tensor)
        gen_image = self.decoder(latent_i)
        latent_o = self.encoder2(gen_image)
        return gen_image, latent_i, latent_o


class CustomGanomalyModel(nn.Module):
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
        latent_vec_size: int,
        channels_start: int,
    ) -> None:
        super().__init__()
        self.generator: Generator = Generator(
            input_size=input_size,
            latent_vec_size=latent_vec_size,
            channels_start=channels_start,
        )
        self.discriminator: Discriminator = Discriminator(
            input_size=input_size,
            channels_start=channels_start,
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
