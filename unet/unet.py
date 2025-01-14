# -*- coding: utf-8 -*-

"""Main module."""

from typing import Optional
import torch.nn as nn
from .encoding import Encoder, EncodingBlock
from .decoding import Decoder
from .conv import ConvolutionalBlock
from .dimAttBlk import DimensionalAttentionBlock

__all__ = ['UNet', 'ParallelUNet', 'UNet2D', 'UNet3D']

class UNet(nn.Module):
    def __init__(
            self,
            in_channels: int = 1,
            out_classes: int = 2,
            dimensions: int = 2,
            num_encoding_blocks: int = 5,
            out_channels_first_layer: int = 64,
            normalization: Optional[str] = None,
            pooling_type: str = 'max',
            upsampling_type: str = 'conv',
            preactivation: bool = False,
            residual: bool = False,
            padding: int = 0,
            padding_mode: str = 'zeros',
            activation: Optional[str] = 'ReLU',
            initial_dilation: Optional[int] = None,
            dropout: float = 0,
            monte_carlo_dropout: float = 0,
            attention: bool = False,
            att_activation: str = 'Tanh'
            ):
        super().__init__()
        depth = num_encoding_blocks - 1

        # Force padding if residual blocks
        if residual:
            padding = 1

        # Encoder
        self.encoder = Encoder(
            in_channels,
            out_channels_first_layer,
            dimensions,
            pooling_type,
            depth,
            normalization,
            preactivation=preactivation,
            residual=residual,
            padding=padding,
            padding_mode=padding_mode,
            activation=activation,
            initial_dilation=initial_dilation,
            dropout=dropout,
        )

        # Bottom (last encoding block)
        in_channels = self.encoder.out_channels
        if dimensions == 2:
            out_channels_first = 2 * in_channels
        else:
            out_channels_first = in_channels

        self.bottom_block = EncodingBlock(
            in_channels,
            out_channels_first,
            dimensions,
            normalization,
            pooling_type=None,
            preactivation=preactivation,
            residual=residual,
            padding=padding,
            padding_mode=padding_mode,
            activation=activation,
            dilation=self.encoder.dilation,
            dropout=dropout,
        )

        # Decoder
        if dimensions == 2:
            power = depth - 1
        elif dimensions == 3:
            power = depth
        in_channels = self.bottom_block.out_channels
        in_channels_skip_connection = out_channels_first_layer * 2**power
        num_decoding_blocks = depth
        self.decoder = Decoder(
            in_channels_skip_connection,
            dimensions,
            upsampling_type,
            num_decoding_blocks,
            normalization=normalization,
            preactivation=preactivation,
            residual=residual,
            padding=padding,
            padding_mode=padding_mode,
            activation=activation,
            initial_dilation=self.encoder.dilation,
            dropout=dropout,
        )

        # Monte Carlo dropout
        self.monte_carlo_layer = None
        if monte_carlo_dropout:
            dropout_class = getattr(nn, 'Dropout{}d'.format(dimensions))
            self.monte_carlo_layer = dropout_class(p=monte_carlo_dropout)

        # Classifier
        if dimensions == 2:
            in_channels = out_channels_first_layer
        elif dimensions == 3:
            in_channels = 2 * out_channels_first_layer
        self.classifier = ConvolutionalBlock(
            dimensions, in_channels, out_classes,
            kernel_size=1, activation=None,
        )

        # Attention 
        self.attention_blocks = nn.ModuleList()
        if attention:
            if dimensions == 3:
                # set out channels from first layer
                channels = 2 * out_channels_first_layer
                for _ in range(num_encoding_blocks):
                    dab = DimensionalAttentionBlock(channels=channels, compression_ratio=2, activation=att_activation)
                    self.attention_blocks.append(dab)
                    # set out channels according to encoding layer out channels
                    channels = 2 * channels
            else:
                print("Attention is only available for 3D models; no attention modules included in this model.")



    def forward(self, x):
        skip_connections, encoding = self.encoder(x)
        if len(self.attention_blocks) > 0:
            for i, skip in enumerate(skip_connections):
                skip_connections[i] = self.attention_blocks[i](skip)
        encoding = self.bottom_block(encoding)
        x = self.decoder(skip_connections, encoding)
        if self.monte_carlo_layer is not None:
            x = self.monte_carlo_layer(x)
        return self.classifier(x)

class ParallelUNet(nn.Module):
    """
    Adapts previous UNet code but allows for sending encoder and decoder to different GPU devices
    """
    def __init__(
            self,
            in_channels: int = 1,
            out_classes: int = 2,
            dimensions: int = 2,
            num_encoding_blocks: int = 5,
            out_channels_first_layer: int = 64,
            normalization: Optional[str] = None,
            pooling_type: str = 'max',
            upsampling_type: str = 'conv',
            preactivation: bool = False,
            residual: bool = False,
            padding: int = 0,
            padding_mode: str = 'zeros',
            activation: Optional[str] = 'ReLU',
            initial_dilation: Optional[int] = None,
            dropout: float = 0,
            monte_carlo_dropout: float = 0,
            attention: bool = False,
            att_activation: str = 'Tanh',
            dev0: str = None,
            dev1: str = None
            ):
        super().__init__()
        depth = num_encoding_blocks - 1
        self.dev0 = dev0
        self.dev1 = dev1

        # Force padding if residual blocks
        if residual:
            padding = 1

        # Encoder
        self.encoder = Encoder(
            in_channels,
            out_channels_first_layer,
            dimensions,
            pooling_type,
            depth,
            normalization,
            preactivation=preactivation,
            residual=residual,
            padding=padding,
            padding_mode=padding_mode,
            activation=activation,
            initial_dilation=initial_dilation,
            dropout=dropout,
        ).to(dev0)

        # Bottom (last encoding block)
        in_channels = self.encoder.out_channels
        if dimensions == 2:
            out_channels_first = 2 * in_channels
        else:
            out_channels_first = in_channels

        self.bottom_block = EncodingBlock(
            in_channels,
            out_channels_first,
            dimensions,
            normalization,
            pooling_type=None,
            preactivation=preactivation,
            residual=residual,
            padding=padding,
            padding_mode=padding_mode,
            activation=activation,
            dilation=self.encoder.dilation,
            dropout=dropout,
        ).to(dev0)

        # Decoder
        if dimensions == 2:
            power = depth - 1
        elif dimensions == 3:
            power = depth
        in_channels = self.bottom_block.out_channels
        in_channels_skip_connection = out_channels_first_layer * 2**power
        num_decoding_blocks = depth
        self.decoder = Decoder(
            in_channels_skip_connection,
            dimensions,
            upsampling_type,
            num_decoding_blocks,
            normalization=normalization,
            preactivation=preactivation,
            residual=residual,
            padding=padding,
            padding_mode=padding_mode,
            activation=activation,
            initial_dilation=self.encoder.dilation,
            dropout=dropout,
        ).to(dev1)

        # Monte Carlo dropout
        self.monte_carlo_layer = None
        if monte_carlo_dropout:
            dropout_class = getattr(nn, 'Dropout{}d'.format(dimensions))
            self.monte_carlo_layer = dropout_class(p=monte_carlo_dropout)

        # Classifier
        if dimensions == 2:
            in_channels = out_channels_first_layer
        elif dimensions == 3:
            in_channels = 2 * out_channels_first_layer
        self.classifier = ConvolutionalBlock(
            dimensions, in_channels, out_classes,
            kernel_size=1, activation=None,
        ).to(dev1)

        # Attention 
        self.attention_blocks = nn.ModuleList()
        if attention:
            if dimensions == 3:
                # set out channels from first layer
                channels = 2 * out_channels_first_layer
                for _ in range(num_encoding_blocks):
                    dab = DimensionalAttentionBlock(channels=channels, compression_ratio=2, activation=att_activation)
                    self.attention_blocks.append(dab)
                    # set out channels according to encoding layer out channels
                    channels = 2 * channels
            else:
                print("Attention is only available for 3D models; no attention modules included in this model.")



    def forward(self, x):
        # send data to correct GPU here
        x = x.to(self.dev0)
        skip_connections, encoding = self.encoder(x)
        if len(self.attention_blocks) > 0:
            for i, skip in enumerate(skip_connections):
                skip_connections[i] = self.attention_blocks[i](skip)
        encoding = self.bottom_block(encoding)
        for i, skip in enumerate(skip_connections):
            skip_connections[i] = skip.to(self.dev1)
        encoding = encoding.to(self.dev1)
        x = self.decoder(skip_connections, encoding)
        if self.monte_carlo_layer is not None:
            x = self.monte_carlo_layer(x)
        return self.classifier(x)

class UNet2D(UNet):
    def __init__(self, *args, **user_kwargs):
        kwargs = {}
        kwargs['dimensions'] = 2
        kwargs['num_encoding_blocks'] = 5
        kwargs['out_channels_first_layer'] = 64
        kwargs.update(user_kwargs)
        super().__init__(*args, **kwargs)


class UNet3D(UNet):
    def __init__(self, *args, **user_kwargs):
        kwargs = {}
        kwargs['dimensions'] = 3
        kwargs['num_encoding_blocks'] = 4
        kwargs['out_channels_first_layer'] = 32
        kwargs['normalization'] = 'batch'
        kwargs.update(user_kwargs)
        super().__init__(*args, **kwargs)
