from collections import OrderedDict
from functools import partial
from typing import Tuple, Union

import torch
import torchaudio
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, Dataset

from pitch_tracker.utils import dataset, files
from pitch_tracker.utils.constants import (F_MIN, HOP_LENGTH, N_CLASS, N_FFT,
                                           N_MELS, PICKING_FRAME_SIZE,
                                           PICKING_FRAME_STEP,
                                           PICKING_FRAME_TIME, SAMPLE_RATE,
                                           STEP_FRAME, STEP_TIME, WIN_LENGTH)
from pitch_tracker.utils.dataset import AudioDataset


class Audio_CNN(nn.Module):
    def __init__(self):
        super(Audio_CNN, self).__init__()
        self.conv2d_block1 = create_conv2d_block(
            conv2d_input=(1,256,3),
            maxpool_kernel_size=3,
        )
        
        self.conv2d_block2 = create_conv2d_block(
            conv2d_input=(256,256,3),
            maxpool_kernel_size=3,
        )

        self.conv2d_block3 = create_conv2d_block(
            conv2d_input=(256,210,3),
            maxpool_kernel_size=3,
        )
        
        self.flatten_layer = torch.nn.Flatten(2)
        self.dense_layer = nn.Linear(74, 128)
        self.output_layer = nn.Linear(128, 88)
        
    def forward(self, x):
        x = self.conv2d_block1(x)
        x = self.conv2d_block2(x)
        x = self.conv2d_block3(x)
        x = self.flatten_layer(x)
        x = self.dense_layer(x)
        x = self.output_layer(x)

        return x
    
def create_conv2d_block(
        conv2d_input: Tuple[int,int,Union[Tuple[int,int], int]],
        maxpool_kernel_size: Union[Tuple[int,int], int, None],) -> nn.Sequential:
    """
    Creates a 2D convolutional block with ReLU activation and batch normalization.

    Args:
        conv2d_input (tuple): A tuple containing the number of input channels,
            the number of output channels and the kernel size for the 2D convolutional layer.
            The kernel size can be an integer or a tuple of two integers.
        maxpool_kernel_size (int or tuple or None): The size of the window to take a max over for
            the MaxPool2d layer. Can be an integer or a tuple of two integers. If None,
            no MaxPool2d layer is added to the block.

    Returns:
        nn.Sequential: A sequential container that holds all layers in the block.
    """
    in_channels, out_channels, (kernel_size) = conv2d_input
    
    conv2d = nn.Conv2d(in_channels, out_channels, kernel_size)
    relu = nn.ReLU()
    batch_norm = nn.BatchNorm2d(out_channels)
    maxpool_2d = nn.MaxPool2d(maxpool_kernel_size) if maxpool_kernel_size else None
    
    conv2d_block = nn.Sequential(
        OrderedDict([
            ('conv2d', conv2d),
            ('relu', relu),
            ('batch_norm', batch_norm),  
        ])
    )

    if maxpool_2d:
        conv2d_block.add_module('maxpool2d', maxpool_2d)
    
    return conv2d_block
