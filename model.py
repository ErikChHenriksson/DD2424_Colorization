import torch
from torch.nn import BatchNorm2d, Linear, ReLU, CrossEntropyLoss, Sequential, Conv2d, MaxPool2d, Module, Softmax2d
import torch.nn.functional as F


class Model:
    def __init__(self):

        self.cnn_layers = Sequential(
            # CONV1
            Conv2d(1, 8, kernel_size=3, stride=1, padding=1),
            ReLU(inplace=True),
            Conv2d(8, 8, kernel_size=3, stride=1, padding=1),
            ReLU(inplace=True),
            BatchNorm2d(8),
            # CONV2
            Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
            ReLU(inplace=True),
            Conv2d(16, 16, kernel_size=3, stride=1, padding=1),
            ReLU(inplace=True),
            BatchNorm2d(16),
            # CONV3
            Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            ReLU(inplace=True),
            Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            ReLU(inplace=True),
            BatchNorm2d(32),
            # CONV4
            Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            ReLU(inplace=True),
            Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            ReLU(inplace=True),
            BatchNorm2d(64),
            # CONV5
            Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            ReLU(inplace=True),
            Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            ReLU(inplace=True),
            BatchNorm2d(64),
            # CONV6
            Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            ReLU(inplace=True),
            Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            ReLU(inplace=True),
            BatchNorm2d(64),
            # CONV7
            Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            ReLU(inplace=True),
            Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            ReLU(inplace=True),
            BatchNorm2d(64),
            # CONV8
            Conv2d(64, 32, kernel_size=3, stride=1, padding=1),
            ReLU(inplace=True),
            Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            ReLU(inplace=True),
            BatchNorm2d(32),
        )
