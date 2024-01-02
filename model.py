"""
"""
import torch
from torch import nn
import torch.nn.functional as F

from tcn import TemporalConvNet

class TCN(nn.Module):
    def __init__(self,
                 input_size,
                 output_size,
                 num_channels,
                 kernel_size,
                 dropout,
                 dilation_size):

        super().__init__()

        self.tcn = TemporalConvNet(input_size,
                                   num_channels,
                                   kernel_size   = kernel_size,
                                   dropout       = dropout,
                                   dilation_size = dilation_size)

        self.linear = nn.Linear(num_channels[-1], output_size)

    def forward(self, inputs):
        """
        Inputs have to have dimension (N, C_in, L_in)
        # TODO: replace with BCELossWithLogits? Removes sigmoid
        """
        y1 = self.tcn(inputs)
        o = self.linear(y1.permute(0, 2, 1)).squeeze(dim = -1)

        return torch.sigmoid(o)
