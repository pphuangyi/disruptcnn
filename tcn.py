# !/usr/bin python
#
# MIT License
#
# Copyright (c) 2018 CMU Locus Lab
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
#
# Temporal Convolutional Network code, from the original repo
# (https://github.com/locuslab/TCN)
# @article{BaiTCN2018,
# 	author    = {Shaojie Bai and J. Zico Kolter and Vladlen Koltun},
# 	title     = {An Empirical Evaluation of Generic Convolutional and
#                 Recurrent Networks for Sequence Modeling},
# 	journal   = {arXiv:1803.01271},
# 	year      = {2018},
# }
#
# modified slightly here for arbitrary dilation factors.

from torch import nn
from torch.nn.utils import weight_norm
from torch.nn.functional import relu


class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[..., :-self.chomp_size].contiguous()


class TemporalResidualBlock(nn.Module):
    """
    Convolution that produce sequence of same length
    with stride = 1, we have
    seq_len_out = seq_len_in + 2 * padding - dilation * (kernel_size - 1).
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 dilation,
                 dropout = 0.2):

        super().__init__()

        padding = dilation * (kernel_size - 1)

        self.conv1 = weight_norm(nn.Conv1d(in_channels,
                                           out_channels,
                                           kernel_size,
                                           padding  = padding,
                                           dilation = dilation))
        self.conv2 = weight_norm(nn.Conv1d(out_channels,
                                           out_channels,
                                           kernel_size,
                                           padding  = padding,
                                           dilation = dilation))

        self.block = nn.Sequential(self.conv1,
                                   Chomp1d(padding),
                                   nn.ReLU(),
                                   nn.Dropout(dropout),
                                   self.conv2,
                                   Chomp1d(padding),
                                   nn.ReLU(), # should we have relu here?
                                   nn.Dropout(dropout))


        if in_channels != out_channels:
            self.downsample = nn.Conv1d(in_channels, out_channels, 1)
        else:
            self.downsample = nn.Identity()

        self.in_channels  = in_channels
        self.out_channels = out_channels

        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)

        if self.in_channels != self.out_channels:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, data):
        out = self.block(data)
        res = self.downsample(data)
        return relu(out + res)


class TemporalConvNet(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels_list,
                 kernel_size   = 2,
                 dilation_size = 2,
                 dropout       = 0.2):

        super().__init__()

        layers = []
        num_levels = len(out_channels_list)

        if isinstance(dilation_size, int):
            dilation_size = [dilation_size ** i for i in range(num_levels)]

        in_ch = in_channels
        for i in range(num_levels):
            out_ch = out_channels_list[i]
            layers += [TemporalResidualBlock(in_ch,
                                             out_ch,
                                             kernel_size,
                                             dilation = dilation_size[i],
                                             dropout  = dropout)]
            in_ch = out_ch

        self.net = nn.Sequential(*layers)

    def forward(self, data):
        return self.net(data)
