"""
No worries, I will use BCELossWithLogits and yes,
I will remove sigmoid for you :)
"""

from torch import nn

from tcn import TemporalConvNet


class TCN(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels_list,
                 kernel_size,
                 dilation_size,
                 dropout):

        super().__init__()

        self.tcn = TemporalConvNet(in_channels,
                                   out_channels_list,
                                   kernel_size   = kernel_size,
                                   dilation_size = dilation_size,
                                   dropout       = dropout)

        self.lin = nn.Linear(out_channels_list[-1], 1)

    def forward(self, data):
        """
        Input has dimension (batch, num_channels, seq_len)
        Need to move channels dimension to the last for the linear map.
        """
        data = self.tcn(data)
        return self.lin(data.permute(0, 2, 1)).squeeze(-1)
