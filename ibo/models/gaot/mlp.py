"""ChannelMLP / LinearChannelMLP — verbatim from GAOT
src/model/layers/mlp.py (only the unused Augmented/Conditioned classes and
the omegaconf-touching config plumbing are dropped; [PORT])."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ChannelMLP(nn.Module):
    """ChannelMLP applies an arbitrary number of layers of
    1d convolution and nonlinearity to the channels of input
    and is invariant to spatial resolution. Input [B, C, N]."""

    def __init__(self, in_channels, out_channels=None, hidden_channels=None,
                 n_layers=2, non_linearity=F.gelu, dropout=0.0, **kwargs):
        super().__init__()
        self.n_layers = n_layers
        self.in_channels = in_channels
        self.out_channels = in_channels if out_channels is None else out_channels
        self.hidden_channels = (in_channels if hidden_channels is None
                                else hidden_channels)
        self.non_linearity = non_linearity
        self.dropout = (nn.ModuleList([nn.Dropout(dropout)
                                       for _ in range(n_layers)])
                        if dropout > 0.0 else None)
        self.fcs = nn.ModuleList()
        for i in range(n_layers):
            if i == 0 and i == (n_layers - 1):
                self.fcs.append(nn.Conv1d(self.in_channels, self.out_channels, 1))
            elif i == 0:
                self.fcs.append(nn.Conv1d(self.in_channels, self.hidden_channels, 1))
            elif i == (n_layers - 1):
                self.fcs.append(nn.Conv1d(self.hidden_channels, self.out_channels, 1))
            else:
                self.fcs.append(nn.Conv1d(self.hidden_channels, self.hidden_channels, 1))

    def forward(self, x):
        reshaped = False
        size = list(x.shape)
        if x.ndim > 3:
            x = x.reshape((*size[:2], -1))
            reshaped = True
        for i, fc in enumerate(self.fcs):
            x = fc(x)
            if i < self.n_layers - 1:
                x = self.non_linearity(x)
            if self.dropout is not None:
                x = self.dropout[i](x)
        if reshaped:
            x = x.reshape((size[0], self.out_channels, *size[2:]))
        return x


class LinearChannelMLP(torch.nn.Module):
    """Reimplementation of the ChannelMLP class using Linear instead of
    Conv. Input [..., layers[0]]."""

    def __init__(self, layers, non_linearity=F.gelu, dropout=0.0):
        super().__init__()
        self.n_layers = len(layers) - 1
        assert self.n_layers >= 1
        self.fcs = nn.ModuleList()
        self.non_linearity = non_linearity
        self.dropout = (nn.ModuleList([nn.Dropout(dropout)
                                       for _ in range(self.n_layers)])
                        if dropout > 0.0 else None)
        for j in range(self.n_layers):
            self.fcs.append(nn.Linear(layers[j], layers[j + 1]))

    def forward(self, x):
        for i, fc in enumerate(self.fcs):
            x = fc(x)
            if i < self.n_layers - 1:
                x = self.non_linearity(x)
            if self.dropout is not None:
                x = self.dropout[i](x)
        return x
