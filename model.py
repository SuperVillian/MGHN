import torch
import torch.nn as nn
from dhg.nn import UniGATConv, MultiHeadWrapper

class NonLinearResidualBlock(nn.Module):
    """
    Non-linear Residual Block simulating the residual function in ResNet.
    """
    def __init__(self, channels):
        super(NonLinearResidualBlock, self).__init__()
        self.fc1 = nn.Linear(channels, channels)
        self.bn1 = nn.BatchNorm1d(channels)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(channels, channels)
        self.bn2 = nn.BatchNorm1d(channels)

    def forward(self, x, prev_col):
        out = self.fc1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.bn2(out)
        return out + prev_col  # Non-linear residual connection

class ReversibleScalingLayer(nn.Module):
    """
    Reversible Scaling Layer for channel-level scaling operations.
    """
    def __init__(self, channels):
        super(ReversibleScalingLayer, self).__init__()
        self.scale = nn.Parameter(torch.ones(1, channels))  # Learnable scaling parameter

    def forward(self, x, prev_col):
        return x * self.scale + prev_col  # Channel-level scaling and addition

class MultiHeadAttentionLayer(nn.Module):
    """
    Multi-Head Attention Layer.
    """
    def __init__(self, in_channels: int, hid_channels: int, num_heads: int = 4, use_bn: bool = False,
                 drop_rate: float = 0.5, atten_neg_slope: float = 0.2) -> None:
        super(MultiHeadAttentionLayer, self).__init__()
        self.drop_layer = nn.Dropout(drop_rate)
        self.multi_head_layer = MultiHeadWrapper(
            num_heads,
            "concat",
            UniGATConv,
            in_channels=in_channels,
            out_channels=hid_channels // num_heads,
            use_bn=use_bn,
            drop_rate=drop_rate,
            atten_neg_slope=atten_neg_slope,
        )

    def forward(self, X: torch.Tensor, hg: "dhg.Hypergraph") -> torch.Tensor:
        X = self.drop_layer(X)
        return self.multi_head_layer(X=X, hg=hg)

class AttentionConvBlock(nn.Module):
    """
    Attention Convolutional Block.
    """
    def __init__(self, channels, num_heads):
        super(AttentionConvBlock, self).__init__()
        self.conv1 = MultiHeadAttentionLayer(channels // 2, channels // 2, num_heads=num_heads)
        self.conv2 = MultiHeadAttentionLayer(channels // 2, channels // 2, num_heads=num_heads)
        self.non_linear = NonLinearResidualBlock(channels)
        self.reversible_scale = ReversibleScalingLayer(channels)

    def forward(self, x, hg, prev_col1=None, prev_col2=None):
        if prev_col1 is not None:
            x = self.non_linear(x, prev_col1)

        x1, x2 = torch.chunk(x, 2, dim=1)  # Split input into two halves
        y1 = x1 + self.conv1(x2, hg)  # F(x2)
        y2 = x2 + self.conv2(y1, hg)   # G(y1)

        x = torch.cat((y1, y2), dim=1)

        if prev_col2 is not None:
            x = self.reversible_scale(x, prev_col2)
        
        return x

    def inverse(self, x, hg, prev_col1=None, prev_col2=None):
        if prev_col1 is not None:
            x = self.reversible_scale(x, prev_col1)

        y1, y2 = torch.chunk(x, 2, dim=1)

        # Reverse processing
        x2 = y2 - self.conv2(y1, hg)  # Inverse G(y1)
        x1 = y1 - self.conv1(x2, hg)  # Inverse F(x2)

        x = torch.cat((x1, x2), dim=1)

        if prev_col2 is not None:
            x = self.non_linear(x, prev_col2)
        
        return x

class ColumnBlock(nn.Module):
    """
    Column Block containing multiple attention convolutional blocks.
    """
    def __init__(self, channels, num_heads):
        super(ColumnBlock, self).__init__()
        self.rev_block = nn.Sequential(
            AttentionConvBlock(channels, num_heads),
            AttentionConvBlock(channels, num_heads),
            AttentionConvBlock(channels, num_heads),
            AttentionConvBlock(channels, num_heads)
        )

    def forward(self, x, hg, prev_col=None):
        if prev_col is None:
            for block in self.rev_block:
                x = block(x, hg, None, None)
        else:
            x = self.rev_block[0](x, hg, prev_col[1], prev_col[0])

            for i, block in enumerate(self.rev_block[1:-1]):
                x = block(x, hg, prev_col[i+2], prev_col[i+1])
            
            x = self.rev_block[-1](x, hg, None, prev_col[-1])

        return x

    def inverse(self, x, hg, prev_col=None):
        x_list = []
        
        if prev_col is None:
            for block in reversed(self.rev_block):
                x_list.append(x)
                x = block.inverse(x, hg, None, None)
        else:
            x_list.append(x)
            x = self.rev_block[-1].inverse(x, hg, prev_col[-1], None)
            for i, block in enumerate(reversed(self.rev_block[1:-1])):
                x_list.append(x)
                x = block(x, hg, prev_col[len(prev_col)-i-2], prev_col[len(prev_col)-i-1])
            x_list.append(x)

        return list(reversed(x_list))

class MGHN(nn.Module):
    """
    Multi-Granularity Hypergraph Network (MGHN).
    """
    def __init__(self, num_columns: int, in_channels: int, hid_channels: int, num_classes: int,
                 num_heads: int = 4, use_bn: bool = False, drop_rate: float = 0.5,
                 atten_neg_slope: float = 0.2):
        super(MGHN, self).__init__()
        self.drop_layer = nn.Dropout(drop_rate)
        self.in_layer = UniGATConv(
            in_channels,
            hid_channels,
            use_bn=use_bn,
            drop_rate=drop_rate,
            atten_neg_slope=atten_neg_slope,
            is_last=False,
        )
        self.columns = nn.ModuleList([ColumnBlock(hid_channels, num_heads) for _ in range(num_columns)])
        self.out_layer = UniGATConv(
            hid_channels * num_columns,  # Adjusted for multiple column outputs
            num_classes,
            use_bn=use_bn,
            drop_rate=drop_rate,
            atten_neg_slope=atten_neg_slope,
            is_last=True,
        )

    def forward(self, X, hg):
        X = self.drop_layer(X)
        X = self.in_layer(X, hg)

        prev_output = None
        column_outputs = []
        for column in self.columns:
            column_output = column(X, hg, prev_col=prev_output)
            column_outputs.append(column_output)
            prev_output = column.inverse(column_output, hg, prev_col=prev_output)  # Update prev_output
        
        X = torch.cat(column_outputs, dim=-1)  # Concatenate column outputs
        return self.out_layer(X, hg)  # Final output layer
