import torch
import torch.nn as nn
import torch.nn.functional as F

from torchinfo import summary


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int, padding: int) -> None:
        super(ConvBlock, self).__init__()

        self.conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=False
        )
        self.group_norm = nn.GroupNorm(out_channels//2, out_channels)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        x = self.conv(input)
        x = self.group_norm(x)
        x = F.relu(x)

        output = F.avg_pool1d(x, kernel_size=2, padding=1)
        output = x

        return output


class Conv160(nn.Module):
    def __init__(self, is_training: bool = True) -> None:
        super(Conv160, self).__init__()

        self.is_training = is_training

        self.conv1 = ConvBlock(
            in_channels=1, out_channels=512,
            kernel_size=10, stride=5, padding=3
        )

        self.conv2 = ConvBlock(
            in_channels=512, out_channels=512,
            kernel_size=4, stride=2, padding=1
        )

        self.conv3 = ConvBlock(
            in_channels=512, out_channels=512,
            kernel_size=4, stride=2, padding=1
        )

        self.conv4 = ConvBlock(
            in_channels=512, out_channels=512,
            kernel_size=4, stride=2, padding=1
        )

        self.conv5 = ConvBlock(
            in_channels=512, out_channels=512,
            kernel_size=4, stride=2, padding=1
        )

        self.conv6 = ConvBlock(
            in_channels=512, out_channels=512,
            kernel_size=4, stride=2, padding=1
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        x = self.conv1(input)
        x = F.dropout(x, p=0.2, training=self.is_training)

        x = self.conv2(x)
        x = F.dropout(x, p=0.2, training=self.is_training)

        x = self.conv3(x)
        x = F.dropout(x, p=0.2, training=self.is_training)

        x = self.conv4(x)
        x = F.dropout(x, p=0.2, training=self.is_training)

        x = self.conv5(x)
        x = F.dropout(x, p=0.2, training=self.is_training)

        x = self.conv6(x)
        x = F.dropout(x, p=0.2, training=self.is_training)
        output = x

        return output


if __name__ == '__main__':
    model = Conv160().cuda()
    summary(model, input_size=(32, 1, 160*1000))
