import torch
import torch.nn as nn
import torch.nn.functional as F

from torchinfo import summary


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> torch.Tensor:
        super(ConvBlock, self).__init__()

        self.conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=(5, 5),
            stride=(1, 1),
            padding=(2, 2),
            bias=False
        )
        self.batch_norm = nn.BatchNorm2d(out_channels)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        x = self.conv(input)
        x = self.batch_norm(x)
        x = F.relu(x)
        output = F.avg_pool2d(x, kernel_size=(2, 2))

        return output


class CNN6(nn.Module):
    def __init__(self, is_training: bool = True) -> None:
        super(CNN6, self).__init__()

        self.is_training = is_training

        self.conv1 = ConvBlock(in_channels=1, out_channels=64)
        self.conv2 = ConvBlock(in_channels=64, out_channels=128)
        self.conv3 = ConvBlock(in_channels=128, out_channels=256)
        self.conv4 = ConvBlock(in_channels=256, out_channels=512)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        x = self.conv1(input)
        x = F.dropout(x, p=0.2, training=self.is_training)

        x = self.conv2(x)
        x = F.dropout(x, p=0.2, training=self.is_training)

        x = self.conv3(x)
        x = F.dropout(x, p=0.2, training=self.is_training)

        x = self.conv4(x)
        x = F.dropout(x, p=0.2, training=self.is_training)
        output = torch.mean(x, dim=3)

        return output


if __name__ == '__main__':
    model = CNN6().cuda()
    summary(model, input_size=(32, 1, 64, 1000))
