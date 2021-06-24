# import torch
import torch.nn as nn
import torch.nn.functional as F
# from torch.utils.tensorboard import SummaryWriter

from torchinfo import summary

from models.raw_model import Conv160
from models.spec_model import CNN6

# from raw_model import Conv160
# from spec_model import CNN6

# https://colab.research.google.com/drive/1UK8BD3xvTpuSj75blz8hThfXksh19YbA?usp=sharing#scrollTo=7JOQBJplT_mw


class Projection(nn.Module):
    def __init__(self, in_channels, hidden_dim, out_channels):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_dim = hidden_dim
        self.out_channels = out_channels

        self.pool1d = nn.AdaptiveAvgPool1d(1)
        self.flatten = nn.Flatten()
        self.linear1 = nn.Linear(self.in_channels, self.hidden_dim, bias=True)
        self.norm = nn.BatchNorm1d(self.hidden_dim)
        self.linear2 = nn.Linear(
            self.hidden_dim, self.out_channels, bias=False)

    def forward(self, input):

        x = self.pool1d(input)
        x = self.flatten(x)
        x = self.linear1(x)
        x = self.norm(x)
        x = F.relu(x)
        x = self.linear2(x)

        return F.normalize(x, dim=1)


class CLModel(nn.Module):
    def __init__(self, is_training=True):
        super(CLModel, self).__init__()

        self.is_training = is_training

        self.raw_model = Conv160()
        self.spec_model = CNN6()

        self.projection = Projection(
            in_channels=512, hidden_dim=512, out_channels=128)

    def forward(self, q, k):
        h_i = self.raw_model(q)
        h_j = self.spec_model(k)

        z_i = self.projection(h_i)
        z_j = self.projection(h_j)

        return z_i, z_j


if __name__ == '__main__':
    model = CLModel()
    summary(model, input_size=[(32, 1, 160*1000), (32, 1, 64, 1000)])

    # writer = SummaryWriter()

    # fake_wave = torch.randn(1, 1, 160*1000)
    # fake_mel = torch.randn(1, 1, 64, 1000)
    # model = CLModel()
    # writer.add_graph(model, [fake_wave, fake_mel])
    # writer.close()
