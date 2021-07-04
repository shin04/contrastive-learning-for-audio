import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio.transforms as audio_nn
from omegaconf import DictConfig

from torchinfo import summary

from models.raw_model import Conv160
from models.spec_model import CNN6
from datasets.utils import random_crop, mel_spec

# https://colab.research.google.com/drive/1UK8BD3xvTpuSj75blz8hThfXksh19YbA?usp=sharing#scrollTo=7JOQBJplT_mw


class Projection(nn.Module):
    def __init__(self, in_channels: int, hidden_dim: int, out_channels: int) -> None:
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

    def forward(self, input: torch.Tensor) -> torch.Tensor:

        x = self.pool1d(input)
        x = self.flatten(x)
        x = self.linear1(x)
        x = self.norm(x)
        x = F.relu(x)
        x = self.linear2(x)

        return F.normalize(x, dim=1)


class CLModel(nn.Module):
    def __init__(self, cfg: DictConfig = None, is_preprocess: bool = False, is_training: bool = True) -> None:
        super(CLModel, self).__init__()

        if (cfg is None) and is_preprocess:
            raise RuntimeError(
                '`cfg` is not None, when `is_preprocess` is True.')

        self.is_preprocess = is_preprocess
        self.is_training = is_training

        if is_preprocess:
            sr = cfg['sr']
            win_size = int(cfg['win_sec']*sr)
            hop_len = int(cfg['hop_sec']*sr)
            n_mels = cfg['n_mels']

            self.mel_spec_trans = audio_nn.MelSpectrogram(
                sample_rate=sr,
                n_fft=win_size,
                win_length=win_size,
                hop_length=hop_len,
                n_mels=n_mels
            )

        self.raw_model = Conv160()
        self.spec_model = CNN6()

        self.projection = Projection(
            in_channels=512, hidden_dim=512, out_channels=128)

    def forward(self, q: torch.Tensor, k: torch.Tensor = None) -> torch.Tensor:
        """
        if runnning preprocess in model, you pass augment only q.
        """

        if self.is_preprocess:
            k = self.mel_spec_trans(q)
            log_offset = 1e-6
            k = torch.log(k + log_offset)

        h_i = self.raw_model(q)
        h_j = self.spec_model(k)

        z_i = self.projection(h_i)
        z_j = self.projection(h_j)

        return z_i, z_j


if __name__ == '__main__':
    model = CLModel(
        cfg={
            'sr': 32000,
            'n_mels': 80,
            'win_sec': 0.2,
            'hop_sec': 0.1,
        },
        is_preprocess=True
    ).cuda()
    summary(model, input_size=(8, 1, 160*1000))
