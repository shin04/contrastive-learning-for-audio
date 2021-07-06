from pathlib import Path
from enum import Enum

import pandas as pd
import soundfile as sf
import torch
from torch.utils.data import Dataset

from .utils import random_crop


class DataType(Enum):
    RAW = 'raw'
    SPECTROGRAM = 'spectrogram'
    LOGMEL = 'logmel'
    MFCC = 'mfcc'


class ESCDataset(Dataset):
    def __init__(
        self, audio_path: str, metadata_path: str,
        folds: int = None,
        sr: int = 32000,
        crop_sec: int = None
    ) -> None:
        self.audio_path = Path(audio_path)

        df = pd.read_csv(Path(metadata_path))
        self.waveforms = []
        self.labels = []
        for i in range(len(df)):
            df_fold = df.loc[i]['fold']
            p = self.audio_path / df.loc[i]['filename']

            label = df.loc[i]['target']

            if (folds is None) or (df_fold in folds):
                waveform, _ = sf.read(p)
                self.waveforms.append(waveform)
                self.labels.append(label)
            else:
                continue

        self.sr = sr
        if crop_sec is None:
            self.crop_size = None
        else:
            self.crop_size = crop_sec * sr

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        wave_data = self.waveforms[idx]

        if self.crop_size is not None:
            wave_data, _ = random_crop(wave_data, self.crop_size)

        wave_data = wave_data.reshape((1, -1))

        return torch.from_numpy(wave_data).float(), self.labels[idx]


if __name__ == '__main__':
    dataset = ESCDataset(
        audio_path='/ml/dataset/esc/audio',
        metadata_path='/ml/dataset/esc/meta/esc50.csv',
        folds=[1],
        sr=32000,
        crop_sec=3
    )

    print(len(dataset))
    print(dataset[0])
