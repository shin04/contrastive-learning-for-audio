from __future__ import annotations
from pathlib import Path
from enum import Enum

import numpy as np
import pandas as pd
import soundfile as sf
from torch.utils.data import Dataset

from utils.audio_checker import get_audio_names
from datasets.utils import random_crop, mel_spec


class DataType(Enum):
    RAW = 'raw'
    SPECTROGRAM = 'spectrogram'
    LOGMEL = 'logmel'
    MFCC = 'mfcc'


class AudioSet(Dataset):
    def __init__(self, metadata_path: Path, sr: int = 32000, crop_sec: int = None):
        meta_df = pd.read_csv(metadata_path, header=None)

        self.audio_pathes = meta_df[0].values.tolist()
        self.sr = sr
        if crop_sec is None:
            self.crop_size = None
        else:
            self.crop_size = crop_sec * sr

    def __len__(self):
        return len(self.audio_pathes)

    def __getitem__(self, idx: int):
        audio_path = self.audio_pathes[idx]

        wave_data, _ = sf.read(audio_path)

        if self.crop_size is not None:
            wave_data, _ = random_crop(wave_data, self.crop_size)

        wave_data = wave_data.reshape((1, -1))

        return np.float32(wave_data)


class CLDataset(Dataset):
    def __init__(
        self, audio_path: str, metadata_path: str = None,
        q_type: DataType = 'raw',
        k_type: DataType = 'raw',
        crop_sec=3,
        n_mels: int = 80,
        freq_shift_size: int = None
    ):
        self.audio_path = Path(audio_path)

        if metadata_path is None:
            self.audio_names = get_audio_names(audio_path, crop_sec)
        else:
            df = pd.read_csv(Path(metadata_path), header=None)
            self.audio_names = df[0].values.tolist()

        # df = pd.read_csv(Path(metadata_path), header=None)
        # self.audio_names = df[0].values.tolist()

        self.q_type = q_type
        self.k_type = k_type
        self.crop_sec = crop_sec
        self.n_mels = n_mels
        self.freq_shift_size = freq_shift_size

    def __len__(self):
        return len(self.audio_names)

    def __getitem__(self, idx):
        data_path = self.audio_names[idx]

        wave_data, sr = sf.read(data_path)

        crop_size = self.crop_sec * sr
        crop_data, _ = random_crop(wave_data, crop_size)

        q_data = crop_data
        q_data = q_data.reshape((1, -1))

        win_size = int(0.2*sr)
        hop_len = int(0.1*sr)
        mel = mel_spec(
            crop_data, sr, win_size, hop_len, self.n_mels, self.freq_shift_size)

        pos_k_data = mel[np.newaxis, :, :]

        return np.float32(q_data), np.float32(pos_k_data), sr


if __name__ == '__main__':
    dataset = CLDataset(
        audio_path='/ml/dataset/audioset/audio/balanced_train_segments',
        metadata_path='/ml/meta_train.csv',
        freq_shift_size=20
    )

    print(len(dataset))
    print(dataset[0][0].shape)

    dataset = AudioSet(
        metadata_path='/ml/meta_train.csv',
        sr=32000,
        crop_sec=3,
    )

    print(len(dataset))
    print(dataset[0].shape)
