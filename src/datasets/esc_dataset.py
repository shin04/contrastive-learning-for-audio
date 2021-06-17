from pathlib import Path
from enum import Enum

import numpy as np
import pandas as pd
import soundfile as sf
from torch.utils.data import Dataset

from .utils import random_crop, mel_spec


class DataType(Enum):
    RAW = 'raw'
    SPECTROGRAM = 'spectrogram'
    LOGMEL = 'logmel'
    MFCC = 'mfcc'


class ESCDataset(Dataset):
    def __init__(
        self, audio_path: str, metadata_path: str,
        data_type: DataType = 'raw',
        data_crop_size=3,
        n_mels: int = 80,
        freq_shift_size: int = None
    ):
        self.audio_path = Path(audio_path)

        df = pd.read_csv(Path(metadata_path))
        self.audio_names = [
            self.audio_path / name for name in df['filename'].values.tolist()]

        self.labels = df['target'].values

        self.data_type = data_type
        self.data_crop_size = data_crop_size
        self.n_mels = n_mels
        self.freq_shift_size = freq_shift_size

    def __len__(self):
        return len(self.audio_names)

    def __getitem__(self, idx):
        data_path = self.audio_names[idx]

        wave_data, sr = sf.read(data_path)

        crop_size = self.data_crop_size * sr
        crop_data, _ = random_crop(wave_data, crop_size)

        if self.data_type == 'spectrogram':
            win_size = int(0.2*sr)
            hop_len = int(0.1*sr)
            mel = mel_spec(
                crop_data, sr, win_size, hop_len, self.n_mels, self.freq_shift_size)
            data = mel[np.newaxis, :, :]
        else:
            data = crop_data.reshape((1, -1))

        return np.float32(data), self.labels[idx]


if __name__ == '__main__':
    dataset = ESCDataset(
        audio_path='/ml/dataset/esc/audio',
        metadata_path='/ml/dataset/esc/meta/esc50.csv',
        freq_shift_size=20
    )

    print(len(dataset))
    print(dataset[0])
