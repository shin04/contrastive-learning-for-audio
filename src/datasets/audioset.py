from __future__ import annotations
from pathlib import Path
from enum import Enum

import numpy as np
import pandas as pd
import soundfile as sf
import librosa
from torch.utils.data import Dataset

from utils.audio_checker import get_audio_names


def random_crop(data, crop_size):
    data_length = len(data)

    start = np.random.randint(0, data_length-crop_size)
    end = start+crop_size

    crop_data = data[start:end]

    return crop_data, start


class DataType(Enum):
    RAW = 'raw'
    SPECTROGRAM = 'spectrogram'
    LOGMEL = 'logmel'
    MFCC = 'mfcc'


class CLDataset(Dataset):
    def __init__(
        self, audio_path: str, metadata_path: str = None,
        q_type: DataType = 'raw',
        k_type: DataType = 'raw',
        data_crop_size=3,
    ):
        self.audio_path = Path(audio_path)

        if metadata_path is None:
            self.audio_names = get_audio_names(audio_path, data_crop_size)
        else:
            df = pd.read_csv(Path(metadata_path), header=None)
            self.audio_names = df[0].values.tolist()

        self.q_type = q_type
        self.k_type = k_type
        self.data_crop_size = data_crop_size

    def __len__(self):
        return len(self.audio_names)

    def __getitem__(self, idx):
        data_path = self.audio_names[idx]

        wave_data, sr = sf.read(data_path)

        crop_size = self.data_crop_size * sr
        crop_data, _ = random_crop(wave_data, crop_size)

        q_data = crop_data
        q_data = q_data.reshape((1, -1))

        win_size = int(0.2*sr)
        hop_len = int(0.1*sr)
        mel = librosa.feature.melspectrogram(
            y=crop_data, sr=sr, n_mels=80, n_fft=win_size, win_length=win_size, hop_length=hop_len)
        # log_mel = np.log(mel)

        pos_k_data = mel[np.newaxis, :, :]

        return np.float32(q_data), np.float32(pos_k_data), sr


if __name__ == '__main__':
    dataset = CLDataset(
        audio_path='/ml/dataset/audio/balanced_train_segments',
        metadata_path='../../meta.csv'
    )

    print(len(dataset))
    print(dataset[0])
