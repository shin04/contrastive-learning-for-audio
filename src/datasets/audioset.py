from __future__ import annotations
from pathlib import Path
from enum import Enum

import numpy as np
import pandas as pd
import soundfile as sf
from torch.utils.data import Dataset
import h5py

from datasets.utils import random_crop, mel_spec


class DataType(Enum):
    RAW = 'raw'
    SPECTROGRAM = 'spectrogram'
    LOGMEL = 'logmel'
    MFCC = 'mfcc'


class HDF5Dataset(Dataset):
    """
    Directory Example
    ---------
    ParentDirectory
    |-- waveform
        |-- eval.h5
        |-- balanced_train.h5
        |-- unbalanced_train_part00.h5
    """

    def __init__(self, hdf5_dir: str, crop_sec: int = None, is_training=True):
        self.is_training = is_training

        self.crop_size = int(32000*crop_sec) if crop_sec is not None else None

        hdf5_dir_path = Path(hdf5_dir)
        if not hdf5_dir_path.exists():
            raise RuntimeError(f'{hdf5_dir_path} is not found.')

        mode = 'train' if is_training else 'eval'
        hdf5_path_list = [
            p for p in hdf5_dir_path.glob('*.h5') if mode in str(p)
        ]

        self.data_len = 0
        self.data_pathes = []
        self.data_idxes = []
        for path in hdf5_path_list:
            with h5py.File(path, 'r') as hf:
                audio_names = hf['audio_name']
                data_len = len(audio_names)
                self.data_len += data_len
                self.data_pathes += [path for _ in range(data_len)]
                self.data_idxes += [i for i in range(data_len)]

    def __len__(self):
        return self.data_len

    def __getitem__(self, idx):
        p = self.data_pathes[idx]

        with h5py.File(p, 'r') as hf:
            waveform = hf['waveform'][self.data_idxes[idx]]
            # target = hf['waveform'][self.data_idxes[idx]]

        if self.crop_size is not None:
            waveform, _ = random_crop(waveform, self.crop_size)

        waveform = waveform.reshape((1, -1))

        # return np.float32(waveform)
        return waveform


class AudioSetDataset(Dataset):
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


if __name__ == '__main__':
    dataset = AudioSetDataset(
        metadata_path='/ml/meta/meta_train.csv',
        sr=32000,
        crop_sec=3,
    )
    print(len(dataset))
    print(dataset[0].shape)

    dataset = HDF5Dataset(
        hdf5_dir='/ml/dataset/hdf5/waveform',
        crop_sec=3
    )
    print(len(dataset))
    print(dataset[0][0].shape)
