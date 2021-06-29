from pathlib import Path
import typing

import numpy as np
import pandas as pd
import librosa
from tqdm import tqdm

from datasets.utils import random_crop, mel_spec


def devide_metadata(metadata: np.ndarray, batch_size: int = 32) -> np.ndarray:
    data_len = len(metadata)
    last_batch_size = data_len % batch_size
    last_batch_meta = metadata[data_len-last_batch_size:]

    res_metadata = metadata[:data_len-last_batch_size]
    batch_meta = []
    for i in range(data_len // batch_size):
        batch_meta.append(res_metadata[i*batch_size:(i+1)*batch_size])
    batch_meta.append(last_batch_meta)

    return np.array(batch_meta)


def load_data(
    metadata: np.ndarray, sr: int = 22050, crop_sec: int = 3
) -> np.ndarray:

    crop_size = crop_sec*sr
    wave_data = np.empty((len(metadata), crop_size))

    for i, p in enumerate(tqdm(metadata)):
        y, _ = librosa.load(p, sr=sr)
        crop_data, _ = random_crop(y, crop_size)
        wave_data[i] = crop_data.reshape((1, -1))

    return wave_data


def augmentation(
    data: np.ndarray, sr: int = 22050,
    aug_func: typing.Callable = mel_spec,
    win_sec: float = 0.2, hop_sec: float = 0.1,
    n_mels: int = 80, freq_shift_size: int = None
) -> np.ndarray:
    win_size = int(win_sec*sr)
    hop_len = int(hop_sec*sr)

    auged_data = []
    for d in tqdm(data):
        auged = aug_func(d, sr, win_size, hop_len, n_mels, freq_shift_size)
        auged_data.append(auged)
    auged_data = np.array(auged_data)

    return auged_data[np.newaxis, :, :]


def run():
    meta_path = Path('../meta/meta_train_not_mount.csv')
    batch_size = 32

    meta_df = pd.read_csv(meta_path, header=None)
    metadata = meta_df[0].values.tolist()

    metadata_by_batch = devide_metadata(metadata, batch_size)

    for i, _metadata in enumerate(metadata_by_batch):
        print('='*10)
        print(f'batch {i}/{batch_size}')

        print('wave data ...')
        wave_data = load_data(_metadata)
        np.save(f'../dataset/generated_dataset/20210629/wave_{i}', wave_data)

        print('spec data ...')
        auged_data = augmentation(wave_data)
        np.save(f'../dataset/generated_dataset/20210629/spec_{i}', auged_data)

    print('complite!')


if __name__ == '__main__':
    run()
