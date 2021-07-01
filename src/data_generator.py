from datetime import datetime
from pathlib import Path
import typing
from multiprocessing import Pool

import numpy as np
import pandas as pd
import librosa
from tqdm import tqdm
import hydra

from datasets.utils import random_crop, mel_spec

TIME_TEMPLATE = '%Y%m%d%H%M%S'


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


def batch_process(
    save_path: Path, metadata: Path, sr: int, aug_func: typing.Callable, batch_num: int,
    win_sec: float, hop_sec: float, n_mels: int, freq_shift_size: int
):
    wave_data = load_data(metadata, sr)
    np.save(save_path / f'wave_{batch_num}', wave_data)

    auged_data = augmentation(
        wave_data, sr, aug_func, win_sec, hop_sec, n_mels, freq_shift_size
    )
    np.save(save_path / f'spec_{batch_num}', auged_data)

    print(f'batch {batch_num} completed !')


@ hydra.main(config_path='../config', config_name='data_generation')
def run(cfg):
    ts = datetime.now().strftime(TIME_TEMPLATE)

    save_path = Path(cfg['save_path']) / ts
    meta_path = Path(cfg['meta_path'])
    batch_size = 32
    sr = 22050
    aug_func = mel_spec
    win_sec = 0.2
    hop_sec = 0.1
    n_mels = 80
    freq_shift_size = None

    meta_df = pd.read_csv(meta_path, header=None)
    metadata = meta_df[0].values.tolist()

    metadata_by_batch = devide_metadata(metadata, batch_size)

    with Pool(7) as pool:
        preprocesses = [
            pool.apply_async(
                batch_process,
                (
                    save_path, metadata_by_batch[batch_num], sr, aug_func,
                    batch_num, win_sec, hop_sec, n_mels, freq_shift_size
                )
            )
            for batch_num in range(len(metadata_by_batch))
        ]

        [preprocess.get() for preprocess in preprocesses]

    print('complite!')


if __name__ == '__main__':
    run()
