from pathlib import Path

from tqdm import tqdm
import soundfile as sf
import pandas as pd


def get_audio_names(audio_path, crop_sec):
    audio_path = Path(audio_path)

    audio_names = []
    for audio_name in tqdm(audio_path.glob('*.wav')):
        wave_data, sr = sf.read(audio_name)

        crop_size = crop_sec * sr
        data_length = len(wave_data)

        if data_length > crop_size:
            audio_names.append(audio_name)

    return audio_names


if __name__ == '__main__':
    audio_names = get_audio_names('/ml/dataset/audio/balanced_train_segments', 3)
    meta_s = pd.Series(audio_names)
    meta_s.to_csv('/ml/meta.csv', header=False, index=False)
