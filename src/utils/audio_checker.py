from pathlib import Path

from tqdm import tqdm
import soundfile as sf
import pandas as pd


def get_audio_names(audio_path, crop_sec):
    audio_path = Path(audio_path)

    audio_names = []
    eval_names = []
    for audio_name in tqdm(audio_path.glob('**/*.wav')):
        wave_data, sr = sf.read(audio_name)

        crop_size = crop_sec * sr
        data_length = len(wave_data)

        if data_length > crop_size:
            if 'eval' == str(audio_name).split('_')[0]:
                eval_names.append(audio_name)
            else:
                audio_names.append(audio_name)

    return audio_names, eval_names


if __name__ == '__main__':
    # audio_names, eval_names = get_audio_names(
    #     '/ml/dataset/audioset/audio', 3)
    audio_names, eval_names = get_audio_names(
        '../../../../nas02/internal/datasets/Audioset/audio', 3)
    meta_s = pd.Series(audio_names)
    eval_meta_s = pd.Series(eval_names)
    meta_s.to_csv('../../meta-train.csv', header=False, index=False)
    eval_meta_s.to_csv('../../meta-eval.csv', header=False, index=False)
