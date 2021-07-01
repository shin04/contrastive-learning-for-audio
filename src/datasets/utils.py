from typing import Union

import numpy as np
import librosa


def random_crop(data: np.ndarray, crop_size: int) -> Union[np.ndarray, int]:
    data_length = len(data)

    start = np.random.randint(0, data_length-crop_size)
    end = start+crop_size

    crop_data = data[start:end]

    return crop_data, start


def mixing(input: np.ndarray, noise: np.ndarray) -> Union[np.ndarray, np.float64]:
    alpha = np.random.beta(5, 2, 1)[0]
    mixed_data = alpha*input + (1-alpha)*noise

    return mixed_data, alpha


def frequemcy_shift_by_spec(input: np.ndarray, sr: int, win_size: int, max_shift: int = 20) -> np.ndarray:
    data_len = input.shape[1]
    shift_size = np.random.randint(-max_shift, max_shift)

    # freqs = librosa.fft_frequencies(sr=sr, n_fft=win_size)
    freq_step = sr // win_size
    shift_step = shift_size // freq_step

    if shift_step > 0:
        shifted_data = np.pad(
            input, ((0, 0), (shift_step, 0)), mode='constant')[:, :data_len]
    elif shift_step < 0:
        shifted_data = np.pad(
            input, ((0, 0), (0, -1*shift_step)), mode='constant')[:, -1*shift_step:]
    else:
        shifted_data = input

    return shifted_data


def mel_spec(input: np.ndarray, sr: int, win_size: int, hop_len: int, n_mels: int, max_sift: int = None) -> np.ndarray:
    spec = librosa.stft(
        y=input, n_fft=win_size, win_length=win_size, hop_length=hop_len)
    spec = np.abs(spec) ** 2.0

    if max_sift is not None:
        spec = frequemcy_shift_by_spec(spec, sr, win_size, max_sift)

    mel_filter_bank = librosa.filters.mel(sr=sr, n_fft=win_size, n_mels=n_mels)
    mel = np.dot(mel_filter_bank, spec)

    # mel = librosa.feature.melspectrogram(
    #     y=crop_data, sr=sr, n_mels=80, n_fft=win_size, win_length=win_size, hop_length=hop_len)
    # log_mel = np.log(mel)

    return mel


if __name__ == '__main__':
    """mixing"""
    print("==mixing"+"="*10)
    data = 2 * np.random.rand(32000*10) - 1
    noise = 100 * np.random.rand(32000*10)

    mixed_data, alpha = mixing(data, noise)

    print(f'original mean: {np.mean(data)}')
    print(f'noise mean: {np.mean(noise)}')
    print(f'mixed mean: {np.mean(mixed_data)} (aplha={alpha})')
    print(f'calc mixed mean: {alpha*np.mean(data)+(1-alpha)*np.mean(noise)}')

    """frequency shift"""
    print("==frequency shift"+"="*10)
    sr = 32000
    data = 2 * np.random.rand(sr*3) - 1
    win_size = int(0.2*sr)
    hop_len = int(0.1*sr)
    # mel = mel_spec(data, sr, win_size, hop_len, 80)
    spec = librosa.stft(
        y=data, n_fft=win_size, win_length=win_size, hop_length=hop_len)
    spec = np.abs(spec) ** 2.0

    shifted_data = frequemcy_shift_by_spec(spec, sr, win_size, 15)
    # print(shifted_data)
    print(spec.shape)
    print(shifted_data.shape)
