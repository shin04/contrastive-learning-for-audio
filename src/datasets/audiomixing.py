from typing import Union

import numpy as np


def mixing(input: np.ndarray, noise: np.ndarray) -> Union[np.ndarray, np.float64]:
    alpha = np.random.beta(5, 2, 1)[0]
    mixed_data = alpha*input + (1-alpha)*noise

    return mixed_data, alpha


if __name__ == '__main__':
    data = 2 * np.random.rand(22050*10) - 1
    noise = 100 * np.random.rand(22050*10)

    mixed_data, alpha = mixing(data, noise)

    print(f'original mean: {np.mean(data)}')
    print(f'noise mean: {np.mean(noise)}')
    print(f'mixed mean: {np.mean(mixed_data)} (aplha={alpha})')
    print(alpha*np.mean(data)+(1-alpha)*np.mean(noise))
