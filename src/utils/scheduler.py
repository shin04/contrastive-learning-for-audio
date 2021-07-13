import numpy as np

'''
in paper,
"starting from an initial learning rate of 10−4, and follows a cosine learning rate decay down to 10−6"
and, "the models are trained up to 400k steps"
'''


class CosineDecayScheduler:
    def __init__(self, max_epochs: int, warmup_lr_limit=0.001, warmup_epochs=0) -> None:
        self._max_epochs = max_epochs
        self._warmup_lr_limit = warmup_lr_limit
        self._warmup_epochs = warmup_epochs

    def __call__(self, epoch: int) -> float:
        epoch = max(epoch, 1)

        if epoch <= self._warmup_epochs:
            return self._warmup_lr_limit * epoch / self._warmup_epochs

        epoch -= 1
        rad = np.pi * epoch / self._max_epochs
        weight = (np.cos(rad) + 1.) / 2
        return self._warmup_lr_limit * weight


if __name__ == '__main__':
    step_num = 30
    scheduler = CosineDecayScheduler(step_num, 0.001)

    x = [i for i in range(step_num)]
    y = []
    for i in x:
        y.append(scheduler(i))
    print(f'{y[-1]:.12f}')
    print(y[-1])
