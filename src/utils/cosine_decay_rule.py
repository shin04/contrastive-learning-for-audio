# import matplotlib.pyplot as plt

'''
in paper,
"starting from an initial learning rate of 10−4, and follows a cosine learning rate decay down to 10−6"
and, "the models are trained up to 400k steps"
so setting power=0.6 realize it

lr=10^-6まで学習回せば良くね？
'''


class CosineDecayScheduler:
    def __init__(self, base_lr: float, max_epoch: int, power=0.9):
        self.base_lr = base_lr
        self.max_epoch = max_epoch
        self.power = power

    def __call__(self, epoch: int):
        return (1 - max(epoch - 1, 1) / self.max_epoch) ** self.power * self.base_lr


if __name__ == '__main__':
    scheduler = CosineDecayScheduler(0.001, 400000, power=0.6)

    x = [i for i in range(400000)]
    y = []
    for i in x:
        y.append(scheduler(i))
    print(f'{y[-1]:f}')
    # plt.plot(x, y)
    # plt.savefig('lr.png')
