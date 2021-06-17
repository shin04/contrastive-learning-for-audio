import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

import config
from datasets.esc_dataset import ESCDataset
from models.esc_mlp import ESC_Model
from models.cl_model import CLModel
from models.raw_model import Conv160
# from models.spec_model import CNN6


def train():
    device = torch.device(config.device)
    n_epoch = 1000
    lr = 0.001
    batch_size = 32

    writer = SummaryWriter(log_dir='../log/esc')

    dataset = ESCDataset(
        audio_path='/ml/dataset/esc/audio',
        metadata_path='/ml/dataset/esc/meta/esc50.csv',
        data_type='raw',
        data_crop_size=3,
        n_mels=80,
        freq_shift_size=20
    )
    dataloader = DataLoader(dataset, batch_size=batch_size,
                            shuffle=True, pin_memory=True)

    cl_model = CLModel()
    cl_model.load_state_dict(torch.load(
        '../results/20210610112655/best.pt'))
    state_dict_keys = list(cl_model.state_dict().keys())

    raw_model_dict = {}
    spec_model_dict = {}
    for k in state_dict_keys:
        _k = k.split('.')
        if 'raw_model' == _k[0]:
            raw_model_dict[".".join(_k[1:])] = cl_model.state_dict()[k]
        elif 'spec_model' == _k[0]:
            spec_model_dict[".".join(_k[1:])] = cl_model.state_dict()[k]

    raw_model = Conv160().cuda()
    raw_model.load_state_dict(raw_model_dict)

    model = ESC_Model(raw_model, 32, 422912, 512, 50).cuda()

    # spec_model = CNN6()
    # spec_model.load_state_dict(spec_model_dict)

    # model = ESC_Model(spec_model, 32, 2048, 512, 50).cuda()

    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    model.train()

    n_batch = len(dataset)//batch_size
    print('n_batch', n_batch)
    global_step = 0
    for epoch in range(n_epoch):
        train_loss = 0
        batch_num = 0
        for data, labels in dataloader:
            if batch_num == n_batch:
                continue
            optimizer.zero_grad()
            data = data.to(device)
            labels = labels.to(device)

            outputs = model(data)

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

            writer.add_scalar("loss", loss.item(), global_step)

            print(
                f'batch: {batch_num}/{n_batch},  loss: {loss.item()}, train loss: {train_loss/(batch_num+1)}')
            batch_num += 1
            global_step += 1

        writer.add_scalar("loss", loss.item(), epoch)
        print(
            f'epoch: {epoch}/{n_epoch},  train loss: {train_loss/(batch_num+1)}')


if __name__ == '__main__':
    train()
