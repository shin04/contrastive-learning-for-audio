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


def train(trainloader, optimizer, device, global_step,  model, criterion, writer, fold):
    model.train()

    n_batch = len(trainloader)
    train_loss = 0
    train_acc = 0
    for batch_num, (t_data, labels) in enumerate(trainloader):
        # if batch_num == n_batch:
        #     continue
        optimizer.zero_grad()
        t_data = t_data.to(device)
        labels = labels.to(device)

        outputs = model(t_data)

        optimizer.zero_grad()

        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

        _, predict = torch.max(outputs.data, 1)
        correct = (predict == labels).sum()
        train_acc += correct.item()

        writer.add_scalar(f"{fold}/loss", loss.item(), global_step)
        writer.add_scalar(f"{fold}/acc", correct.item(), global_step)
        print(
            f'batch: {batch_num}/{n_batch}, '
            f'loss: {loss.item()}, train loss: {train_loss/(batch_num+1)}, '
            f'acc: {correct.item()/len(labels)}, train acc: {train_acc/(len(labels)*(batch_num+1))} ')
        global_step += 1

    train_loss /= n_batch
    train_acc /= len(trainloader.dataset)

    return global_step, train_loss, train_acc


def valid(validloader, device, model, criterion):
    model.eval()

    valid_loss = 0
    valid_acc = 0

    with torch.no_grad():
        for i, (t_data, labels) in enumerate(validloader):
            t_data = t_data.to(device)
            labels = labels.to(device)

            outputs = model(t_data)

            loss = criterion(outputs, labels)
            valid_loss += loss.item()

            _, predict = torch.max(outputs.data, 1)
            correct = (predict == labels).sum()
            valid_acc += correct.item()

        valid_loss /= len(validloader)
        valid_acc /= len(validloader.dataset)

    print(f'val loss: {valid_loss}, val acc: {valid_acc}')


def run():
    device = torch.device(config.device)
    n_epoch = 100
    lr = 0.001
    batch_size = 32

    writer = SummaryWriter(log_dir='../log/esc')

    dataloaders = []
    for i in range(5):
        folds = [i+1 for i in range(5)]
        trainset = ESCDataset(
            audio_path='/ml/dataset/esc/audio',
            metadata_path='/ml/dataset/esc/meta/esc50.csv',
            folds=folds.remove(i+1),
            data_type='raw',
            data_crop_size=3
        )

        validset = ESCDataset(
            audio_path='/ml/dataset/esc/audio',
            metadata_path='/ml/dataset/esc/meta/esc50.csv',
            folds=[i+1],
            data_type='raw',
            data_crop_size=3
        )
        trainloader = DataLoader(trainset, batch_size=batch_size,
                                 shuffle=True, pin_memory=True)
        validloader = DataLoader(validset, batch_size=batch_size,
                                 shuffle=True, pin_memory=True)
        dataloaders.append((trainloader, validloader))

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

    model = ESC_Model(raw_model, 422912, 512, 50).cuda()

    # spec_model = CNN6()
    # spec_model.load_state_dict(spec_model_dict)

    # model = ESC_Model(spec_model, 32, 2048, 512, 50).cuda()

    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    for k_fold, (trainloader, validloader) in enumerate(dataloaders):
        print('='*10)
        print(f'fold: {k_fold}')

        train_global_step = 0
        for epoch in range(n_epoch):
            train_global_step, train_loss, train_acc = train(
                trainloader, optimizer, device, train_global_step, model, criterion, writer, k_fold)
            valid(validloader, device, model, criterion)

            writer.add_scalar(f"{k_fold}/loss/epoch", train_loss, epoch)
            writer.add_scalar(f"{k_fold}/acc/epoch", train_acc, epoch)

            print(
                f'epoch: {epoch}/{n_epoch}, train loss: {train_loss}, train acc: {train_acc}')

    writer.close()


if __name__ == '__main__':
    run()
