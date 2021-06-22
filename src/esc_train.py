from datetime import datetime
from pathlib import Path

import hydra

import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

from datasets.esc_dataset import ESCDataset
from models.esc_mlp import ESC_Model
from models.cl_model import CLModel
from models.raw_model import Conv160
# from models.spec_model import CNN6

TIME_TEMPLATE = '%Y%m%d%H%M%S'


def train(trainloader, optimizer, device, global_step,  model, criterion, writer, fold):
    model.train()

    n_batch = len(trainloader)
    train_loss = 0
    train_acc = 0
    total = 0
    for batch_num, (t_data, labels) in enumerate(trainloader):
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
        total += labels.size(0)
        correct = (predict == labels).sum()
        train_acc += correct.item()

        writer.add_scalar(f"{fold}/loss", loss.item(), global_step)
        writer.add_scalar(f"{fold}/acc", correct.item() /
                          labels.size(0), global_step)
        print(
            f'batch: {batch_num}/{n_batch}, '
            f'loss: {loss.item()}, train loss: {train_loss/(batch_num+1)}, '
            f'acc: {correct.item()/labels.size(0)}, train acc: {train_acc/total} ')
        global_step += 1

    train_loss /= n_batch
    train_acc /= len(trainloader.dataset)

    return global_step, train_loss, train_acc


def valid(validloader, device, model, criterion):
    model.eval()

    valid_loss = 0
    valid_acc = 0
    total = 0

    with torch.no_grad():
        for i, (t_data, labels) in enumerate(validloader):
            t_data = t_data.to(device)
            labels = labels.to(device)

            outputs = model(t_data)

            loss = criterion(outputs, labels)
            valid_loss += loss.item()

            _, predict = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct = (predict == labels).sum()
            valid_acc += correct.item()

        valid_loss /= len(validloader)
        valid_acc /= total

    print(f'val loss: {valid_loss}, val acc: {valid_acc}')


def test(testloader, device, model):
    model.eval()

    valid_acc = 0
    total = 0

    with torch.no_grad():
        for i, (t_data, labels) in enumerate(testloader):
            t_data = t_data.to(device)
            labels = labels.to(device)

            outputs = model(t_data)

            _, predict = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct = (predict == labels).sum()
            valid_acc += correct.item()

        valid_acc /= total

    print(f'test acc: {valid_acc}')


@ hydra.main(config_path='/ml/config', config_name='esc')
def run(cfg):
    """set config"""
    path_cfg = cfg['path']
    # preprocess_cfg = cfg['preprocess']
    train_cfg = cfg['training']

    """set path"""
    ts = datetime.now().strftime(TIME_TEMPLATE)
    print(f"TIMESTAMP: {ts}")

    audio_path = Path(path_cfg['audio'])
    meta_path = Path(path_cfg['meta'])
    pretrain_model_path = Path(path_cfg['pretrain_model'])

    log_path = Path(path_cfg['tensorboard']) / ts
    if not log_path.exists():
        log_path.mkdir(parents=True)

    print('PATH')
    print(f'audio: {audio_path}')
    print(f'meta: {meta_path}')
    print(f'tensorboard: {log_path}')

    """set parameters"""
    device = torch.device(cfg['device'])
    n_epoch = train_cfg['n_epoch']
    batch_size = train_cfg['batch_size']
    lr = train_cfg['lr']

    writer = SummaryWriter(log_dir=log_path)

    print('PARAMETERS')
    print(f'device: {device}')
    print(f'n_epoch: {n_epoch}')
    print(f'batch_size: {batch_size}')
    print(f'lr: {lr}')

    """
    dividing data, reference below
    https://github.com/karolpiczak/paper-2015-esc-convnet/issues/2
    """
    fold_dict_list = [
        {"train": [2, 3, 4], "valid": [5], "test": [1]},
        {"train": [3, 4, 5], "valid": [1], "test": [2]},
        {"train": [1, 4, 5], "valid": [2], "test": [3]},
        {"train": [1, 2, 5], "valid": [3], "test": [4]},
        {"train": [1, 2, 3], "valid": [4], "test": [5]},
    ]

    print('FOLD')
    for i, fold_dict in enumerate(fold_dict_list):
        print(f'flod {i}: {fold_dict}')

    for k_fold, fold_dict in enumerate(fold_dict_list):
        print(f'===== fold: {k_fold}')

        """prepare dataset"""
        trainset = ESCDataset(
            audio_path=audio_path,
            metadata_path=meta_path,
            folds=fold_dict['train'],
            data_type='raw',
            data_crop_size=3
        )

        validset = ESCDataset(
            audio_path=audio_path,
            metadata_path=meta_path,
            folds=fold_dict['valid'],
            data_type='raw',
            data_crop_size=3
        )

        testset = ESCDataset(
            audio_path=audio_path,
            metadata_path=meta_path,
            folds=fold_dict['test'],
            data_type='raw',
            data_crop_size=3
        )

        trainloader = DataLoader(
            trainset, batch_size=batch_size, shuffle=True, pin_memory=True)
        validloader = DataLoader(
            validset, batch_size=batch_size, shuffle=True, pin_memory=True)
        testloader = DataLoader(
            testset, batch_size=batch_size, shuffle=False, pin_memory=True)

        """prepare model"""
        cl_model = CLModel()
        cl_model.load_state_dict(torch.load(pretrain_model_path))
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

        """prepare optimizer and loss function"""
        optimizer = optim.Adam(model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()

        """training and test"""
        train_global_step = 0
        for epoch in range(n_epoch):
            train_global_step, train_loss, train_acc = train(
                trainloader, optimizer, device, train_global_step, model, criterion, writer, k_fold)
            valid(validloader, device, model, criterion)

            writer.add_scalar(f"{k_fold}/loss/epoch", train_loss, epoch)
            writer.add_scalar(f"{k_fold}/acc/epoch", train_acc, epoch)

            print(
                f'epoch: {epoch}/{n_epoch}, train loss: {train_loss}, train acc: {train_acc}')
        test(testloader, device, model)

    writer.close()


if __name__ == '__main__':
    run()
