from datetime import datetime
from pathlib import Path
import os
import time

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


def train(trainloader, optimizer, device, global_step,  model, criterion, fold, writer=None):
    model.train()

    n_batch = len(trainloader)
    train_loss = 0
    train_acc = 0
    total = 0
    for batch_num, (t_data, labels) in enumerate(trainloader):
        s_time = time.time()

        optimizer.zero_grad()
        t_data = t_data.to(device)
        labels = labels.to(device)

        outputs = model(t_data)

        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()
        train_loss += loss.item()

        _, predict = torch.max(outputs.data, 1)
        total += labels.size(0)

        correct = (predict == labels).sum()
        train_acc += correct.item()

        if writer is not None:
            writer.add_scalar(f"{fold}/loss", loss.item(), global_step)
            writer.add_scalar(f"{fold}/acc", correct.item() /
                              labels.size(0), global_step)
        print(
            f'batch: {batch_num}/{n_batch}, '
            f'loss: {loss.item():.6f}, train loss: {train_loss/(batch_num+1):.6f}, '
            f'acc: {correct.item()/labels.size(0):.6f}, train acc: {train_acc/total:.6f}, '
            f'time: {time.time()-s_time:.5f}'
        )
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

    print(f'val loss: {valid_loss: .6f}, val acc: {valid_acc: .6f}')


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

    print(f'test acc: {valid_acc: .6f}')


@ hydra.main(config_path='/ml/config', config_name='esc')
def run(cfg):
    """set config"""
    debug = cfg['debug']
    path_cfg = cfg['path']
    preprocess_cfg = cfg['preprocess']
    train_cfg = cfg['training']

    """set pathes"""
    ts = datetime.now().strftime(TIME_TEMPLATE)
    print("TIMESTAMP:", ts)
    print('DEBUG MODE:', debug)

    audio_path = Path(path_cfg['audio'])
    meta_path = Path(path_cfg['meta'])
    pretrain_model_path = Path(path_cfg['pretrain_model'])

    log_path = Path(path_cfg['tensorboard']) / ts
    if (not debug) and (not log_path.exists()):
        log_path.mkdir(parents=True)

    print('PATH')
    print('audio:', audio_path)
    print('meta:', meta_path)
    if not debug:
        print(f'tensorboard: {log_path}')

    """set parameters"""
    device = torch.device(cfg['device'])
    num_worker = train_cfg['num_worker']
    if num_worker == -1:
        num_worker = os.cpu_count()

    n_epoch = train_cfg['n_epoch']
    batch_size = train_cfg['batch_size']
    lr = train_cfg['lr']

    sr = preprocess_cfg['sr']
    crop_sec = preprocess_cfg['crop_sec']
    n_mels = preprocess_cfg['n_mels']
    freq_shift_size = preprocess_cfg['freq_shift_size']

    print('PARAMETERS')
    print("device:", device)
    print("num_worker:", num_worker)
    print("n_epoch:", n_epoch)
    print("batch_size:", batch_size)
    print("lr:", lr)
    print("sr", sr)
    print("crop secconds", crop_sec)
    print("n_mels:", n_mels)
    print("frequency shift size:", freq_shift_size)

    """tensorboard"""
    if not debug:
        writer = SummaryWriter(log_dir=log_path)
    else:
        writer = None

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
            sr=sr,
            crop_sec=crop_sec
        )

        validset = ESCDataset(
            audio_path=audio_path,
            metadata_path=meta_path,
            folds=fold_dict['valid'],
            sr=sr,
            crop_sec=crop_sec
        )

        testset = ESCDataset(
            audio_path=audio_path,
            metadata_path=meta_path,
            folds=fold_dict['test'],
            sr=sr,
            crop_sec=crop_sec
        )

        trainloader = DataLoader(
            trainset, batch_size=batch_size, shuffle=True, num_workers=num_worker, pin_memory=True)
        validloader = DataLoader(
            validset, batch_size=batch_size, shuffle=True, num_workers=num_worker, pin_memory=True)
        testloader = DataLoader(
            testset, batch_size=batch_size, shuffle=False, num_workers=num_worker, pin_memory=True)

        """prepare model"""
        cl_model = CLModel(cfg=preprocess_cfg, is_preprocess=True).to(device)
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

        raw_model = Conv160().to(device)
        raw_model.load_state_dict(raw_model_dict)

        model = ESC_Model(raw_model, 512*600, 512, 50).to(device)

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
                trainloader, optimizer, device, train_global_step, model, criterion, k_fold, writer)
            valid(validloader, device, model, criterion)

            if not debug:
                writer.add_scalar(f"{k_fold}/loss/epoch", train_loss, epoch)
                writer.add_scalar(f"{k_fold}/acc/epoch", train_acc, epoch)

            print(
                f'epoch: {epoch}/{n_epoch}, train loss: {train_loss: .6f}, train acc: {train_acc: .6f}')
        test(testloader, device, model)

    if not debug:
        writer.close()


if __name__ == '__main__':
    run()
