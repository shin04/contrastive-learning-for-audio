from datetime import datetime
from pathlib import Path
import os

import hydra

import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

from datasets.esc_dataset import ESCDataset
from models.esc_mlp import ESC_Model
from models.raw_model import Conv160
from models.spec_model import CNN6
from esc.training import train, valid
from esc.prediction import prediction

TIME_TEMPLATE = '%Y%m%d%H%M%S'


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

    result_path = Path(path_cfg['result']) / ts
    if (not debug) and (not result_path.exists()):
        result_path.mkdir(parents=True)

    print('PATH')
    print('audio:', audio_path)
    print('meta:', meta_path)
    if not debug:
        print(f'tensorboard: {log_path}')
        print(f'result: {result_path}')

    """set parameters"""
    device = torch.device(cfg['device'])

    use_pretrained = train_cfg['pretrained']
    data_format = train_cfg['data_format']

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
    print("use_pretrained:", use_pretrained)
    print("data_format:", data_format)
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

    preds_by_fold = []

    for k_fold, fold_dict in enumerate(fold_dict_list):
        print(f'===== fold: {k_fold}')

        # path to save weight
        if not debug:
            weight_path = result_path / f'fold{k_fold}-best.pt'

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
        if data_format == 'raw':
            base_model = Conv160().to(device)
        elif data_format == 'spec':
            base_model = CNN6().to(device)
        else:
            raise ValueError(f'unexpected parameter data_format="{data_format}"')

        if use_pretrained:
            pretrain_state_dict = (torch.load(pretrain_model_path))
            state_dict_keys = list(pretrain_state_dict.keys())

            pretrained_model_state_dict = {}
            for k in state_dict_keys:
                _k = k.split('.')
                if data_format in _k:
                    pretrained_model_state_dict[".".join(_k[1:])] = pretrain_state_dict[k]

            base_model.load_state_dict(pretrain_state_dict)

        model = ESC_Model(base_model, 512*600, 512, 50).to(device)

        """prepare optimizer and loss function"""
        optimizer = optim.Adam(model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()

        """training and test"""
        best_loss = 10000
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

            if (not debug) and (best_loss > train_loss):
                best_loss = train_loss
                with open(weight_path, 'wb') as f:
                    torch.save(model.state_dict(), f)

        valid_acc, preds = prediction(testloader, device, model)
        preds_by_fold.append(preds)
        print(f'test acc: {valid_acc: .6f}')

    if not debug:
        writer.close()


if __name__ == '__main__':
    run()
