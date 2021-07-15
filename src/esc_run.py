from datetime import datetime
from pathlib import Path
import os

import hydra
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

from datasets.esc_dataset import ESCDataset
from esc.training import train, valid
from esc import prediction
from esc.model_setup import model_setup
from utils.callback import EarlyStopping

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
    print('pretrained model path:', pretrain_model_path)
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

    predictions = np.array([])

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
            validset, batch_size=batch_size, shuffle=False, num_workers=num_worker, pin_memory=True)
        testloader = DataLoader(
            testset, batch_size=batch_size, shuffle=False, num_workers=num_worker, pin_memory=True)

        """prepare model"""
        model = model_setup(
            data_format, use_pretrained, pretrain_model_path
        ).to(device)

        """prepare optimizer and loss function"""
        early_stopping = EarlyStopping(patience=15, verbose=True)
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.1)
        criterion = nn.CrossEntropyLoss()

        """training and test"""
        best_loss = 10000
        train_global_step = 0
        for epoch in range(n_epoch):
            train_global_step, train_loss, train_acc = train(
                trainloader, optimizer, device, train_global_step, model, criterion, k_fold, writer)
            valid_loss, valid_acc = valid(
                validloader, device, model, criterion)

            if not debug:
                writer.add_scalar(
                    f"fold:{k_fold}/train-loss", train_loss, epoch)
                writer.add_scalar(f"fold:{k_fold}/train-acc", train_acc, epoch)
                writer.add_scalar(
                    f"fold:{k_fold}/valid-loss", valid_loss, epoch)
                writer.add_scalar(f"fold:{k_fold}/valid-acc", valid_acc, epoch)

            print(
                f'epoch: {epoch}/{n_epoch}, '
                f'train loss: {train_loss: .6f}, '
                f'train acc: {train_acc: .6f}, '
                f'valid loss: {valid_loss: .6f}, '
                f'valid acc: {valid_acc: .6f}'
            )

            if (not debug) and (best_loss > train_loss):
                best_loss = train_loss
                with open(weight_path, 'wb') as f:
                    torch.save(model.state_dict(), f)

            early_stopping(valid_loss, model)
            if early_stopping.early_stop:
                print("Early stopping")
                break

        fold_preds = prediction.predict(
            testloader, device, data_format, weight_path)
        predictions = np.concatenate([predictions, fold_preds])

    if not debug:
        writer.close()

    acc = prediction.calc_accuracy(meta_path, predictions)
    np.save(result_path / 'preds', np.array(predictions))
    print(f'test accuracy: {acc}')
    print('complete!!')


if __name__ == '__main__':
    run()
