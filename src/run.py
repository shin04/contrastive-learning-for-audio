from datetime import datetime
from pathlib import Path
import time

import hydra

import torch
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from datasets.audioset import AudioSetDataset, HDF5Dataset
from models.cl_model import CLModel
from utils.scheduler import CosineDecayScheduler
from utils.losses import nt_xent_loss

TIME_TEMPLATE = '%Y%m%d%H%M%S'


@ hydra.main(config_path='../config', config_name='pretrain')
def train(cfg):
    """set config"""
    path_cfg = cfg['path']
    preprocess_cfg = cfg['preprocess']
    train_cfg = cfg['training']

    """set pathes"""
    audio_path = path_cfg['audio']
    hdf_path = path_cfg['hdf']
    metadata_path = path_cfg['meta']

    if train_cfg['ckpt'] == -1:
        ts = datetime.now().strftime(TIME_TEMPLATE)
    else:
        ts = str(train_cfg['ckpt'])
    print(f"TIMESTAMP: {ts}")

    # checkpoint path
    model_ckpt_path = Path(path_cfg['model']) / ts
    if not model_ckpt_path.exists():
        if train_cfg['ckpt'] != -1:
            raise RuntimeError('checkpoint file is not found')
        model_ckpt_path.mkdir(parents=True)

    # tensorboard path
    log_path = Path(path_cfg['tensorboard']) / ts
    if not log_path.exists():
        if train_cfg['ckpt'] != -1:
            raise RuntimeError('tensorboard log file is not found')
        log_path.mkdir(parents=True)

    # result path
    result_path = Path(path_cfg['result']) / ts
    if not result_path.exists():
        result_path.mkdir(parents=True)

    print("PATH")
    print("audio path: ", audio_path)
    print("hdf path:", hdf_path)
    print("metadata:", metadata_path)
    print("checkpoint:", model_ckpt_path)
    print("best model:", result_path)
    print("tensorboard:", log_path)

    """set training parameter"""
    device = torch.device(cfg['device'])
    n_epoch = train_cfg['n_epoch']
    batch_size = train_cfg['batch_size']
    lr = train_cfg['lr']
    temperature = train_cfg['temperature']

    dataset_type = preprocess_cfg['dataset_type']
    sr = preprocess_cfg['sr']
    crop_sec = preprocess_cfg['crop_sec']
    n_mels = preprocess_cfg['n_mels']
    freq_shift_size = preprocess_cfg['freq_shift_size']

    print("TRAINING PARAMETERS")
    print("device:", device)
    print("n_epoch:", n_epoch)
    print("batch_size:", batch_size)
    print("lr:", lr)
    print("temperature:", temperature)
    print("dataset type", dataset_type)
    print("sr", sr)
    print("crop secconds", crop_sec)
    print("n_mels:", n_mels)
    print("frequency shift size:", freq_shift_size)

    """tensorboard"""
    writer = SummaryWriter(log_dir=log_path)

    """prepare dataset"""
    if dataset_type == 'hdf5':
        dataset = HDF5Dataset(hdf5_dir=hdf_path, crop_sec=crop_sec)
    else:
        dataset = AudioSetDataset(
            metadata_path=Path(metadata_path), sr=sr, crop_sec=crop_sec,)
    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=False)

    """prepare models"""
    model = CLModel(cfg=preprocess_cfg, is_preprocess=True).to(device)

    optimizer = optim.Adam(model.parameters(), lr=lr, amsgrad=False)
    lr_scheduler_func = CosineDecayScheduler(base_lr=1, max_epoch=n_epoch)
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=lr_scheduler_func)

    """if exist pretrained model checkpoint, use it"""
    if len(list(model_ckpt_path.glob(r'model-epoch-*.ckpt'))) > 0:
        ckpt_file = list(model_ckpt_path.glob(r'model-epoch-*.ckpt'))[0]
        checkpoint = torch.load(ckpt_file)
        print(f'use checkpoint at {ckpt_file}')
        print(f'epoch: {checkpoint["epoch"]}')
        print(f'loss: {checkpoint["loss"]}')

        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        lr_scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        s_epoch = checkpoint['epoch'] + 1
        loss = checkpoint['loss']
    else:
        s_epoch = 0

    model.train()

    """training"""
    best_loss = 10000
    global_step = 0
    for epoch in range(s_epoch, n_epoch):
        print(f'epoch: {epoch}')

        loss_epoch = 0

        for step in range(len(dataloader)):
            data_iter = iter(dataloader)
            q = next(data_iter)

            q = q.to(device)
            z_i, z_j = model(q)

            loss = nt_xent_loss(z_i, z_j, temperature)

            optimizer.zero_grad()

            loss.backward()
            optimizer.step()

            if step % 100 == 0:
                print(
                    f"Step [{step}/{len(dataloader)}], Loss: {loss.item()}, Time: {process_time}")

            writer.add_scalar("Loss/train_epoch", loss.item(), global_step)

            loss_epoch += loss.item()
            global_step += 1

        print(
            f"Epoch [{epoch}/{n_epoch}],  Loss: {loss_epoch / len(dataloader)},  lr: {lr_scheduler.get_last_lr()}")
        writer.add_scalar("Loss/train", loss_epoch / len(dataloader), epoch)
        writer.add_scalar("learning_rate", lr, epoch)

        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': lr_scheduler.state_dict(),
            'loss': loss_epoch,
        }, model_ckpt_path/f'model-epoch-{epoch}.ckpt')
        old_ckpt = Path(model_ckpt_path/f'model-epoch-{epoch-1}.ckpt')

        # if prior check point exist, delete it
        if old_ckpt.exists():
            old_ckpt.unlink()

        if best_loss > loss_epoch:
            best_loss = loss_epoch
            with open(result_path / 'best.pt', 'wb') as f:
                torch.save(model.state_dict(), f)

        lr_scheduler.step()

    print(f'complete training: {result_path}')
    writer.close()


if __name__ == '__main__':
    train()
