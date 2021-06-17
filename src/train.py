from datetime import datetime
from pathlib import Path
import argparse

import torch
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

import config
from datasets.audioset import CLDataset
from models.cl_model import CLModel
from utils.cosine_decay_rule import CosineDecayScheduler

TIME_TEMPLATE = '%Y%m%d%H%M%S'


def nt_xent_loss(q, pos_k, temperature):
    # reference
    # https://www.youtube.com/watch?v=_1eKr4rbgRI
    # https://colab.research.google.com/drive/1UK8BD3xvTpuSj75blz8hThfXksh19YbA?usp=sharing#scrollTo=GBNm6bbDT9J3

    out = torch.cat([q, pos_k], dim=0)
    n_samples = len(out)

    # Full similarity matrix
    cov = torch.mm(out, out.t().contiguous())
    sim = torch.exp(cov / temperature)

    mask = ~torch.eye(n_samples, device=sim.device).bool()
    neg = sim.masked_select(mask).view(n_samples, -1).sum(dim=-1)

    # Positive similarity
    pos = torch.exp(torch.sum(q * pos_k, dim=-1) / temperature)
    pos = torch.cat([pos, pos], dim=0)

    loss = -torch.log(pos / neg).mean()

    return loss


def train(args):
    """set pathes"""
    ts = datetime.now().strftime(TIME_TEMPLATE)
    print(f"TIMESTAMP: {ts}")

    if args.ckpt is None:
        model_ckpt_path = Path('../models/contrastive_learning') / ts
        if not model_ckpt_path.exists():
            model_ckpt_path.mkdir(parents=True)
    else:
        try:
            model_ckpt_path = Path(args.ckpt)
        except TypeError:
            print(f'file "{args.ckpt}" is not exist')
            return

        if not model_ckpt_path.exists():
            print(f'file "{model_ckpt_path}" is not exist')
            return

    result_path = Path('../results') / ts
    if not result_path.exists():
        result_path.mkdir(parents=True)

    if args.tboard is None:
        log_path = Path('../log/tensorboard') / ts
        if not log_path.exists():
            log_path.mkdir(parents=True)
    else:
        try:
            log_path = Path(args.tboard)
        except TypeError:
            print(f'file "{args.tboard}" is not exist')
            return

        if not log_path.exists():
            print(f'file "{log_path}" is not exist')
            return

    print("PATH")
    print("checkpoint:", model_ckpt_path)
    print("best model:", result_path)
    print("tensorboard:", log_path)

    """set training parameter"""
    device = torch.device(config.device)
    n_epoch = config.n_epoch
    temperature = config.temperature
    lr = config.lr
    batch_size = config.batch_size
    n_mels = config.n_mels
    freq_shift_size = config.freq_shift_size

    print("TRAINING PARAMETERS")
    print("device:", device)
    print("n_epoch:", n_epoch)
    print("temperature:", temperature)
    print("lr:", lr)
    print("batch_size:", batch_size)
    print("n_mels:", n_mels)
    print("freq_shift_size:", freq_shift_size)

    """tensorboard"""
    writer = SummaryWriter(log_dir=log_path)

    """prepare dataset"""
    dataset = CLDataset(
        audio_path=config.audio_path, metadata_path=config.metadata_path,
        q_type='raw', k_type='raw', data_crop_size=3,
        n_mels=n_mels, freq_shift_size=freq_shift_size
    )
    dataloader = DataLoader(dataset, batch_size=batch_size,
                            shuffle=True,  pin_memory=True)

    """prepare models"""
    model = CLModel().to(device)
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
    best_loss = -1
    global_step = 0
    for epoch in range(s_epoch, n_epoch):
        print(f'epoch: {epoch}')

        loss_epoch = 0

        for step in range(len(dataloader)):
            (q, pos_k, _) = next(iter(dataloader))
            q = q.to(device)
            pos_k = pos_k.to(device)

            z_i, z_j = model(q, pos_k)

            loss = nt_xent_loss(z_i, z_j, temperature)

            loss.backward()
            optimizer.step()

            if step % 100 == 0:
                print(f"Step [{step}/{len(dataloader)}],  Loss: {loss.item()}")

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

        if best_loss < loss_epoch:
            best_loss = loss_epoch
            with open(result_path / 'best.pt', 'wb') as f:
                torch.save(model.state_dict(), f)

        lr_scheduler.step(epoch)

    print(f'complete training: {result_path}')
    writer.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--ckpt', help='path to model check point')
    parser.add_argument('-t', '--tboard', help='path to tensorboard')

    args = parser.parse_args()

    train(args)
