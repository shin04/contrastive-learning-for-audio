from datetime import datetime
from pathlib import Path

import torch
from torch.utils.data import DataLoader
import torch.optim as optim

import config
from datasets.audioset import CLDataset
from models.cl_model import CLModel
from utils.cosine_decay_rule import CosineDecayScheduler

TIME_TEMPLATE = '%Y%m%d%H%M%S'


def nt_xent_loss(q, pos_k, temperature):
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


def main():
    ts = datetime.now().strftime(TIME_TEMPLATE)
    result_path = Path('../results') / ts

    if not result_path.exists():
        result_path.mkdir(parents=True)

    device = torch.device(config.device)

    dataset = CLDataset(
        audio_path=config.audio_path, metadata_path=config.metadata_path,
        q_type='raw', k_type='raw', data_crop_size=3
    )
    dataloader = DataLoader(
        dataset, batch_size=config.batch_size,
        shuffle=True,  pin_memory=True
    )

    model = CLModel().to(device)
    optimizer = optim.Adam(model.parameters(), lr=config.lr, amsgrad=False)
    lr_scheduler_func = CosineDecayScheduler(base_lr=1, max_epoch=config.n_epoch)
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_scheduler_func)
    model.train()

    n_epoch = config.n_epoch
    temperature = config.temperature
    best_loss = -1

    for epoch in range(n_epoch):
        print(f'epoch: {epoch}')

        loss_epoch = 0

        for step in range(len(dataloader)):
            (q, pos_k, _) = next(iter(dataloader))
            q = q.to(device)
            pos_k = pos_k.to(device)

            z_i, z_j = model(q, pos_k)

            loss = nt_xent_loss(z_i, z_j, temperature)

            loss.backward()
            # optimizer.step()
            lr_scheduler.step(epoch)

            loss_epoch += loss.item()

            if step % 100 == 0:
                print(f"Step [{step}/{len(dataloader)}]\t Loss: {loss.item()}\t lr: {lr_scheduler.get_lr()}")

        print(
            f"Epoch [{epoch}/{n_epoch}]\t Loss: {loss_epoch / len(dataloader)}")

        if best_loss < loss_epoch:
            best_loss = loss_epoch
            with open(result_path / 'best.pt', 'wb') as f:
                torch.save(model.state_dict(), f)

    print(f'complete training: {result_path}')


if __name__ == '__main__':
    main()
