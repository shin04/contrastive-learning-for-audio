from os import device_encoding
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
# import torch.nn.functional as F

import config
from datasets.audioset import CLDataset
from models.cl_model import CLModel
from models.raw_model import Conv160
from models.spec_model import CNN6

# https://www.youtube.com/watch?v=_1eKr4rbgRI
# https://colab.research.google.com/drive/1UK8BD3xvTpuSj75blz8hThfXksh19YbA?usp=sharing#scrollTo=GBNm6bbDT9J3


def nt_xent_loss(q, pos_k, temperature):
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
    model.train()

    n_epoch = config.n_epoch
    temperature = config.temperature
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
            optimizer.step()

            loss_epoch += loss.item()

            if step % 100 == 0:
                print(f"Step [{step}/{len(dataloader)}]\t Loss: {loss.item()}")

        print(
            f"Epoch [{epoch}/{n_epoch}]\t Loss: {loss_epoch / len(dataloader)}")


if __name__ == '__main__':
    main()
