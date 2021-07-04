import torch


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
