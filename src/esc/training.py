# import time

import torch


def train(trainloader, optimizer, device, global_step,  model, criterion, fold, writer=None):
    model.train()

    n_batch = len(trainloader)
    train_loss = 0
    train_acc = 0
    total = 0
    for batch_num, (t_data, labels) in enumerate(trainloader):
        # s_time = time.time()

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
            writer.add_scalar(f"fold:{fold}/loss", loss.item(), global_step)
            writer.add_scalar(f"fold:{fold}/acc", correct.item() /
                              labels.size(0), global_step)

        # print(
        #     f'batch: {batch_num}/{n_batch}, '
        #     f'loss: {loss.item():.6f}, train loss: {train_loss/(batch_num+1):.6f}, '
        #     f'acc: {correct.item()/labels.size(0):.6f}, train acc: {train_acc/total:.6f}, '
        #     f'time: {time.time()-s_time:.5f}, '
        # )

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

    return valid_loss, valid_acc
