import torch


def prediction(testloader, device, model):
    model.eval()

    test_corr = 0
    valid_acc = 0
    total = 0
    preds = []

    with torch.no_grad():
        for i, (t_data, labels) in enumerate(testloader):
            t_data = t_data.to(device)
            labels = labels.to(device)

            outputs = model(t_data)

            _, predict = torch.max(outputs.data, 1)
            preds += list(predict.to('cpu').detach().numpy().copy())

            total += labels.size(0)
            correct = (predict == labels).sum()
            test_corr += correct.item()

        valid_acc = test_corr / total

    return valid_acc, preds
