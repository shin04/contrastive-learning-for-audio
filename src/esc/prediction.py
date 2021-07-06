from pathlib import Path
from typing import Union

import numpy as np
import torch
from torch.data.utils import DataLoader

from esc.model_setup import model_setup


def prediction(
    testloader: DataLoader, device: torch.device,
    data_format: str, weight_path: Path
) -> Union[float, np.ndarray]:

    model = model_setup(data_format, False, None).to(device)
    model.load_state_dict(torch.load(weight_path))
    model.eval()

    pred_corr = 0
    pred_acc = 0
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
            pred_corr += correct.item()

        pred_acc = pred_corr / total

    return pred_acc, np.array(preds)
