from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score

from esc.model_setup import model_setup


def predict(
    testloader: DataLoader, device: torch.device,
    data_format: str, weight_path: Path
) -> np.ndarray:

    model = model_setup(data_format, False, None).to(device)
    model.load_state_dict(torch.load(weight_path))
    model.eval()

    preds = []

    with torch.no_grad():
        for i, (t_data, labels) in enumerate(testloader):
            t_data = t_data.to(device)
            labels = labels.to(device)

            outputs = model(t_data)

            _, predict = torch.max(outputs.data, 1)
            preds += list(predict.to('cpu').detach().numpy().copy())

    return np.array(preds)


def calc_accuracy(meta_path: Path, predictions: np.array) -> float:
    labels = pd.read_csv(meta_path)['target'].values()
    acc = accuracy_score(labels, predictions)

    return acc
