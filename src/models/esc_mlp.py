import torch
import torch.nn as nn
import torch.nn.functional as F

from torchinfo import summary

from models.raw_model import Conv160
from models.spec_model import CNN6


class ESC_Model(nn.Module):
    def __init__(
        self, base_model, in_channels, hidden_dim, out_channels,
        use_pretrained=True, is_training=True
    ):

        super(ESC_Model, self).__init__()

        self.base_model = base_model
        if use_pretrained:
            for param in self.base_model.parameters():
                param.requires_grad = False

        self.is_training = is_training

        self.flatten = nn.Flatten()
        self.hidden_layer = nn.Linear(in_channels, hidden_dim)
        self.norm = nn.BatchNorm1d(hidden_dim)
        self.out_layer = nn.Linear(hidden_dim, out_channels)

    def forward(self, input):
        x = self.base_model(input)
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.flatten(x)

        x = self.hidden_layer(x)
        # x = F.dropout(x, p=0.2, training=self.training)
        x = self.norm(x)
        x = F.relu(x)

        digit = self.out_layer(x)

        return digit


if __name__ == '__main__':
    state_dict = torch.load('/ml/results/20210704200524/best.pt')
    state_dict_keys = list(state_dict.keys())

    raw_model_dict = {}
    spec_model_dict = {}
    for k in state_dict_keys:
        _k = k.split('.')
        if 'raw_model' == _k[0]:
            raw_model_dict[".".join(_k[1:])] = state_dict[k]
        elif 'spec_model' == _k[0]:
            spec_model_dict[".".join(_k[1:])] = state_dict[k]

    raw_model = Conv160().cpu()
    raw_model.load_state_dict(raw_model_dict)
    model = ESC_Model(raw_model, 512*1000, 512, 50).cpu()
    summary(model, input_size=(32, 1, 160*1000))

    spec_model = CNN6().cpu()
    spec_model.load_state_dict(spec_model_dict)
    model = ESC_Model(spec_model, 2048, 512, 50).cpu()
    summary(model, input_size=(32, 1, 64, 1000))
