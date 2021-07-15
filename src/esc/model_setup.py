from pathlib import Path

import torch
import torch.nn as nn

from models.raw_model import Conv160
from models.spec_model import CNN6
from models.esc_mlp import ESC_Model


def model_setup(
    data_format: str, use_pretrained: bool = False,
    pretrain_model_path: Path = None, is_training: bool = True
) -> nn.Module:

    if use_pretrained and (pretrain_model_path is None):
        raise ValueError('pretrain_model_path is None')

    if data_format == 'raw':
        base_model = Conv160()
    elif data_format == 'spec':
        base_model = CNN6()
    else:
        raise ValueError(f'unexpected parameter data_format="{data_format}"')

    if use_pretrained:
        pretrain_state_dict = (torch.load(pretrain_model_path))
        state_dict_keys = list(pretrain_state_dict.keys())

        pretrained_model_state_dict = {}
        for k in state_dict_keys:
            _k = k.split('.')
            if data_format in _k[0]:
                pretrained_model_state_dict[".".join(
                    _k[1:])] = pretrain_state_dict[k]

        base_model.load_state_dict(pretrained_model_state_dict)

    model = ESC_Model(base_model, 512*600, 512, 50,
                      use_pretrained, is_training)

    return model
