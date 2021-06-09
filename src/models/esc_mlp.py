import torch.nn as nn
import torch.nn.functional as F

from torchinfo import summary


class ESC_Model(nn.Module):
    def __init__(self, base_model, in_channels, hidden_dim, out_channels, is_training=True):
        super(ESC_Model, self).__init__()

        self.base_model = base_model(pretrained=True)
        for param in self.base_model.parameters():
            param.requires_grad = False

        self.features = self.base_model.features

        self.in_channels = in_channels
        self.hidden_dim = hidden_dim
        self.out_channels = out_channels

        self.is_training = is_training

        self.hidden_layer = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.norm = nn.BatchNorm1d(self.hidden_dim)

    def forward(self, input):
        x = self.features(input)

        x = self.hidden_layer(x)
        x = self.norm(x)
        digit = F.relu(x)

        out = F.softmax(digit)

        return out


if __name__ == '__main__':
    model = ESC_Model().cuda()
    summary(model, input_size=(32, 1, 512*2))
