import torch.nn as nn
from ncps.torch import LTC
from ncps.wirings import AutoNCP


class LiquidConv50D2(nn.Module):
    def __init__(self):
        super(LiquidConv50D2, self).__init__()

        self.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=4,
            kernel_size=(1, 3),
        )
        self.conv2 = nn.Conv2d(
            in_channels=4,
            out_channels=8,
            kernel_size=(1, 3),
        )

        self.liquid1 = LTC(32, AutoNCP(64, 32))
        self.liquid2 = LTC(32, AutoNCP(32, 16))

        self.fc = nn.Linear(16, 1)
        self.sigmoid = nn.Sigmoid()

    def __repr__(self):
        return "LiquidConv50D2"

    def forward(self, x):
        B, T, F = x.shape

        x = x.unsqueeze(1)
        x = self.conv1(x)
        x = nn.functional.relu(x)
        x = self.conv2(x)

        x = x.permute(0, 2, 1, 3)
        x = x.reshape(B, T, -1)

        x, _ = self.liquid1(x)
        x, _ = self.liquid2(x)

        x = x[:, -1, :]

        x = self.fc(x)
        x = self.sigmoid(x)
        return x
