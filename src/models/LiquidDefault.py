import torch.nn as nn
from ncps.torch import LTC
from ncps.wirings import AutoNCP


class LiquidDefault(nn.Module):
    def __init__(self):
        super(LiquidDefault, self).__init__()

        self.liquid1 = LTC(8, AutoNCP(64, 32))
        self.liquid2 = LTC(32, AutoNCP(32, 16))

        self.fc = nn.Linear(16, 1)
        self.sigmoid = nn.Sigmoid()

    def __repr__(self):
        return "LiquidDefault"

    def forward(self, x):
        x, _ = self.liquid1(x)
        x, _ = self.liquid2(x)

        x = x[:, -1, :]  # Take the last time step

        x = self.fc(x)
        x = self.sigmoid(x)
        return x
