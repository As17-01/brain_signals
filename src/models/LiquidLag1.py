import torch.nn as nn
import torch

from ncps.torch import LTC
from ncps.wirings import AutoNCP
from src.utils import get_default_device


class LiquidLag1(nn.Module):
    def __init__(self):
        super(LiquidLag1, self).__init__()

        self.liquid1 = LTC(16, AutoNCP(64, 32))
        self.liquid2 = LTC(32, AutoNCP(32, 16))

        self.fc = nn.Linear(16, 1)
        self.sigmoid = nn.Sigmoid()

        self.device = get_default_device()

    def __repr__(self):
        return "LiquidLag1"

    def forward(self, x):
        x_lagged = torch.cat(
            [
                torch.zeros_like(x[:, :1, :], device=self.device),
                x[:, :-1, :],
            ],
            dim=1
        )
        x = torch.cat([x, x_lagged], dim=2)

        x, _ = self.liquid1(x)
        x, _ = self.liquid2(x)

        x = x[:, -1, :]

        x = self.fc(x)
        x = self.sigmoid(x)
        return x
