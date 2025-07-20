import torch.nn as nn
from ncps.torch import LTC
from ncps.wirings import AutoNCP


class LiquidConv50(nn.Module):
    def __init__(self):
        super(LiquidConv50, self).__init__()

        self.causal_conv = nn.Conv1d(
            in_channels=8,
            out_channels=50,
            kernel_size=3
        )

        self.liquid1 = LTC(50, AutoNCP(64, 32))
        self.liquid2 = LTC(32, AutoNCP(32, 16))

        self.fc = nn.Linear(16, 1)
        self.sigmoid = nn.Sigmoid()

    def __repr__(self):
        return "LiquidConv50"

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = nn.functional.pad(x, (2, 0))
        x = self.causal_conv(x)
        x = nn.functional.relu(x)
        x = x.permute(0, 2, 1)

        x, _ = self.liquid1(x)
        x, _ = self.liquid2(x)

        x = x[:, -1, :]

        x = self.fc(x)
        x = self.sigmoid(x)
        return x
