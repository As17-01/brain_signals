import torch.nn as nn
import torch
from src.utils import get_default_device


class GRULag1(nn.Module):
    def __init__(self):
        super(GRULag1, self).__init__()

        self.dropout1 = nn.Dropout(0.2)
        self.gru1 = nn.GRU(
            input_size=8, hidden_size=20, batch_first=True, bidirectional=False
        )

        self.dropout2 = nn.Dropout(0.2)
        self.gru2 = nn.GRU(
            input_size=20, hidden_size=20, batch_first=True, bidirectional=False
        )

        self.dropout3 = nn.Dropout(0.2)
        self.gru3 = nn.GRU(
            input_size=20, hidden_size=10, batch_first=True, bidirectional=False
        )

        self.dropout4 = nn.Dropout(0.2)
        self.gru4 = nn.GRU(
            input_size=10, hidden_size=10, batch_first=True, bidirectional=False
        )

        self.fc = nn.Linear(10, 1)
        self.sigmoid = nn.Sigmoid()

        self.device = get_default_device()

    def __repr__(self):
        return "GRULag1"

    def forward(self, x):
        # x_lagged = torch.cat(
        #     [
        #         torch.zeros_like(x[:, :1, :], device=self.device),
        #         x[:, :-1, :],
        #     ],
        #     dim=1
        # )
        # x = torch.cat([x, x_lagged], dim=2)

        x = self.dropout1(x)
        x, _ = self.gru1(x)

        x = self.dropout2(x)
        x, _ = self.gru2(x)

        x = self.dropout3(x)
        x, _ = self.gru3(x)

        x = self.dropout4(x)
        x, _ = self.gru4(x)

        x = x[:, -1, :]

        x = self.fc(x)
        x = self.sigmoid(x)
        return x
