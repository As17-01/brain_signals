import torch.nn as nn
import torch
from src.utils import get_default_device


class BiLSTMLag1(nn.Module):
    def __init__(self):
        super(BiLSTMLag1, self).__init__()

        self.dropout1 = nn.Dropout(0.2)
        self.lstm1 = nn.LSTM(
            input_size=16, hidden_size=20, batch_first=True, bidirectional=True
        )

        self.dropout2 = nn.Dropout(0.2)
        self.lstm2 = nn.LSTM(
            input_size=40, hidden_size=20, batch_first=True, bidirectional=True
        )

        self.dropout3 = nn.Dropout(0.2)
        self.lstm3 = nn.LSTM(
            input_size=40, hidden_size=10, batch_first=True, bidirectional=True
        )

        self.dropout4 = nn.Dropout(0.2)
        self.lstm4 = nn.LSTM(
            input_size=20, hidden_size=10, batch_first=True, bidirectional=True
        )

        self.fc = nn.Linear(20, 1)
        self.sigmoid = nn.Sigmoid()

        self.device = get_default_device()

    def __repr__(self):
        return "BiLSTMLag1"

    def forward(self, x):
        x_lagged = torch.cat(
            [
                torch.zeros_like(x[:, :1, :], device=self.device),
                x[:, :-1, :],
            ],
            dim=1
        )
        x = torch.cat([x, x_lagged], dim=2)

        x = self.dropout1(x)
        x, _ = self.lstm1(x)

        x = self.dropout2(x)
        x, _ = self.lstm2(x)

        x = self.dropout3(x)
        x, _ = self.lstm3(x)

        x = self.dropout4(x)
        x, _ = self.lstm4(x)

        x = x[:, -1, :]

        x = self.fc(x)
        x = self.sigmoid(x)
        return x
