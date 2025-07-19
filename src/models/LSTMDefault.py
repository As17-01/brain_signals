import torch.nn as nn


class LSTMDefault(nn.Module):
    def __init__(self):
        super(LSTMDefault, self).__init__()

        self.dropout1 = nn.Dropout(0.2)
        self.lstm1 = nn.LSTM(
            input_size=8, hidden_size=20, batch_first=True, bidirectional=False
        )

        self.dropout2 = nn.Dropout(0.2)
        self.lstm2 = nn.LSTM(
            input_size=20, hidden_size=20, batch_first=True, bidirectional=False
        )

        self.dropout3 = nn.Dropout(0.2)
        self.lstm3 = nn.LSTM(
            input_size=20, hidden_size=10, batch_first=True, bidirectional=False
        )

        self.dropout4 = nn.Dropout(0.2)
        self.lstm4 = nn.LSTM(
            input_size=10, hidden_size=10, batch_first=True, bidirectional=False
        )

        self.fc = nn.Linear(10, 1)
        self.sigmoid = nn.Sigmoid()

    def __repr__(self):
        return "LSTMDefault"

    def forward(self, x):
        x = self.dropout1(x)
        x, _ = self.lstm1(x)

        x = self.dropout2(x)
        x, _ = self.lstm2(x)

        x = self.dropout3(x)
        x, _ = self.lstm3(x)

        x = self.dropout4(x)
        x, _ = self.lstm4(x)

        x = x[:, -1, :]  # Take the last time step

        x = self.fc(x)
        x = self.sigmoid(x)
        return x
