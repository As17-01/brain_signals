import torch.nn as nn


class LSTMConv48D2(nn.Module):
    def __init__(self):
        super(LSTMConv48D2, self).__init__()

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

        self.dropout1 = nn.Dropout(0.2)
        self.lstm1 = nn.LSTM(
            input_size=32, hidden_size=20, batch_first=True, bidirectional=False
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
        return "LSTMConv48D2"

    def forward(self, x):
        B, T, F = x.shape

        x = x.unsqueeze(1)
        x = self.conv1(x)
        x = nn.functional.relu(x)
        x = self.conv2(x)

        x = x.permute(0, 2, 1, 3)
        x = x.reshape(B, T, -1)

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
