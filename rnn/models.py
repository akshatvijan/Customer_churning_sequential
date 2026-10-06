import torch.nn as nn


class ChurnRNN(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=1):
        super().__init__()
        self.rnn = nn.RNN(input_size, hidden_size, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):                    # x: (batch, months, features)
        output, hidden = self.rnn(x)         # output: (batch, months, hidden)
        return self.fc(output[:, -1, :])     # logits from the final month


class ChurnLSTM(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=1, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=num_layers, batch_first=True,
                            dropout=dropout if num_layers > 1 else 0.0)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        output, (hidden, cell) = self.lstm(x)    # hidden: (layers, batch, hidden)
        return self.fc(hidden[-1])               # last layer's hidden state
