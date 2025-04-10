import pennylane as qml
import torch
import torch.nn as nn

num_qubit = 4
dev = qml.device('default.qubit', wires=num_qubit)


class CNNExtract(nn.Module):
    def __init__(self, in_channels=1, out_channels=16):
        super(CNNExtract, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(8, out_channels, kernel_size=3, padding=1),
            nn.ReLU()
        )

    def forward(self, x):
        feat = self.conv(x)
        feat = feat.mean(dim=[2, 3])
        return feat


class CNNLSTMPolicy(nn.Module):
    def __init__(self, feature_dim=16, hidden_dim=32, output_dim=100, num_layers=1):
        super(CNNLSTMPolicy, self).__init__()

        self.cnn_extractor = CNNExtract(in_channels=1, out_channels=feature_dim)

        self.lstm = nn.LSTM(input_size=feature_dim,
                            hidden_size=hidden_dim,
                            num_layers=num_layers,
                            batch_first=True)

        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        batch_size, seq_len, h, w = x.shape
        x = x.view(batch_size * seq_len, 1, h, w)

        feat = self.cnn_extractor(x)
        feat = feat.view(batch_size, seq_len, -1)

        out, (h_n, c_n) = self.lstm(feat)
        last_output = h_n[-1]
        output = self.fc(last_output)
        return output


class CNNLSTMValue(nn.Module):
    """
    Optional for REINFORCE w Baseline
    """

    def __init__(self, feature_dim=16, hidden_dim=32):
        super(CNNLSTMValue, self).__init__()

        # 여기서는 간단히 CNNExtract -> LSTM -> FC(1)
        self.cnn_extractor = CNNExtract(in_channels=1, out_channels=feature_dim)

        self.lstm = nn.LSTM(input_size=feature_dim,
                            hidden_size=hidden_dim,
                            batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)  # scalar output

    def forward(self, x):
        batch_size, seq_len, h, w = x.shape
        x = x.view(batch_size * seq_len, 1, h, w)

        feat = self.cnn_extractor(x)
        feat = feat.view(batch_size, seq_len, -1)

        out, (h_n, c_n) = self.lstm(feat)
        last_output = h_n[-1]
        value = self.fc(last_output)  # shape: [batch_size, 1]
        return value


class GRUPolicy(nn.Module):
    def __init__(self, num_layers=64, embedding_dim=32, hidden_dim=64):
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings=num_layers, embedding_dim=embedding_dim)
        self.gru = nn.GRU(input_size=embedding_dim, hidden_size=hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, num_layers)  # 각 레이어 인덱스(0~63)에 대한 logit 출력

    def forward(self, layer_seq):
        """
        layer_seq: List of layer indices (length T)
        Output: logits of shape [1, num_layers]
        """
        if len(layer_seq) == 0:
            # 첫 스텝엔 zero vector로 시작 (학습 가능하도록)
            dummy_input = torch.zeros(1, 1, self.embedding.embedding_dim).to(self.fc.weight.device)
            _, h = self.gru(dummy_input)
        else:
            input_seq = torch.tensor(layer_seq, dtype=torch.long).unsqueeze(0).to(self.fc.weight.device)
            embedded = self.embedding(input_seq)  # [1, T, D]
            _, h = self.gru(embedded)  # h: [1, 1, hidden_dim]

        logits = self.fc(h.squeeze(0))  # [1, num_layers]
        return logits


class GRUValue(nn.Module):
    def __init__(self, num_layers=64, embedding_dim=32, hidden_dim=64):
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings=num_layers, embedding_dim=embedding_dim)
        self.gru = nn.GRU(input_size=embedding_dim, hidden_size=hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, layer_seq):
        if len(layer_seq) == 0:
            dummy_input = torch.zeros(1, 1, self.embedding.embedding_dim).to(self.fc.weight.device)
            _, h = self.gru(dummy_input)
        else:
            input_seq = torch.tensor(layer_seq, dtype=torch.long).unsqueeze(0).to(self.fc.weight.device)
            embedded = self.embedding(input_seq)
            _, h = self.gru(embedded)

        value = self.fc(h.squeeze(0))  # shape: [1, 1]
        return value
