import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self, dim, hidden_dim, dropout):
        super().__init__()
        self.fc1  = nn.Linear(dim, hidden_dim, bias=False)
        self.fc2  = nn.Linear(hidden_dim, dim, bias=False)
        self.dropout1 = nn.Dropout(p=dropout)
        self.dropout2 = nn.Dropout(p=dropout)

    def forward(self, x):
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = self.dropout2(x)

        return x


class MixerLayer(nn.Module):
    def __init__(self, seq_len, num_features, d_c, d_s, dropout):
        super().__init__()
        self.mlp1 = MLP(seq_len, d_c, dropout)
        self.mlp2 = MLP(num_features, d_s, dropout)
        self.layernorm1 = nn.LayerNorm([seq_len, num_features])
        self.layernorm2 = nn.LayerNorm([seq_len, num_features])

    def forward(self, x):
        residual = x

        x = self.layernorm1(x)
        x = x.transpose(1, 2)
        x = self.mlp1(x)
        x = x.transpose(1, 2)

        x += residual
        residual = x

        x = self.layernorm2(x)
        x = self.mlp2(x)

        x += residual

        return x
