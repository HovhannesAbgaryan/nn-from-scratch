import torch.nn as nn
import torch
from typing import Sequence, Union


class Block(nn.Module):
    def __init__(self, input_size: int, output_size: int, use_batch_norm: bool = True, use_dropout: bool = False,
                 dropout_rate: float = 0.2,
                 activation: str = "relu"):

        if activation not in ["relu", "sigmoid"]:
            raise ValueError("activation must be either 'relu' or 'sigmoid'")

        super(Block, self).__init__()
        self.linear = nn.Linear(input_size, output_size)

        self.activation: Union[nn.ReLU, nn.Sigmoid] = nn.ReLU() if activation == "relu" else nn.Sigmoid()

        self.use_batch_norm = use_batch_norm
        self.use_dropout = use_dropout

        if self.use_batch_norm:
            self.batch_norm = nn.BatchNorm1d(output_size)
        if self.use_dropout:
            self.dropout = nn.Dropout(dropout_rate)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        out = self.activation(self.linear(X))
        if self.use_batch_norm:
            out = self.batch_norm(out)

        if self.use_dropout:
            out = self.dropout(out)
        return out


class HeartDiseaseDetector(nn.Module):
    def __init__(self, input_size: int = 25,
                 hidden_sizes: Sequence[int] | None = None,
                 output_size: int = 1,
                 use_batch_norm: bool = True,
                 use_dropout: bool = False,
                 dropout_rate: float = 0.2):
        super(HeartDiseaseDetector, self).__init__()
        self.layers = nn.ModuleList()

        if hidden_sizes is None:
            hidden_sizes = []

        if len(hidden_sizes) == 0:
            self.layers.append(Block(input_size, output_size, False, False, dropout_rate, activation="sigmoid"))
        else:
            self.layers.append(Block(input_size, hidden_sizes[0], use_batch_norm, use_dropout, dropout_rate))

            for i in range(1, len(hidden_sizes)):
                self.layers.append(
                    Block(hidden_sizes[i - 1], hidden_sizes[i], use_batch_norm, use_dropout, dropout_rate))

            self.layers.append(Block(hidden_sizes[-1], output_size, False, False, dropout_rate, activation="sigmoid"))

        # self._initialize_weights()

    def _initialize_weights(self):
        if len(self.layers) > 0:
            for layer in self.layers[:-1]:
                if isinstance(layer, nn.Linear):
                    nn.init.kaiming_normal_(layer.weight, nonlinearity='relu')
                    nn.init.constant_(layer.bias, 0)
            layer = self.layers[-1]
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_normal_(layer.weight, nonlinearity='sigmoid')
                nn.init.constant_(layer.bias, 0)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        out = X
        for layer in self.layers:
            out = layer(out)
        return out
