from typing import List, OrderedDict, Tuple

import torch
import torch.nn as nn


class Classifier(nn.Module):
    def __init__(
        self,
        layers: List,
        act_fn: str = "LeakyReLU",
        dropout: float = 0.2,
        act_fin: str = "Softmax",
    ) -> None:
        super().__init__()

        self.in_features = layers[0]
        self.num_classes = layers[-1]
        self.dropout = dropout
        self.act_fn = act_fn
        self.act_fin = eval(f"nn.{act_fin}")(dim=-1)

        in_features, out_features = layers[:-2], layers[1:-1]
        unit = [
            Unit(i, o, self.dropout, act_fn) for i, o in zip(in_features, out_features)
        ]

        self.fc = nn.Sequential(*unit)
        self.fc.add_module("fc_classes", nn.Linear(layers[-2], self.num_classes))

    def forward(self, *x: Tuple) -> torch.Tensor:
        x = torch.stack(x).type(torch.FloatTensor)
        x = x.view(-1, self.in_features)
        x = self.fc(x)
        return self.act_fin(x)


class Unit(nn.Module):
    def __init__(
        self, in_features: int, out_features: int, dropout: float, act_fn: str
    ) -> None:
        super().__init__()
        self.fc = nn.Sequential(
            OrderedDict(
                [
                    ("fc", nn.Linear(in_features, out_features)),
                    ("bn", nn.BatchNorm1d(out_features)),
                    ("act_fn", eval(f"nn.{act_fn}")()),
                    ("dropout", nn.Dropout(dropout)),
                ]
            )
        )

    def forward(self, x: torch.Tensor):
        return self.fc(x)
