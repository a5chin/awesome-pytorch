from torch import nn
from typing import List

from containers import Classifier


class TorchNet:
    def create_model(
        self,
        layers: List,
		act_fn: str='LeakyReLU',
		dropout: float=0.2,
		act_fin: str='Softmax'
    ) -> nn.Module:
        self.model = Classifier(
            layers=layers,
            act_fn=act_fn,
            dropout=dropout,
            act_fin=act_fin
        )
        return self.model
