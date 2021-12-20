from torch import nn
from typing import List

from containers import Classifier


class TorchNet:
    def create_model(
        self,
        layers: List,
		act_fn: str='LeakyReLU',
		dropout: float=0.2,
		act_fin: str='Softmax',
        init_weights: bool=True
    ) -> nn.Module:
        self.model = Classifier(
            layers=layers,
            act_fn=act_fn,
            dropout=dropout,
            act_fin=act_fin
        )
        self.model.apply(TorchNet.init_weights)
        return self.model

    @staticmethod
    def init_weights(m) -> None:
        classname = m.__class__.__name__
        if classname.find('Linear') != -1 or classname.find('Bilinear') != -1:
            nn.init.kaiming_uniform_(a=2, mode='fan_in', nonlinearity='leaky_relu', tensor=m.weight)
            if m.bias is not None: nn.init.zeros_(tensor=m.bias)

        elif classname.find('Conv') != -1:
            nn.init.kaiming_uniform_(a=2, mode='fan_in', nonlinearity='leaky_relu', tensor=m.weight)
            if m.bias is not None: nn.init.zeros_(tensor=m.bias)

        elif classname.find('BatchNorm') != -1 or classname.find('GroupNorm') != -1 or classname.find('LayerNorm') != -1:
            nn.init.uniform_(a=0, b=1, tensor=m.weight)
            nn.init.zeros_(tensor=m.bias)

        elif classname.find('Cell') != -1:
            nn.init.xavier_uniform_(gain=1, tensor=m.weiht_hh)
            nn.init.xavier_uniform_(gain=1, tensor=m.weiht_ih)
            nn.init.ones_(tensor=m.bias_hh)
            nn.init.ones_(tensor=m.bias_ih)

        elif classname.find('RNN') != -1 or classname.find('LSTM') != -1 or classname.find('GRU') != -1:
            for w in m.all_weights:
                nn.init.xavier_uniform_(gain=1, tensor=w[2].data)
                nn.init.xavier_uniform_(gain=1, tensor=w[3].data)
                nn.init.ones_(tensor=w[0].data)
                nn.init.ones_(tensor=w[1].data)

        elif classname.find('Embedding') != -1:
            nn.init.kaiming_uniform_(a=2, mode='fan_in', nonlinearity='leaky_relu', tensor=m.weight)
