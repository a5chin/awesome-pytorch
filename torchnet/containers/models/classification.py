import torch
import torch.nn as nn

from typing import List


class Classifier(nn.Module):
	def __init__(
		self,
		layers: List,
		act_fn: str='LeakyReLU',
		bn: str=True,
		dropout: float=0.2,
		act_fin: str='Softmax'
	) -> None:
		super().__init__()

		self.in_features = layers[0]
		self.num_classes = layers[-1]
		self.bn = True
		self.dropout = dropout
		self.act_fn = act_fn
		self.act_fin = eval(f'nn.{act_fin}')(dim=-1)

		self.fc = nn.Sequential()
		for i in range(len(layers) - 2):
			self.fc.add_module(
				f'fc{i}',
				nn.Linear(layers[i], layers[i + 1])
			)
			if self.bn:
				self.fc.add_module(
					f'bn{i}',
					nn.BatchNorm1d(layers[i + 1])
				)
			self.fc.add_module(
				f'act_fn{i}',
				eval(f'nn.{act_fn}')()
			)
			self.fc.add_module(
				f'dropout{i}',
				nn.Dropout(self.dropout)
			)
		self.fc.add_module(
			f'fc{i + 1}',
			nn.Linear(layers[-2], layers[-1])
		)

	def forward(self, *x):
		x = torch.stack(x).type(torch.FloatTensor)
		x = x.view(-1, self.in_features)
		x = self.fc(x)
		return self.act_fin(x)
