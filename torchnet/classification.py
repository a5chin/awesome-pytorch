import pandas as pd
import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from tqdm import tqdm
from typing import List, Optional

from containers import Classifier, ClfDataset


class TorchNet:
	def train(
		self,
		model: nn.Module,
		lr: float=1e-3,
		optimizer: str='SGD',
		criterion: str='CrossEntropyLoss',
		total_epoch: int=20,
	):
		self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
		self.optimizer = eval(f'optim.{optimizer}')(model.parameters(), lr=lr)
		self.criterion = eval(f'nn.{criterion}')()

		for epoch in range(total_epoch):
			total, train_loss, train_acc = 0, 0.0, 0.0
			model.train()

			with tqdm(enumerate(self.train_dataloader, 0), total=len(self.train_dataloader)) as pbar:
				pbar.set_description(f'[Epoch {epoch + 1}/{total_epoch}]')

				for _, data in pbar:
					items, target = data
					item = items.values()

					self.optimizer.zero_grad()

					out = model(*item)
					loss = self.criterion(out, target)

					loss.backward()
					self.optimizer.step()

					preds = out.detach().numpy().argmax(axis=1)
					print(preds)

	def set_data(self, data, target, ignore_features=[], ratio=0.8, batch_size=32) -> None:
		dataset = ClfDataset(
			data=data,
			target=target,
			ignore_features=ignore_features,
		)
		total_data = len(dataset)
		len_train = int(total_data * ratio)
		len_val = total_data - len_train
		train_dataset, val_dataset = random_split(
			dataset=dataset,
			lengths=[len_train, len_val]
		)
		self.train_dataloader = DataLoader(
			dataset=train_dataset,
			batch_size=batch_size,
			shuffle=True,
			drop_last=True
		)
		self.val_dataloader = DataLoader(
			dataset=val_dataset,
			batch_size=1,
			shuffle=False
		)

	@staticmethod
	def create_model(
		layers: List,
		bn: str=True,
		act_fn: str='LeakyReLU',
		dropout: float=0.2,
		act_fin: str='Softmax',
		init_weights: bool=True
	) -> nn.Module:
		model = Classifier(
			layers=layers,
			act_fn=act_fn,
			dropout=dropout,
			act_fin=act_fin
		)
		if init_weights:
			model.apply(TorchNet.init_weights)
		return model


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


torchnet = TorchNet()
model = torchnet.create_model(layers=[6, 5, 4, 3, 2])
df = pd.read_csv('torchnet/data/train.csv')
torchnet.set_data(data=df, target='Survived', ignore_features=['Name', 'Sex', 'Age', 'Ticket', 'Cabin', 'Embarked'])
torchnet.train(model)