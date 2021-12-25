import re
import pandas as pd
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from collections import OrderedDict
from pathlib import Path
from typing import Optional, List

from containers import Classifier, ClfDataset
from containers import Metrics


class TorchNet:
	def __init__(self) -> None:
		self.metrics = Metrics()
		self.log_dir = Path('logs')
		self.ckpt = Path('ckpt')
		self.ckpt.mkdir(exist_ok=True)
		self.best_acc = 0.0
		self.train_writer = SummaryWriter(log_dir=self.log_dir / 'train')
		self.val_writer = SummaryWriter(log_dir=self.log_dir / 'val')

	def train(
		self,
		model: nn.Module,
		lr: float=1e-3,
		optimizer: str='SGD',
		criterion: str='CrossEntropyLoss',
		total_epoch: int=20,
	) -> None:
		self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
		self.optimizer = eval(f'optim.{optimizer}')(model.parameters(), lr=lr)
		self.criterion = eval(f'nn.{criterion}')()

		for epoch in range(total_epoch):
			total, train_loss = 0, 0.0
			accuracy, recall, precision = 0.0, 0.0, 0.0
			model.train()

			with tqdm(enumerate(self.train_dataloader, 0), total=len(self.train_dataloader)) as pbar:
				pbar.set_description(f'[Epoch {epoch + 1}/{total_epoch}]')

				for _, data in pbar:
					items, target = data
					items = items.values()

					self.optimizer.zero_grad()

					preds = model(*items)
					loss = self.criterion(preds, target)

					loss.backward()
					self.optimizer.step()

					total += self.batch_size
					train_loss += loss.item() * self.batch_size
					accuracy += self.metrics.accuracy(preds, target) * self.batch_size
					recall += self.metrics.recall(preds, target) * self.batch_size
					precision += self.metrics.precision(preds, target) * self.batch_size

					pbar.set_postfix(
						OrderedDict(Loss=train_loss / total, Accuracy=accuracy / total, Recall=recall / total, Precision=precision / total)
					)

				self.train_writer.add_scalar('loss', train_loss / total, epoch)
				self.train_writer.add_scalar('accuracy', accuracy / total, epoch)
				self.train_writer.add_scalar('recall', recall / total, epoch)
				self.train_writer.add_scalar('precision', precision / total, epoch)

			self.evaluate(model=model, epoch=epoch)

	def evaluate(self, model: nn.Module, dataloader: DataLoader=None, epoch: Optional[int]=None) -> None:
		accuracy, recall, precision = 0.0, 0.0, 0.0
		dataloader = self.val_dataloader if dataloader == None else dataloader
		model.eval()
		with torch.inference_mode():
			for data in dataloader:
				items, target = data
				items = items.values()
				preds = model(*items)

				accuracy = self.metrics.accuracy(preds, target)
				recall = self.metrics.recall(preds, target)
				precision = self.metrics.precision(preds, target)

			if epoch != None:
				self.train_writer.add_scalar('accuracy', accuracy, epoch)
				self.train_writer.add_scalar('recall', recall, epoch)
				self.train_writer.add_scalar('precision', precision, epoch)

				if self.best_acc < accuracy:
					self.best_acc = accuracy
					torch.save(model.state_dict(), self.ckpt / Path('best_ckpt.pth'))

	def set_data(self, data, target, ignore_features=[], ratio=0.8, batch_size=32) -> None:
		self.batch_size = batch_size
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
			batch_size=len_val,
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
model = torchnet.create_model(layers=[5, 4, 3, 2])
df = pd.read_csv('torchnet/data/train.csv')
torchnet.set_data(data=df, target='Survived', ignore_features=['PassengerId', 'Name', 'Sex', 'Age', 'Ticket', 'Cabin', 'Embarked'])
torchnet.train(model, total_epoch=50)