import pandas as pd
import torch
from torch import nn, optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from pathlib import Path
from typing import Optional, List, OrderedDict

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
		scheduler: str='OneCycleLR',
		criterion: str='NLLLoss',
		total_epoch: int=20,
	) -> nn.Module:
		"""Train the model.

		Args:
			model (nn.Module): the model for training.
			lr (float, optional): learning rate. Defaults to 1e-3.
			optimizer (str, optional): A type of optimizers. Defaults to 'SGD'.
			scheduler (str, optional): A types of schedulers. Defaults to 'OneCycleLR'.
			criterion (str, optional): A types of loss functions. Defaults to 'NLLLoss'.
			total_epoch (int, optional): Number of epochs. Defaults to 20.

		Returns:
			nn.Module: The trained model.

		Example::
			>>> torchnet = TorchNet()
			>>> model = torchnet.create_model(layers=[5, 32, 256, 1024, 256, 32, 8, 2])
			>>> df = pd.read_csv('assets/data/train.csv')
			>>> torchnet.set_data(
			>>> 	data=df,
			>>> 	target='Survived',
			>>> 	ignore_features=['PassengerId', 'Name', 'Sex', 'Age', 'Ticket', 'Cabin', 'Embarked']
			>>> )
			>>> trained_model = torchnet.train(model, total_epoch=100)
		"""
		self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
		self.optimizer = eval(f'optim.{optimizer}')(model.parameters(), lr=lr)
		self.scheduler = eval(f'lr_scheduler.{scheduler}')(
			optimizer=self.optimizer,
			max_lr=lr,
			total_steps=len(self.train_dataloader),
		)
		self.criterion = eval(f'nn.{criterion}')()

		for epoch in range(total_epoch):
			total, train_loss = 0, 0.0
			accuracy = recall = precision = 0.0
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
						OrderedDict(
							Loss=train_loss / total,
							Accuracy=accuracy / total,
							Recall=recall / total,
							Precision=precision / total
						)
					)

				self.train_writer.add_scalar('loss', train_loss / total, epoch)
				self.train_writer.add_scalar('accuracy', accuracy / total, epoch)
				self.train_writer.add_scalar('recall', recall / total, epoch)
				self.train_writer.add_scalar('precision', precision / total, epoch)

			self.evaluate(model, epoch)

			self.scheduler.step()

		model.load_state_dict(
			torch.load(self.ckpt / 'best_ckpt.pth'), strict=False
		)
		return model

	def evaluate(
		self,
		model: nn.Module,
		dataloader: DataLoader=None,
		epoch: Optional[int]=None
	) -> None:
		"""Evaluate the model.

		Args:
			model (nn.Module): The trained model.
			dataloader (DataLoader, optional): DataLoader for evaluating. Defaults to None.
			epoch (Optional[int], optional): Number of epochs to record. Defaults to None.
		"""
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
					torch.save(model.state_dict(), self.ckpt / 'best_ckpt.pth')

	def set_data(
		self,
		data: pd.DataFrame,
		target: str,
		ignore_features: List=[],
		ratio: float=0.8,
		batch_size: int=32
	) -> None:
		"""Set data to be used to train and evaluate the model.

		Args:
			data (pd.DataFrame): Data to be used.
			target (str): Data to be estimated.
			ignore_features (List, optional): Data to be ignore. Defaults to [].
			ratio (float, optional): Percentage of train data in total. Defaults to 0.8.
			batch_size (int, optional): batch size used for training. Defaults to 32.

		Example::
			>>> torchnet = TorchNet()
			>>> df = pd.read_csv('assets/data/train.csv')
			>>> torchnet.set_data(
			>>> 	data=df,
			>>> 	target='Survived',
			>>> 	ignore_features=['PassengerId', 'Name', 'Sex', 'Age', 'Ticket', 'Cabin', 'Embarked']
			>>> )
		"""
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
			shuffle=False,
			drop_last=True
		)

	@staticmethod
	def create_model(
		layers: List,
		act_fn: str='LeakyReLU',
		dropout: float=0.2,
		act_fin: str='LogSoftmax',
		init_weights: bool=True
	) -> nn.Module:
		"""Create the model.

		Args:
			layers (List): Number of neurons per layer.
			act_fn (str, optional): A type of activate functions. Defaults to 'LeakyReLU'.
			dropout (float, optional): Probability to dropout. Defaults to 0.2.
			act_fin (str, optional): Output layer functions. Defaults to 'LogSoftmax'.
			init_weights (bool, optional): Whether to initialize weights or not. Defaults to True.

		Returns:
			nn.Module: Created model.

		Example::
			>>> torchnet = TorchNet()
			>>> model = torchnet.create_model(layers=[5, 32, 256, 1024, 256, 32, 8, 2])
		"""
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
		"""Initialize model weights. Defaults to True.

		Args:
			m (nn.Module): The model.
		"""
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
