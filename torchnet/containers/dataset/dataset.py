import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
from typing import Union, List


class ClfDataset(Dataset):
	def __init__(
		self,
		data: pd,
		target: str,
		ignore_features: List=[],
	):
		super().__init__()
		targets = data.groupby(target)
		data[target] = targets.ngroup()
		self.num_classes = targets.ngroups
		self.data = data.drop(columns=ignore_features)
		self.target = target
		self.targets = data[target]
		self.ignore_features = ignore_features

	def __getitem__(self, index):
		data = self.data
		target = self.targets[index]
		target = torch.as_tensor(target)
		target = F.one_hot(target, num_classes=self.num_classes)
		return {column: data[column][index] for column in data.columns}, target.type(torch.FloatTensor)

	def __len__(self):
		return len(self.data)
