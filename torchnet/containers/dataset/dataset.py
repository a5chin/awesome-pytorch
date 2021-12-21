import pandas as pd
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
		self.data = data
		self.target = target
		self.ignore_features = ignore_features

	def __getitem__(self, index):
		data = self.data.drop(columns=self.ignore_features)
		return {column: data[column][index] for column in data.columns}, data[self.target][index]

	def __len__(self):
		return len(self.data)
