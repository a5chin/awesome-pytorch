import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F
from torch.utils.data import Dataset
from typing import Tuple, List


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
		data = data.drop(columns=ignore_features)
		self.data = self.normalize(data)
		self.target = target
		self.targets = data[target]
		self.ignore_features = ignore_features

	# TODO: Normalizations
	def normalize(self, data: pd.DataFrame) -> pd.DataFrame:
		for name in data:
			d = data[name][0]
			if isinstance(d, np.float32) or isinstance(d, str):
				numeric = True
			else:
				numeric = False
		return data

	def __getitem__(self, index: int) -> Tuple[pd.DataFrame, torch.FloatTensor]:
		data = self.data
		target = self.targets[index]
		target = torch.as_tensor(target)
		target = F.one_hot(target, num_classes=self.num_classes)
		return {column: data[column][index] for column in data.columns}, target.type(torch.FloatTensor)

	def __len__(self) -> int:
		return len(self.data)
