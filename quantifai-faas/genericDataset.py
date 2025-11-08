import torch
from torch.utils.data import Dataset
from typing import Tuple, Optional
import logging


class GenericDataset(Dataset):
    def __init__(self, data_file: str, shape: Optional[Tuple[int, ...]] = None):
        dataset = torch.load(data_file)
        self.data = dataset["data"]
        self.labels = dataset["labels"]

        # Convert shape to a tuple if it is a torch.Size object
        if isinstance(shape, torch.Size):
            self.input_format = tuple(shape)  # Convert torch.Size to tuple
            logging.info(f"Converted shape to tuple: {self.input_format}")
        else:
            self.input_format = shape

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        data, label = self.data[idx], self.labels[idx]

        if self.input_format:
            try:
                data = data.view(*self.input_format)
            except Exception as e:
                print(f"Error reshaping data: {e}")
                print(f"Current data shape: {data.shape}, expected shape: {self.input_format}")
                raise

        return data, label
