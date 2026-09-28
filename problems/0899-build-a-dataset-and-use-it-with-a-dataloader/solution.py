import torch
from torch.utils.data import Dataset


class TensorPairDataset(Dataset):

    def __init__(self, X: torch.Tensor, y: torch.Tensor):
        '''store X and y on the instance'''

        self.X = X
        self.y = y

    def __len__(self):
        '''return the number of samples'''

        return self.X.shape[0]

    def __getitem__(self, idx: int):
        '''return the (x, y) pair at idx'''

        return self.X[idx], self.y[idx]
