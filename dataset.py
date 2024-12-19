import torch as T

class Dataset(T.utils.data.Dataset):
  def __init__(self,tensor):
    self.tensor = tensor

  def __getitem__(self,index):
    return self.tensor[index]

  def __len__(self):
    return self.tensor.size(0)
