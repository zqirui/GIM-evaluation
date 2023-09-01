from dataclasses import dataclass
from typing import Union

import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import RandomSampler, SequentialSampler, Sampler

from framework.datasets import SimpleDataset

@dataclass
class Downsampler:
    """
    Downsampler for subsampling
    """
    full_data : Dataset
    target_size : int
    sampler : Sampler = None
    batch_size : int = 128
    num_worker : int = 0
    shuffle : bool = False
    shuffle_replacement : bool = False

    def __post_init__(self):
        assert self.target_size <= len(self.full_data), "Target size bigger than full dataset length!"
        if self.shuffle:
            self.sampler = RandomSampler(self.full_data, replacement=self.shuffle_replacement, num_samples=self.target_size)
        else:
            self.sampler = SequentialSampler(self.full_data)

    def downsample(self,  return_tensor : bool = False) -> Union[Dataset,torch.Tensor.type]:
        """
        Downsample 
        """
        loader = DataLoader(dataset=self.full_data, batch_size=self.batch_size, sampler=self.sampler, num_workers=self.num_worker)
        samples = self._accumulater_samples(loader)
        if not self.shuffle and self.target_size < len(self.full_data):
            samples = samples[:self.target_size]
        if return_tensor:
            return samples
        reduced_dataset = SimpleDataset(samples)
        return reduced_dataset
    
    def _accumulater_samples(self, dl : DataLoader) -> torch.Tensor.type:
        samples = []
        for sample in dl:
            samples.append(sample)
        samples = torch.vstack(samples)
        return samples

