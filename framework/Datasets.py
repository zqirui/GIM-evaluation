
from typing import Any, Callable

import torch
from torch.utils.data import Dataset
import torchvision.datasets as dset
from torchvision.datasets.folder import default_loader
from torchvision.transforms import transforms

class CustomDataset(dset.ImageFolder):
    """
    Custom dataset based on ImagesFolder only returning image tensors
    """
    def __init__(
        self,
        root: str,
        transform: Callable[..., Any] | None = None,
        target_transform: Callable[..., Any] | None = None,
        loader: Callable[[str], Any] = default_loader,
        is_valid_file: Callable[[str], bool] | None = None,
    ):
        super().__init__(root, transform, target_transform, loader, is_valid_file)

    def __getitem__(self, index: int) -> torch.Tensor.type:
        sample, _ = super().__getitem__(index)
        sample = transforms.PILToTensor()(sample)
        return sample
    
class SimpleDataset(Dataset):
    """
    Simple dataset based on Tensors only returning image tensors
    """
    def __init__(self, data: torch.Tensor.type):
        self.data = data

    def __getitem__(self, index) -> torch.Tensor.type:
        x = self.data[index]
        return x
    
    def __len__(self) -> int:
        return len(self.data)