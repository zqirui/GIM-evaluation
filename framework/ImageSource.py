from dataclasses import dataclass
from typing import Any, Optional, Sequence, Callable, Tuple

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.datasets as dset
from torchvision.datasets.folder import default_loader
from torchvision.transforms import transforms


class CustomDataset(dset.ImageFolder):
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
 

@dataclass
class ImageSource:
    """
    Image Source
    """

    folder_path: str
    source_name: str
    # Optional Image transforms
    transforms: Optional[Sequence[Callable]] = None
    # Optionally initialize via pytorch Dataset
    dataset: Optional[Dataset] = None

    def __post_init__(self):
        self.dataset = (
            CustomDataset(
                root=self.folder_path,
                transform=self.transforms if self.transforms is not None else None,
            )
            if self.dataset is None
            else self.dataset
        )

    def get_dataset(self) -> Dataset:
        """
        Return pytorch dataset (or derivatives)
        """
        return self.dataset

    def get_dataloader(
        self, batch_size: int = 64, shuffle: bool = True, num_worker: int = 0
    ) -> DataLoader:
        """
        Return pytorch dataloader
        """
        dataloader = DataLoader(self.dataset, batch_size, shuffle, num_worker)
        return dataloader
