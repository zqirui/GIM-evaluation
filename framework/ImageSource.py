from dataclasses import dataclass
from typing import Any, Optional, Sequence, Callable


from torch.utils.data import Dataset, DataLoader

from framework.Datasets import CustomDataset

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
