from dataclasses import dataclass
from typing import Optional, Sequence, Callable


from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SequentialSampler, RandomSampler

from framework.datasets import CustomImageDataset

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
            CustomImageDataset(
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
        self, batch_size: int = 64, shuffle: bool = True, num_worker: int = 0, subsample : bool = False, subsample_n : int = 50000
    ) -> DataLoader:
        """
        Return pytorch dataloader
        """
        if not shuffle:
            dataloader = DataLoader(dataset=self.dataset, batch_size=batch_size, num_workers=num_worker, sampler=SequentialSampler(self.dataset))
        else:
            dataloader = DataLoader(dataset=self.dataset, batch_size=batch_size, num_workers=num_worker, sampler=RandomSampler(self.dataset, num_samples=len(self.dataset) if not subsample else subsample_n))
        return dataloader
