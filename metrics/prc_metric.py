from dataclasses import dataclass
from typing import Tuple

import torch
from torch.utils.data import Dataset, RandomSampler, DataLoader
import torch_fidelity
from torch_fidelity.metric_prc import KEY_METRIC_PRECISION, KEY_METRIC_RECALL, KEY_METRIC_F_SCORE

from metrics.MetricsBase import MetricsBase
from framework.Configs import EvalConfig, PlatformConfig
from framework.Datasets import SimpleDataset

@dataclass
class PRC(MetricsBase):
    """
    Improved Precision Recall (Alpha/Beta) Score 
    """
    eval_config: EvalConfig = None
    platform_config: PlatformConfig = None
    real_img: Dataset = None
    generated_img: Dataset = None
    
    def __post_init__(self):
        assert self.real_img is not None, "Empty real dataset!"
        assert self.generated_img is not None, "Empty generated dataset"
        # get dataset of real image of same sample size
        if len(self.real_img) == len(self.generated_img):
            self.real_img_downsampled = self.real_img
        else:
            sample_size = len(self.generated_img)
            self.real_img_downsampled = self._subsample_real_imgs(sample_size)

    def calculate(self) -> float | Tuple[float, float, float]:
        assert len(self.real_img_downsampled) == len(self.generated_img), "Different sample sizes of real and generated images!"
        metric_dict = torch_fidelity.calculate_metrics(
            input1=self.real_img_downsampled,
            input2=self.generated_img,
            cuda=self.platform_config.cuda,
            prc=True,
            prc_neighborhood=self.eval_config.prc_neighborhood,
            prc_batch_size=self.eval_config.prc_batch_size,
            verbose=self.platform_config.verbose,
            save_cpu_ram=self.platform_config.save_cpu_ram,
        )
        return (
            metric_dict[KEY_METRIC_PRECISION],
            metric_dict[KEY_METRIC_RECALL],
            metric_dict[KEY_METRIC_F_SCORE],
        )
    
    def _subsample_real_imgs(self, n : int) -> Dataset:
        """
        Create a sub-dataset of same size of generated images by uniform sampling
        """
        # init uniform sampler to get subset
        uniform_sampler = RandomSampler(self.real_img, num_samples=n)
        dataloader = DataLoader(dataset=self.real_img, batch_size=128, sampler=uniform_sampler)
        # gather and init new dataset
        samples = []
        for sample in dataloader:
            samples.append(sample)
        samples = torch.vstack(samples)
        assert len(samples) == n, "[ERROR]: Mismatch in sample size during subsampling for PRC!"
        dataset = SimpleDataset(samples)
        return dataset

