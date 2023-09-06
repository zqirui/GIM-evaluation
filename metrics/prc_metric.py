from dataclasses import dataclass
from typing import Tuple

import torch
from torch.utils.data import Dataset, RandomSampler, DataLoader
import torch_fidelity
from torch_fidelity.metric_prc import KEY_METRIC_PRECISION, KEY_METRIC_RECALL, KEY_METRIC_F_SCORE

from metrics.metrics_base import MetricsBase
from framework.configs import EvalConfig, PlatformConfig, FeatureExtractor
from framework.datasets import SimpleDataset
from framework.downsampler import Downsampler
from framework.feature_extractor.vggface_torch_fidelity import VGGFaceFETorchFidelityWrapper

@dataclass
class PRC(MetricsBase):
    """
    Improved Precision Recall (Alpha/Beta) Score 
    """
    eval_config: EvalConfig = None
    platform_config: PlatformConfig = None
    real_img: Dataset = None
    generated_img: Dataset = None
    num_samples: int = 50000
    feature_extractor: str = None
    
    def __post_init__(self):
        assert self.real_img is not None, "Empty real dataset!"
        assert self.generated_img is not None, "Empty generated dataset"
        if len(self.generated_img) == self.num_samples:
            pass
        else:
            downsampler = Downsampler(full_data=self.generated_img, target_size=self.num_samples, shuffle=True)
            self.generated_img = downsampler.downsample()
        # get dataset of real image of same sample size
        if len(self.real_img) == self.num_samples:
            self.real_img_downsampled = self.real_img
        else:
            downsampler = Downsampler(full_data=self.real_img, target_size=self.num_samples, shuffle=True)
            self.real_img_downsampled = downsampler.downsample()

        if self.feature_extractor_flag == FeatureExtractor.VGGFaceResNet50:
            self.feature_extractor = VGGFaceFETorchFidelityWrapper.get_default_name()
        else:
            self.feature_extractor = None

    def calculate(self) -> float | Tuple[float, float, float]:
        assert len(self.real_img_downsampled) == len(self.generated_img), "Different sample sizes of real and generated images!"
        metric_dict = torch_fidelity.calculate_metrics(
            input1=self.real_img_downsampled,
            input2=self.generated_img,
            cuda=self.platform_config.cuda,
            feature_extractor=self.feature_extractor,
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
    
    def get_real_subsampled_imgs(self) -> Dataset:
        """
        Return subsampled real dataset
        """
        return self.real_img_downsampled

