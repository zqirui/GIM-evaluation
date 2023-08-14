from dataclasses import dataclass
from typing import Tuple, Union

from torch.utils.data import Dataset

from metrics.MetricsBase import MetricsBase
from metrics.IS_FID_KID.is_fid_kid import IsFidKidBase

@dataclass
class IS(MetricsBase):
    """
    Inception Score Metric
    """
    
    inception_base: IsFidKidBase = None
    generated_img: Dataset = None

    def calculate(self) -> float | Tuple[float, float]:
        is_mean, is_std = self.inception_base.get_Is(self.generated_img)
        return is_mean, is_std
    

@dataclass
class FID(MetricsBase):
    """
    Inception Score Metric
    """
    
    inception_base: IsFidKidBase = None
    real_img: Dataset = None
    generated_img: Dataset = None

    def calculate(self) -> float | Tuple[float, float]:
        return self.inception_base.get_Fid(self.real_img, self.generated_img)
    
@dataclass
class KID(MetricsBase):
    """
    Inception Score Metric
    """
    
    inception_base: IsFidKidBase = None
    real_img: Dataset = None
    generated_img: Dataset = None

    def calculate(self) -> float | Tuple[float, float]:
        kid_mean, kid_std = self.inception_base.get_Kid(self.real_img, self.generated_img)
        return kid_mean, kid_std
    