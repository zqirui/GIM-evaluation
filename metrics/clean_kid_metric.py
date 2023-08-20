from dataclasses import dataclass
from typing import Tuple, Union

from cleanfid import fid as clean_metric

from metrics.metrics_base import MetricsBase
from framework.configs import EvalConfig, PlatformConfig

@dataclass
class CleanKID(MetricsBase):
    """
    Clean FID Metric
    """
    eval_config: EvalConfig = None
    platform_config: PlatformConfig = None
    real_img_path: str = None
    generated_img_path: str = None

    def calculate(self) -> float | Tuple[float, float]:
        clean_kid = clean_metric.compute_kid(
            fdir1=self.real_img_path,
            fdir2=self.generated_img_path,
            batch_size=self.eval_config.clean_kid_batch_size,
            num_workers=0 # >0 throws pickle error in multiprocessing
        )
        return clean_kid