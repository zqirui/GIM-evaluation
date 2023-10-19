from dataclasses import dataclass
from typing import Tuple

import torch
from third_party.FID_IS_infinity.score_infinity import calculate_FID_infinity_path

from metrics.metrics_base import MetricsBase
from framework.configs import EvalConfig, PlatformConfig

@dataclass
class FID_infty(MetricsBase):
    """
    FID infinity Metric
    """
    eval_config: EvalConfig = None
    platform_config: PlatformConfig = None
    real_img_path: str = None
    precomputed_real_features: torch.Tensor.type = None
    precomputed_fake_features: torch.Tensor.type = None
    generated_img_path: str = None

    def calculate(self) -> float | Tuple[float, float]:
        fid_infty, self.precomputed_real_features = calculate_FID_infinity_path(
            real_path=self.real_img_path,
            fake_path=self.generated_img_path,
            batch_size=self.eval_config.fid_infty_batch_size,
            feature_extractor=self.feature_extractor_flag,
            real_features=self.precomputed_real_features,
            fake_features=self.precomputed_fake_features,
            rtn_real_feat=True,
        )
        return fid_infty
    
    def get_real_features(self):
        """
        Return precomputed real features
        """
        return self.precomputed_real_features