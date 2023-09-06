from dataclasses import dataclass
from typing import Tuple, Union

from third_party.FID_IS_infinity.score_infinity import calculate_IS_infinity_path

from metrics.metrics_base import MetricsBase
from framework.configs import EvalConfig, PlatformConfig

@dataclass
class IS_infty(MetricsBase):
    """
    IS infinity Metric
    """
    eval_config: EvalConfig = None
    platform_config: PlatformConfig = None
    generated_img_path: str = None

    def calculate(self) -> float | Tuple[float, float]:
        is_infty = calculate_IS_infinity_path(
            path=self.generated_img_path,
            batch_size=self.eval_config.is_infty_batch_size,
            feature_extractor=self.feature_extractor_flag,
        )

        return is_infty