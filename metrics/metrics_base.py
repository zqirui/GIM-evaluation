from dataclasses import dataclass
from typing import Optional, Union, Tuple

from framework.feature_extractor.feature_extractor_base import FeatureExtractorBase
from framework.configs import FeatureExtractor


@dataclass
class MetricsBase:
    """
    Metrics Base Module
    """

    # Metric Name
    name: str
    # Optional Feature Extractor
    feature_extractor: Optional[FeatureExtractorBase] = None
    # Optional Feature Extractor Flag
    feature_extractor_flag : Optional[FeatureExtractor] = FeatureExtractor.InceptionV3

    def calculate(self) -> Union[float, Tuple[float, float]]:
        """
        Main Calculation Method
        """
        raise NotImplementedError()
