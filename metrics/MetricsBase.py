from dataclasses import dataclass
from typing import Optional, Union, Tuple

from framework.FeatureExtractorBase import FeatureExtractorBase


@dataclass
class MetricsBase:
    """
    Metrics Base Module
    """

    # Metric Name
    name: str
    # Optional Feature Extractor
    feature_extractor: Optional[FeatureExtractorBase] = None

    def calculate(self) -> Union[float, Tuple[float, float]]:
        """
        Main Calculation Method
        """
        raise NotImplementedError()
