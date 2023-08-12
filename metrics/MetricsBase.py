from dataclasses import dataclass
from typing import Optional

from framework.FeatureExtractorBase import FeatureExtractorBase

@dataclass
class MetricsBase():
    """
    Metrics Base Module 
    """
    # Metric Name
    name: str
    # Optional Feature Extractor 
    feature_extractor: Optional[FeatureExtractorBase]

    def calculate(self) -> float:
        """
        Main Calculation Method
        """
        raise NotImplementedError()

