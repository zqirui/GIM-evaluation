from dataclasses import dataclass
import torch

@dataclass
class FeatureExtractorBase():
    """
    Feature Extractor Base class
    """
    name: str = ''

    def extract(self, samples : torch.Tensor.type) -> torch.Tensor.type:
        """
        Feature extraction method
        """
        raise NotImplementedError()