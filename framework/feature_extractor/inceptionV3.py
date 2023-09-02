from dataclasses import dataclass

import torch
from torchvision.models.inception import inception_v3, Inception_V3_Weights
from third_party.FID_IS_infinity.inception import WrapInception

from framework.feature_extractor.feature_extractor_base import FeatureExtractorBase


@dataclass
class InceptionV3FE(FeatureExtractorBase):
    """
    InceptionV3FE Feature Extractor
    """
    model : WrapInception = None
    device : torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # get features from last pooling before logits
    last_pool : bool = False 
    def __post_init__(self):
        self.name = "InceptionV3"
        if self.model is None:
            # load pre-trained InceptionV3
            inception = inception_v3(weights=Inception_V3_Weights.IMAGENET1K_V1, transform_input=False)
            self.model = WrapInception(inception.eval()).to(self.device)

    def extract(self, samples: torch.Tensor.type) -> torch.Tensor.type:
        assert len(samples.size()) == 4, "input not size (batch, channel, width, height)"
        pool, logits = self.model.forward(samples.to(self.device), interpolation = 'bicubic')
        if self.last_pool:
            return pool
        return logits
        