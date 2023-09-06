from typing import Tuple

import torch
from torch_fidelity import FeatureExtractorBase

from framework.feature_extractor.vggface_nn_module import VGGFaceNNModule

class VGGFaceFETorchFidelityWrapper(FeatureExtractorBase):
    """
    VGGFace Feature Extractor Torch Fidelity Wrapper
    """
    def __init__(self, name, features_list, **kwargs,):
        super().__init__(name, features_list)
        self.model = VGGFaceNNModule()

    def forward(self, samples):
        features = {}
        remaining_features = self.features_list.copy()

        pool, logits = self.model(samples)
        features['2048'] = pool
        if '2048' in remaining_features:
            remaining_features.remove('2048')
        features['8631_logits'] = logits
        if '8631_logits' in remaining_features:
            remaining_features.remove('8631_logits')
        return tuple(features[x] for x in self.features_list)

    
    @staticmethod
    def get_provided_features_list() -> Tuple[str,...]:
        return '2048', '8631_logits'

    @staticmethod
    def get_default_feature_layer_for_metric(metric) -> dict:
        return {
            'isc': '8631_logits',
            'fid': '2048',
            'kid': '2048',
            'prc': '2048',
        }[metric]
    
    @staticmethod
    def get_default_name() -> str:
        return "VGGFaceResNet50"

    @staticmethod
    def can_be_compiled() -> bool:
        return True
    
    @staticmethod
    def get_dummy_input_for_compile() -> torch.Tensor.type:
        return (torch.rand([1, 3, 64, 64]) * 255).to(torch.uint8).float()