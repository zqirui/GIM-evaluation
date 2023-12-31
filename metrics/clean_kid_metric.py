from dataclasses import dataclass
from typing import Tuple, Union

import numpy as np
import torch
from cleanfid import fid as clean_metric

from metrics.metrics_base import MetricsBase
from framework.configs import EvalConfig, PlatformConfig, FeatureExtractor
from framework.feature_extractor.vggface_nn_module import VGGFaceNNModuleFeatures

@dataclass
class CleanKID(MetricsBase):
    """
    Clean FID Metric
    """
    eval_config: EvalConfig = None
    platform_config: PlatformConfig = None
    real_img_path: str = None
    generated_img_path: str = None
    real_features: Union[np.ndarray, torch.Tensor.type] = None
    gen_features: Union[np.ndarray, torch.Tensor.type] = None

    def __post_init__(self):
        print(f"[INFO]: Feature Extractor used: {self.feature_extractor_flag.value}")
        if torch.is_tensor(self.real_features):
            self.real_features = self.real_features.cpu().numpy()
        if torch.is_tensor(self.gen_features):
            self.gen_features = self.gen_features.cpu().numpy()

    def calculate(self) -> float | Tuple[float, float]:
        clean_kid, self.real_features, self.gen_features = clean_metric.compute_kid(
            fdir1=self.real_img_path,
            fdir2=self.generated_img_path,
            batch_size=self.eval_config.clean_kid_batch_size,
            num_workers=0, # >0 throws pickle error in multiprocessing
            custom_feat_extractor= None if self.feature_extractor_flag == FeatureExtractor.InceptionV3 else VGGFaceNNModuleFeatures(),
            rtn_features=True,
            fdir1_features=self.real_features,
            fdir2_features=self.gen_features,
        )
        return clean_kid
    
    def get_real_features(self):
        """
        Return computed real features
        """
        return self.real_features
    
    def get_gen_features(self):
        """
        Return computed generated features
        """
        return self.gen_features