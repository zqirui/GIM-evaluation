from dataclasses import dataclass
from typing import Tuple, Union

import numpy as np
from cleanfid import fid as clean_metric

from metrics.metrics_base import MetricsBase
from framework.configs import EvalConfig, PlatformConfig, FeatureExtractor
from framework.feature_extractor.vggface_nn_module import VGGFaceNNModuleFeatures

@dataclass
class CleanFID(MetricsBase):
    """
    Clean FID Metric
    """
    eval_config: EvalConfig = None
    platform_config: PlatformConfig = None
    real_img_path: str = None
    generated_img_path: str = None
    real_features: np.ndarray = None
    gen_features: np.ndarray = None

    def __post_init__(self):
        print(f"[INFO]: Feature Extractor used: {self.feature_extractor_flag.value}")

    def calculate(self) -> float | Tuple[float, float]:
        clean_fid, self.real_features, self.gen_features = clean_metric.compute_fid(
            fdir1=self.real_img_path,
            fdir2=self.generated_img_path,
            batch_size=self.eval_config.clean_fid_batch_size,
            num_workers=0, # >0 throws pickle error in multiprocessing
            custom_feat_extractor= None if self.feature_extractor_flag == FeatureExtractor.InceptionV3 else VGGFaceNNModuleFeatures(),
            rtn_features=True,
            fdir1_features=self.real_features,
            fdir2_features=self.gen_features,
        )
        return clean_fid
    
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