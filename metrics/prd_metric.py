from dataclasses import dataclass
from typing import Tuple, List

import torch
from torch.utils.data import Dataset
import third_party.PRD.prd_score as prd
import numpy as np

from metrics.metrics_base import MetricsBase
from framework.configs import EvalConfig, PlatformConfig
from framework.downsampler import Downsampler
from framework.feature_extractor.feature_extraction_helper import FeatureExtractionHelper

@dataclass
class PRD(MetricsBase):
    """
    Precision Recall (PRD) by Sajjadi et al. (2018)
    """
    eval_config :  EvalConfig = None
    platform_config : PlatformConfig = None
    real_img : Dataset = None
    real_features : torch.Tensor.type = None
    generated_img : Dataset = None
    real_to_real : bool = False

    def __post_init__(self):
        if self.real_features is None:
            if len(self.real_img) > self.eval_config.prd_num_samples:
                print("[INFO]: Downsampling real data")
                downsampler = Downsampler(full_data=self.real_img, 
                                        target_size=self.eval_config.c2st_num_samples,
                                        shuffle=True)
                self.real_img = downsampler.downsample()
            print("[INFO]: Compute real features")
            print(f"[INFO]: Feature Extractor used: {self.feature_extractor.name if self.feature_extractor is not None else 'None'}")
            self.real_features = FeatureExtractionHelper.feature_extraction(self.real_img, self.feature_extractor)
        else:
            print("[INFO]: Used cached real features")
        
        if len(self.generated_img) > self.eval_config.prd_num_samples:
            print("[INFO]: Downsampling generated data")
            downsampler = Downsampler(full_data=self.generated_img,
                                      target_size=self.eval_config.c2st_num_samples,
                                      shuffle=True)
            self.generated_img = downsampler.downsample()

    def calculate(self) -> float | Tuple[float, float, np.ndarray]:
        print("[INFO]: Compute generated features")
        gen_features = FeatureExtractionHelper.feature_extraction(self.generated_img, self.feature_extractor) if not self.real_to_real else self.real_features
        assert len(self.real_features.size()) == len(gen_features.size()), "Mismatch in dimension shapes!"
        assert self.real_features.size(0) == gen_features.size(0), "Mismatch in sample size of real and generated imgs!"
        assert self.real_features.size(1) == gen_features.size(1), "Mismatch in feature dim!"
        precisions, recalls = prd.compute_prd_from_embedding(self.real_features.numpy(), gen_features.numpy())
        f8_max_precision, f8_max_recall = prd.prd_to_max_f_beta_pair(precisions, recalls)
        return f8_max_precision, f8_max_recall, np.vstack([precisions, recalls])

    @staticmethod
    def plot_prd(precision_recall_pairs : List[np.ndarray], model_names : List[str], out_file_path = None) -> None:
        """
        Plot PRD given list of 2D precision, recall pairs
        """
        prd.plot(precision_recall_pairs = precision_recall_pairs, labels=model_names, out_path=out_file_path, legend_loc='best')


    def get_real_features(self) -> torch.Tensor.type:
        """
        Return precomputed real features
        """
        return self.real_features