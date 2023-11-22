from dataclasses import dataclass
from typing import Tuple, Union

import torch
from torchmetrics.functional import pairwise_cosine_similarity
from tqdm import tqdm

from metrics.metrics_base import MetricsBase
from framework.configs import EvalConfig, PlatformConfig

@dataclass
class MiFID(MetricsBase):
    """
    Memorization-informed FID (Bai et al., 2021)

    See: https://arxiv.org/abs/2106.03062
    """
    eval_config: EvalConfig = None
    platform_config: PlatformConfig = None
    real_features: torch.Tensor.type = None
    gen_features: torch.Tensor.type = None
    fid: Union[float, torch.Tensor.type] = None

    def __post_init__(self):
        print(f"[INFO]: Feature Extractor used: {self.feature_extractor_flag.value}")
        if not torch.is_tensor(self.real_features):
            self.real_features = torch.tensor(self.real_features)
        if not torch.is_tensor(self.gen_features):
            self.gen_features = torch.tensor(self.gen_features)
        if not torch.is_tensor(self.gen_features):
            self.fid = torch.tensor(self.fid).clamp(min=1e-3)    
        assert len(self.real_features.size()) == 2, "[ERROR]: Real features not in shape (batch, feature_dim)"
        assert len(self.gen_features.size()) == 2, "[ERROR]: Generated features not in shape (batch, feature_dim)"

    def calculate(self) -> float | Tuple[float, float]:
        similarities = []
        for gen_split in tqdm(torch.split(self.gen_features, 10000), ascii=True, desc="[INFO]: MiFID Splits"):
            pairwise_cos = pairwise_cosine_similarity(gen_split, self.real_features)
            similarity = torch.min(1 - pairwise_cos, dim=1)
            similarities.append(similarity.values)
        avg_sim = torch.cat(similarities).mean().clamp(min=1e-6, max=1.0)
        mem_penalty =  avg_sim if avg_sim < self.eval_config.mifid_tau else 1.0
        mifid = self.fid / mem_penalty
        return mifid, mem_penalty
