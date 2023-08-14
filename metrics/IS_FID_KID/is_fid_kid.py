# noqa
from dataclasses import dataclass
from typing import Tuple

from torch.utils.data import Dataset
import torch_fidelity
from torch_fidelity.metric_isc import KEY_METRIC_ISC_MEAN, KEY_METRIC_ISC_STD
from torch_fidelity.metric_kid import KEY_METRIC_KID_MEAN, KEY_METRIC_KID_STD
from torch_fidelity.metric_fid import KEY_METRIC_FID

from framework.Configs import PlatformConfig, EvalConfig


@dataclass
class IsFidKidBase:
    """
    Inception Score (IS),
    Fréchet Inception Distance (FID),
    Kernel Inception Distance (KID)
    Parent class (all three scores are always computed for effiency reasons)
    """

    eval_config: EvalConfig
    platform_config: PlatformConfig
    metric_dict: dict = None

    def _compute_metric_dict(self, real_img: Dataset, generated_img: Dataset) -> None:
        """
        Compute FID, KID based on torch-fidelity
        """
        self.metric_dict = torch_fidelity.calculate_metrics(
            input1=real_img,
            input2=generated_img,
            cuda=self.platform_config.cuda,
            fid=True,
            kid=True,
            verbose=self.platform_config.verbose,
            kid_subsets=self.eval_config.kid_subsets,
            kid_subset_size=self.eval_config.kid_subset_size,
            kid_degree=self.eval_config.kid_degree,
            kid_coef0=self.eval_config.kid_coef0,
        )

    def get_Is(self, generated_img: Dataset) -> Tuple[float, float]:
        """
        Return inception score (mean, std)
        """

        is_dict = torch_fidelity.calculate_metrics(
            input1=generated_img,
            input2=None,
            cuda=self.platform_config.cuda,
            isc=True,
            verbose=self.platform_config.verbose,
            isc_splits=self.eval_config.is_splits,
        )
        return (
            is_dict[KEY_METRIC_ISC_MEAN],
            is_dict[KEY_METRIC_ISC_STD],
        )

    def get_Fid(self, real_img: Dataset, generated_img: Dataset) -> float:
        """
        Return FID
        """
        if self.metric_dict is None:
            self._compute_metric_dict(real_img, generated_img)
        return self.metric_dict[KEY_METRIC_FID]

    def get_Kid(self, real_img: Dataset, generated_img: Dataset) -> Tuple[float, float]:
        """
        Return KID (mean, std)
        """
        if self.metric_dict is None:
            self._compute_metric_dict(real_img, generated_img)
        return (
            self.metric_dict[KEY_METRIC_KID_MEAN],
            self.metric_dict[KEY_METRIC_KID_MEAN],
        )
