# noqa
from dataclasses import dataclass
from typing import Tuple

from torch.utils.data import Dataset
import torch_fidelity
from torch_fidelity.metric_isc import KEY_METRIC_ISC_MEAN, KEY_METRIC_ISC_STD
from torch_fidelity.metric_kid import KEY_METRIC_KID_MEAN, KEY_METRIC_KID_STD
from torch_fidelity.metric_fid import KEY_METRIC_FID

from framework.configs import PlatformConfig, EvalConfig, FeatureExtractor
from framework.feature_extractor.vggface_torch_fidelity import VGGFaceFETorchFidelityWrapper

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
    feature_extractor_flag: FeatureExtractor = FeatureExtractor.InceptionV3
    feature_extractor: str = None

    def __post_init__(self):
        if self.feature_extractor_flag == FeatureExtractor.VGGFaceResNet50:
            self.feature_extractor = VGGFaceFETorchFidelityWrapper.get_default_name()
        else:
            self.feature_extractor = None

    def _compute_metric_dict(self, real_img: Dataset, generated_img: Dataset) -> None:
        """
        Compute FID, KID based on torch-fidelity
        """
        self.metric_dict = torch_fidelity.calculate_metrics(
            input1=real_img,
            input2=generated_img,
            cuda=self.platform_config.cuda,
            feature_extractor=self.feature_extractor,
            fid=True,
            kid=True,
            verbose=self.platform_config.verbose,
            kid_subsets=self.eval_config.kid_subsets,
            kid_subset_size=self.eval_config.kid_subset_size,
            kid_degree=self.eval_config.kid_degree,
            kid_coef0=self.eval_config.kid_coef0,
        )

    def get_is(self, generated_img: Dataset) -> Tuple[float, float]:
        """
        Return inception score (mean, std)
        """

        is_dict = torch_fidelity.calculate_metrics(
            input1=generated_img,
            input2=None,
            cuda=self.platform_config.cuda,
            feature_extractor=self.feature_extractor,
            isc=True,
            verbose=self.platform_config.verbose,
            isc_splits=self.eval_config.is_splits,
        )
        return (
            is_dict[KEY_METRIC_ISC_MEAN],
            is_dict[KEY_METRIC_ISC_STD],
        )

    def get_fid(self, real_img: Dataset, generated_img: Dataset) -> float:
        """
        Return FID
        """
        if self.metric_dict is None:
            self._compute_metric_dict(real_img, generated_img)
        return self.metric_dict[KEY_METRIC_FID]

    def get_kid(self, real_img: Dataset, generated_img: Dataset) -> Tuple[float, float]:
        """
        Return KID (mean, std)
        """
        if self.metric_dict is None:
            self._compute_metric_dict(real_img, generated_img)
        return (
            self.metric_dict[KEY_METRIC_KID_MEAN],
            self.metric_dict[KEY_METRIC_KID_STD],
        )
