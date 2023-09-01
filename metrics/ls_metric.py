from dataclasses import dataclass

import torch
from torch.utils.data import DataLoader, Dataset
from third_party.LS.compute_LS import gpu_LS
import numpy as np

from metrics.metrics_base import MetricsBase
from framework.configs import EvalConfig, PlatformConfig
from framework.downsampler import Downsampler

@dataclass
class LS(MetricsBase):
    """
    Likeliness Scores Metric
    """
    eval_config: EvalConfig = None
    platform_config: PlatformConfig = None
    real_img: Dataset = None
    generated_img: Dataset = None
    real_down_t: torch.Tensor.type = None
    plot_title: str = ""
    real_to_real: bool = False

    def __post_init__(self):
        if self.eval_config.ls_n_samples > 0:
            if self.real_down_t is None:
                self._set_real_subset()
            else:
                if self.real_down_t.size(0) != self.eval_config.ls_n_samples:
                    self._set_real_subset()

    def _set_real_subset(self) -> None:
        """
        Sets downsampled real dataset
        """ 
        downsampler = Downsampler(full_data=self.real_img,
                            target_size=self.eval_config.ls_n_samples,
                            num_worker=self.platform_config.num_worker,
                            shuffle=True)
        self.real_down_t = downsampler.downsample(return_tensor=True)
    
    def get_real_subset(self) -> torch.Tensor.type:
        """
        Return subsampled real data
        """
        return self.real_down_t

    def calculate(self) -> float:
        if self.eval_config.ls_n_folds == 0:
            return self._calculate_n_samples()
        else:
            return self._calculate_k_fold()
            
    def _calculate_n_samples(self) -> float:
        """
        Calculate single time LS
        """
        # one time num_sample computation
        if not self.real_to_real:
            downsampler = Downsampler(full_data=self.generated_img,
                                      target_size=self.eval_config.ls_n_samples,
                                      shuffle=True,
                                      num_worker=self.platform_config.num_worker)
            generated = downsampler.downsample(return_tensor=True)
        else:
            # real to real comparison use exact same samples
            generated = self.real_down_t

        if self.platform_config.cuda:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            device = torch.device("cpu")
            
        ls = gpu_LS(real=self.real_down_t.float().to(device),
                    gen=generated.float().to(device),
                    plot_dist=self.eval_config.ls_plot_distances,
                    plot_title=self.plot_title)
        torch.cuda.empty_cache()
        return ls
    
    def _calculate_k_fold(self) -> float:
        """
        Calculate k fold cross validation LS
        """
        if self.platform_config.cuda:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            device = torch.device("cpu")
        downsampler = Downsampler(full_data=self.generated_img,
                                  target_size=len(self.generated_img),
                                  num_worker=self.platform_config.num_worker,
                                  shuffle=False)
        generated = downsampler.downsample(return_tensor=True)
        fold_size = generated.size(0) // self.eval_config.ls_n_folds
        print(f"[INFO]: {self.eval_config.ls_n_folds}-Fold Cross Validation, Fold Size: {fold_size}")
        ls_scores = []
        
        for fold in torch.split(generated, fold_size):
            # for k fold new real subset per fold
            downsampler = Downsampler(full_data=self.real_img,
                                      target_size=fold.size(0),
                                      num_worker=self.platform_config.num_worker,
                                      shuffle=True)
            real = downsampler.downsample(return_tensor=True)
            ls = gpu_LS(real=real.float().to(device), 
                        gen=fold.float().to(device),
                        plot_dist=self.eval_config.ls_plot_distances,
                        plot_title=self.plot_title)
            ls_scores.append(ls)
            torch.cuda.empty_cache()
        torch.cuda.empty_cache()
        return np.mean(ls_scores)
        

