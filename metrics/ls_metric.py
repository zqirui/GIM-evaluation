from dataclasses import dataclass

import torch
from torch.utils.data import DataLoader, Dataset
from third_party.LS.compute_LS import gpu_LS
import numpy as np

from metrics.metrics_base import MetricsBase
from framework.configs import EvalConfig, PlatformConfig
from framework.image_source import ImageSource

@dataclass
class LS(MetricsBase):
    """
    Likeliness Scores Metric
    """
    eval_config: EvalConfig = None
    platform_config: PlatformConfig = None
    real_src: ImageSource = None
    generated_src: ImageSource = None
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
        
        if self.real_src.source_name == self.generated_src.source_name:
            self.real_to_real = True

    def _set_real_subset(self) -> None:
        """
        Sets downsampled real dataset
        """    
        dl_real = self.real_src.get_dataloader(num_worker=self.eval_config.ls_num_worker, 
                                                batch_size=64,
                                                shuffle=True,
                                                subsample=True,
                                                subsample_n=self.eval_config.ls_n_samples)
        self.real_down_t = self._dl_to_tensor(dl_real)
    
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

            
    def _dl_to_tensor(self, dataloader : DataLoader) -> torch.Tensor.type:
        """
        Returns single tensor given dataloader
        """
        samples = []
        for sample in dataloader:
            samples.append(sample)
        samples = torch.vstack(samples)
        return samples
            
    def _calculate_n_samples(self) -> float:
        """
        Calculate single time LS
        """
        # one time num_sample computation
        if not self.real_to_real:
            dl_gen = self.generated_src.get_dataloader(num_worker=self.eval_config.ls_num_worker, 
                                                batch_size=64,
                                                shuffle=True,
                                                subsample=True,
                                                subsample_n=self.eval_config.ls_n_samples)
            generated = self._dl_to_tensor(dl_gen)
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
        dl_gen = self.generated_src.get_dataloader(num_worker=self.eval_config.ls_num_worker, 
                                            batch_size=64,
                                            shuffle=False)
        generated = self._dl_to_tensor(dl_gen)
        fold_size = generated.size(0) // self.eval_config.ls_n_folds
        print(f"[INFO]: {self.eval_config.ls_n_folds}-Fold Cross Validation, Fold Size: {fold_size}")
        ls_scores = []
        
        for fold in torch.split(generated, fold_size):
            # for k fold new real subset per fold
            dl_real = self.real_src.get_dataloader(num_worker=self.eval_config.ls_num_worker, 
                                            batch_size=64,
                                            shuffle=True,
                                            subsample=True,
                                            subsample_n=fold.size(0))
            real = self._dl_to_tensor(dl_real)
            ls = gpu_LS(real=real.float().to(device), 
                        gen=fold.float().to(device),
                        plot_dist=self.eval_config.ls_plot_distances,
                        plot_title=self.plot_title)
            ls_scores.append(ls)
            torch.cuda.empty_cache()
        torch.cuda.empty_cache()
        return np.mean(ls_scores)
        

