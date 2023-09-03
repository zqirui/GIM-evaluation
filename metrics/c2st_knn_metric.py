# pylint: disable=no-member
# pylint: disable=invalid-name

# Optional reparameterization into [0,1] with 1 = 0.5 accuracy

from dataclasses import dataclass
from typing import Tuple

import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from tqdm import tqdm
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SequentialSampler

from framework.configs import EvalConfig, PlatformConfig
from framework.downsampler import Downsampler
from metrics.metrics_base import MetricsBase

@dataclass
class C2STKNN(MetricsBase):
    """
    Classifier Two-Sample Test KNN
    """
    eval_config : EvalConfig = None
    platform_config : PlatformConfig = None
    real_img : Dataset = None
    generated_img : Dataset = None
    real_features : torch.Tensor.type = None
    real_to_real : bool = False

    def __post_init__(self):
        # downsample both real and generated imgs
        print(f"[INFO]: Feature Extractor used: {self.feature_extractor.name if self.feature_extractor is not None else 'None'}")
        # feature extraction real images
        if self.real_features is None:
            if len(self.real_img) > self.eval_config.c2st_num_samples:
                print("[INFO]: Downsampling real data")
                downsampler = Downsampler(full_data=self.real_img, 
                                        target_size=self.eval_config.c2st_num_samples,
                                        shuffle=True)
                self.real_img = downsampler.downsample()
            print("[INFO]: Compute real features")
            # only compute first time
            self.real_features = self._feature_extraction(self.real_img)
        else:
            print("[INFO]: Used cached real features")

        if len(self.generated_img) > self.eval_config.c2st_num_samples and not self.real_to_real:
            print("[INFO]: Downsampling generated data")
            downsampler = Downsampler(full_data=self.generated_img,
                                      target_size=self.eval_config.c2st_num_samples,
                                      shuffle=True)
            self.generated_img = downsampler.downsample()

    def calculate(self) -> float | Tuple[float, float]:
        # feature extraction generated images
        print("[INFO]: Compute generated features")
        if not self.real_to_real:
            generated_features = self._feature_extraction(self.generated_img)
        else:
            generated_features = self.real_features
        # build dataset of equal |R| = |G|, R = 1, G = 1
        X = torch.vstack([self.real_features, generated_features])
        y = torch.zeros(X.size(0))
        y[:self.eval_config.c2st_num_samples] = 1
        X = X.numpy()
        y = y.numpy()
        # KNN cross validation 
        knn_acc = self._knn_classification(X, y)
        return knn_acc
        
    def _feature_extraction(self, all_samples : Dataset) -> torch.Tensor.type:
        """
        Perform feature extraction given a Dataset
        Returns a Tensor
        """
        dl = DataLoader(all_samples, batch_size=32, sampler=SequentialSampler(all_samples))
        samples_features = []
        for batch_samples in tqdm(dl, ascii=True, desc="[INFO]: Feature Extraction"):
            samples_features.append(self.feature_extractor.extract(batch_samples).detach().cpu())
            torch.cuda.empty_cache()
        samples_features = torch.vstack(samples_features)
        return samples_features

    def _knn_classification(self, X : np.ndarray, y : np.ndarray) -> float:
        """
        Returns the cross validation accuracy of KNN classification
        """
        assert X.shape[0] == y.shape[0], "Mismatch of sample size and label size!"
        skf = StratifiedKFold(n_splits=self.eval_config.c2st_folds)
        accuracies = []
        k = self.eval_config.c2st_k
        for train_idx, test_idx in tqdm(skf.split(X,y), ascii=True, desc="[INFO]: KNN Cross Valiation", total=self.eval_config.c2st_folds):
            if self.eval_config.c2st_k_adaptive:
                k = int(np.floor(np.sqrt(len(X[train_idx])))) # Lopez-Paz (2018)
            knn = KNeighborsClassifier(n_neighbors=k)
            knn.fit(X[train_idx], y[train_idx])
            accuracies.append(accuracy_score(y[test_idx], knn.predict(X[test_idx])))
        return np.mean(accuracies)

    def get_real_features(self) -> torch.Tensor.type:
        """
        Return calculated real features
        """
        return self.real_features