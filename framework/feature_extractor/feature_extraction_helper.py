import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SequentialSampler
from tqdm import tqdm

from framework.feature_extractor.feature_extractor_base import FeatureExtractorBase

class FeatureExtractionHelper():
    """
    Feature Extractiom Helper Class
    """

    @staticmethod
    def feature_extraction(all_samples : Dataset, feature_extractor : FeatureExtractorBase) -> torch.Tensor.type:
        """
        Feature extraction given Dataset and Feature Extractor
        """
        dl = DataLoader(all_samples, batch_size=32, sampler=SequentialSampler(all_samples))
        samples_features = []
        for batch_samples in tqdm(dl, ascii=True, desc="[INFO]: Feature Extraction"):
            samples_features.append(feature_extractor.extract(batch_samples).detach().cpu())
            torch.cuda.empty_cache()
        samples_features = torch.vstack(samples_features)
        return samples_features