# pylint: disable=wrong-import-position
from dataclasses import dataclass, field
from typing import List, Union
import os

os.environ["OMP_NUM_THREADS"] = "4"
import json

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from torch_fidelity import register_feature_extractor

from framework.image_source import ImageSource
from framework.configs import EvalConfig, PlatformConfig, FeatureExtractor
from framework.feature_extractor.inceptionV3 import FeatureExtractorBase, InceptionV3FE
from framework.feature_extractor.vggface import VGGFaceFE
from framework.feature_extractor.vggface_torch_fidelity import (
    VGGFaceFETorchFidelityWrapper,
)
from metrics.is_fid_kid.is_fid_kid import IsFidKidBase
from metrics.is_fid_kid.metrics import IS, FID, KID
from metrics.prc_metric import PRC
from metrics.fid_infty_metric import FID_infty
from metrics.is_infty_metric import IS_infty
from metrics.clean_fid_metric import CleanFID
from metrics.clean_kid_metric import CleanKID
from metrics.ls_metric import LS
from metrics.c2st_knn_metric import C2STKNN
from metrics.prd_metric import PRD
from metrics.mifid_metric import MiFID


@dataclass
class ResultDict:
    """
    Custom Dict for saving results
    """

    data: dict
    normalized_data: Union[dict, None] = None

    def add(self, key: str, value: dict):
        """
        Add a metric for one generator
        """
        self.data[key] = value

    def normalize_scores(self, print_results: bool = True, exlude_IS: bool = False):
        """
        Create Global rankings and normalize scores
        """
        self.normalized_data = dict()
        # reanrrage metrics in new dict
        aux_dict = dict()
        for generator in self.data:
            for metric in self.data[generator]:
                try:
                    aux_dict[metric][generator] = self.data[generator][metric]
                except KeyError:
                    aux_dict[metric] = {}
                    aux_dict[metric][generator] = self.data[generator][metric]

        # collect to be norm scores
        for metric in aux_dict:
            if self.to_normalize(metric):
                metric_scores = []
                for generator in aux_dict[metric]:
                    metric_scores.append(aux_dict[metric][generator])
                metric_scores_norm = self.normalize(metric_scores)
                if "Distance" in metric or "FID" in metric or "KID" in metric:
                    metric_scores_norm = 1 - metric_scores_norm
                for i, generator in enumerate(aux_dict[metric]):
                    try:
                        self.normalized_data[generator][metric] = metric_scores_norm[i]
                    except KeyError:
                        self.normalized_data[generator] = {}
                        self.normalized_data[generator][metric] = metric_scores_norm[i]
            else:
                # already normalized
                for generator in aux_dict[metric]:
                    self.normalized_data[generator][metric] = aux_dict[metric][
                        generator
                    ]

        # add overall ranking
        for generator in self.normalized_data:
            scores_norm = []
            for metric in self.normalized_data[generator]:
                if metric in self.get_norm_metric_names():                 
                    scores_norm.append(self.normalized_data[generator][metric])
                    
            # calc delta norm
            scores_norm = np.asanyarray(scores_norm)
            self.normalized_data[generator]["Delta Norm"] = (
                np.mean(scores_norm[2:])
                if exlude_IS
                else np.mean(scores_norm)
            )

        if print_results:
            self.print(dictionary=self.normalized_data, round_scores=True, norm=True)

    def get_norm_metric_names(self) -> List[str]:
        """
        Return Name of Metrics for delta norm
        """
        return [
            "Inception Score Mean",
            "Frechet Inception Distance",
            "Kernel Inception Distance Mean",
            "Precision",
            "Recall",
            "FID Infinity (Approx.)",
            "IS Infinity (Approx.)",
            "Clean FID",
            "MiFID",
            "Clean KID",
            "LS",
            "C2ST Adaptive KNN Normalized",
            "PRD F8 Max Recall",
            "PRD F8 Max Precision",
        ]

    def get_thesis_metric_names(self) -> List[str]:
        """
        Return Name of Metrics reported in Thesis
        """
        return [
            "Inception Score Mean",
            "Frechet Inception Distance",
            "Kernel Inception Distance Mean",
            "Precision",
            "Recall",
            "FID Infinity (Approx.)",
            "IS Infinity (Approx.)",
            "Clean FID",
            "MiFID",
            "Clean KID",
            "LS",
            "C2ST Adaptive KNN Accuracy",
            "PRD F8 Max Recall",
            "PRD F8 Max Precision",
        ]

    def normalize(self, x: List) -> List[float]:
        """
        Normalize to [0,1] based on the given sequence
        """
        x = np.asarray(x)
        x_norm = (x - np.min(x)) / (np.max(x) - np.min(x))
        return x_norm

    def to_normalize(self, metric: str) -> bool:
        """
        Check if a metric have to be normalized
        """
        to_norm = ["Inception", "FID", "KID", "IS"]
        for category in to_norm:
            if category in metric:
                if "Std" in metric:
                    return False
                return True

        return False

    def print(
        self,
        dictionary: Union[dict, None] = None,
        round_scores: bool = False,
        norm: bool = False,
    ):
        """
        Custom print function
        """
        if dictionary is None:
            data = self.data
        else:
            data = dictionary
        for generator in data:
            print("------------------------------------")
            print(f"Generator Name: {generator}")
            print("------------------------------------")
            for metric in data[generator]:
                if not norm:
                    score = (
                        np.round(data[generator][metric], decimals=3)
                        if round_scores
                        else data[generator][metric]
                    )
                    print(f"{metric}: {score}")
                else:
                    if metric in self.get_norm_metric_names() or metric == "Delta Norm":
                        score = (
                            np.round(data[generator][metric], decimals=3)
                            if round_scores
                            else data[generator][metric]
                        )
                        print(f"{metric}: {score}")

    def write_to_json(self, data: Union[dict, None] = None, file="default.json"):
        """
        Write dict out to json
        """
        if data is None:
            data = self.data
        json_obj = json.dumps(data)
        with open(file, "w") as outf:
            outf.write(json_obj)

    def read_from_json(self, file="default.json", norm=False):
        """
        Read dict from json
        """
        with open(file, "r") as f:
            if not norm:
                self.data = json.load(f)
            else:
                self.normalized_data = json.load(f)

    def get_as_pd(self, only_thesis_metrics : bool = False, normalized_scores : bool = False, norm_exclude_IS : bool = False):
        """
        Return Dict as pandas df
        """
        aux_dict = dict()
        gen_names = []
        if normalized_scores:
            self.normalize_scores(print_results=False, exlude_IS=norm_exclude_IS)
        data = self.data if not normalized_scores else self.normalized_data
        for generator in data:
            gen_names.append(generator)
            for metric in data[generator]:
                try:
                    aux_dict[metric][generator] = data[generator][metric]
                except KeyError:
                    aux_dict[metric] = {}
                    aux_dict[metric][generator] = data[generator][metric]

        aux_array = []
        metric_names = []
        for metric in aux_dict:  #pylint:disable=consider-using-dict-items
            metric_results = []
            if only_thesis_metrics:
                if not normalized_scores:
                    if metric not in self.get_thesis_metric_names():
                        continue
                else:
                    if metric not in self.get_norm_metric_names():
                        continue
            metric_names.append(metric)
            for generator in aux_dict[metric]:
                metric_results.append(np.round(aux_dict[metric][generator],3))
            aux_array.append(metric_results)

        aux_array = np.asarray(aux_array)

        df = pd.DataFrame(aux_array, columns = gen_names)
        df.index = [r'$IS$',r'$IS_{\infty}$',r'$FID$',r'$FID_{\infty}$',r'$CleanFID$',r'$MiFID$',r'$KID$',r'$CleanKID$',"PRC-Precision","PRC-Recall",r"$PRD \; F_{\frac{1}{8}} \; Precision$",r"$PRD \; F_{8} \; Recall$","LS","C2ST-KNN",]
        df[df < 0.0] = 0.0
        return df

@dataclass
class PRDMappings:
    """
    Simple Map holding precision and recall values
    """

    precision_recall_pairs: List[np.ndarray] = field(default_factory=list)
    names: List[str] = field(default_factory=list)


@dataclass
class ManagerHelper:
    """
    Manager helper class, holding auxiliary variables
    """

    real_images_src: ImageSource
    generated_images_srcs: List[ImageSource]
    fid_infty_features: Union[torch.Tensor.type, None] = None
    cleanfid_features: Union[torch.Tensor.type, None] = None
    cleanfid_fake_features: Union[torch.Tensor.type, None] = None
    cleankid_features: Union[torch.Tensor.type, None] = None
    real_images_subsampled: Union[Dataset, None] = None
    ls_real_subset: Union[torch.Tensor.type, None] = None
    c2st_real_features: Union[torch.Tensor.type, None] = None
    feature_extractor: Union[FeatureExtractorBase, None] = None
    prd_real_features: Union[torch.Tensor.type, None] = None
    prd_mappings: PRDMappings = PRDMappings()

    def contains_generator(self, src_name: str) -> bool:
        """
        Check for generator source
        """
        return any(src.source_name == src_name for src in self.generated_images_srcs)

    def get_generator_src(self, src_name: str) -> Union[ImageSource, None]:
        """
        Try to get generator source by name
        """
        for src in self.generated_images_srcs:
            if src.source_name == src_name:
                return src
        return None


class PlatformManager:
    """
    Main platform managing class
    """

    def __init__(
        self,
        eval_config: EvalConfig,
        platform_config: PlatformConfig = PlatformConfig(),
        generated_images_path: str = os.path.join(os.getcwd(), "generated_images"),
        real_images_path: str = os.path.join(os.getcwd(), "original_images"),
    ) -> None:
        """
        Init constructor
        """
        self.eval_cfg = eval_config
        self.platform_cfg = platform_config
        self.out_dict = ResultDict(data={})
        self.comparator_dict = None
        real_img_src_name = "CelebA64 (Original)"
        real_images_src = ImageSource(real_images_path, real_img_src_name)
        print(f"[INFO]: Real images source found, name:{real_img_src_name}")
        # create one source per sub-folder in generated images parent folder
        generator_srcs = []
        subfolders = next(os.walk(generated_images_path))[1]
        print(f"[INFO]: {len(subfolders)} different generator sources found. Names:")
        for generator_folder_name in subfolders:
            print(generator_folder_name)
            generator_folder_path = os.path.join(
                generated_images_path, generator_folder_name
            )
            generated_src = ImageSource(generator_folder_path, generator_folder_name)
            generator_srcs.append(generated_src)

        self.helper = ManagerHelper(real_images_src, generator_srcs)

        if self.eval_cfg.feature_extractor == FeatureExtractor.VGGFaceResNet50:
            try:
                register_feature_extractor(
                    VGGFaceFETorchFidelityWrapper.get_default_name(),
                    VGGFaceFETorchFidelityWrapper,
                )
            except ValueError:
                # if already registered
                pass

    def calc_metrics(self) -> ResultDict:
        """
        Main calculation method
        """

        # get tensor datasets
        real_img = self.helper.real_images_src.get_dataset()

        # add real imag src if real-to-real comparison is desired
        print(
            f"[INFO]: Comparison real-to-real ({self.platform_cfg.compare_real_to_real})"
        )
        if self.platform_cfg.compare_real_to_real:
            self.helper.generated_images_srcs.append(self.helper.real_images_src)

        for generator_src in self.helper.generated_images_srcs:
            self.comparator_dict = {}
            generated_img = generator_src.get_dataset()
            self.helper.cleanfid_fake_features = None
            real_to_real = (
                True
                if self.helper.real_images_src.source_name == generator_src.source_name
                else False
            )
            print(f"[START]: Calculating Metrics for {generator_src.source_name}")

            # check for IS, FID, KID
            if self.eval_cfg.inception_score or self.eval_cfg.fid or self.eval_cfg.kid or self.eval_cfg.mifid:
                is_fid_kid_base = IsFidKidBase(
                    self.eval_cfg,
                    self.platform_cfg,
                    feature_extractor_flag=self.eval_cfg.feature_extractor,
                )

            if self.eval_cfg.inception_score:
                print(
                    f"[INFO]: Start Calculation IS, Source = {generator_src.source_name}"
                )
                name = "Inception Score"
                metric_is = IS(
                    name=name,
                    inception_base=is_fid_kid_base,
                    generated_img=generated_img,
                )
                mean, std = metric_is.calculate()
                self.comparator_dict.update({name + " Mean": mean, name + " Std": std})
                print("[INFO]: IS finished")

            if self.eval_cfg.fid:
                self.compute_fid(generator_src=generator_src, is_fid_kid_base=is_fid_kid_base, real_img=real_img, generated_img=generated_img)

            if self.eval_cfg.kid:
                print(
                    f"[INFO]: Start Calculation KID, Source = {generator_src.source_name}"
                )
                name = "Kernel Inception Distance"
                metric_kid = KID(
                    name=name,
                    inception_base=is_fid_kid_base,
                    real_img=real_img,
                    generated_img=generated_img,
                )
                mean, std = metric_kid.calculate()
                self.comparator_dict.update({name + " Mean": mean, name + " Std": std})
                print("[INFO]: KID finished")

            if self.eval_cfg.prc:
                self.compute_prc(real_img=real_img, generator_src=generator_src)

            if self.eval_cfg.clean_fid:
                self.compute_cleanfid(generator_src=generator_src)
                
            if self.eval_cfg.mifid:
                print(
                    f"[INFO]: Start Calculation MiFID, Source = {generator_src.source_name}"
                )
                name = "MiFID"
                if "Frechet Inception Distance" not in self.comparator_dict:
                    self.compute_fid(generator_src=generator_src,
                                     is_fid_kid_base=is_fid_kid_base,
                                     real_img=real_img,
                                     generated_img=generated_img)
                    
                if self.helper.cleanfid_features is None or self.helper.cleanfid_fake_features is None:
                    print(
                    "[INFO]: No precomputed real or generated features found. Compute from CleanFID"
                    )
                    self.compute_cleanfid(generator_src)               
   
                metric_mifid = MiFID(name=name,
                                     eval_config=self.eval_cfg,
                                     platform_config=self.platform_cfg,
                                     feature_extractor_flag=self.eval_cfg.feature_extractor,
                                     real_features=self.helper.cleanfid_features,
                                     gen_features=self.helper.cleanfid_fake_features,
                                     fid=self.comparator_dict["Frechet Inception Distance"]) 
                mifid, m_tau = metric_mifid.calculate()
                print(f"[INFO]: MiFID m_tau = {m_tau}")
                self.comparator_dict.update({name: float(mifid), "m_tau": float(m_tau)})
                print("[INFO]: MiFID finished")

            if self.eval_cfg.fid_infinity:
                print(
                    f"[INFO]: Start Calculation FID infinity, Source = {generator_src.source_name}"
                )
                name = "FID Infinity (Approx.)"
                metric_fid_infty = FID_infty(
                    name=name,
                    eval_config=self.eval_cfg,
                    platform_config=self.platform_cfg,
                    real_img_path=self.helper.real_images_src.folder_path,
                    generated_img_path=generator_src.folder_path,
                    feature_extractor_flag=self.eval_cfg.feature_extractor,
                    precomputed_real_features=None
                    if self.helper.cleanfid_features is None
                    else self.helper.cleanfid_features,
                    precomputed_fake_features=None 
                    if self.helper.cleanfid_fake_features is None
                    else self.helper.cleanfid_fake_features
                )
                fid_infty = metric_fid_infty.calculate()
                self.helper.fid_infty_features = (
                    metric_fid_infty.get_real_features()
                    if self.helper.fid_infty_features is None
                    else self.helper.fid_infty_features
                )
                self.comparator_dict.update({name: fid_infty})
                print("[INFO]: FID infinity finished")

            if self.eval_cfg.is_infinity:
                print(
                    f"[INFO]: Start Calculation IS infinity, Source = {generator_src.source_name}"
                )
                name = "IS Infinity (Approx.)"
                metric_is_infty = IS_infty(
                    name=name,
                    eval_config=self.eval_cfg,
                    platform_config=self.platform_cfg,
                    generated_img_path=generator_src.folder_path,
                    feature_extractor_flag=self.eval_cfg.feature_extractor,
                )
                is_infty = metric_is_infty.calculate()
                self.comparator_dict.update({name: is_infty})
                print("[INFO]: IS infinity finished")

            if self.eval_cfg.clean_kid:
                print(
                    f"[INFO]: Start Calculation Clean KID, Source = {generator_src.source_name}"
                )
                name = "Clean KID"
                metric_clean_kid = CleanKID(
                    name=name,
                    eval_config=self.eval_cfg,
                    platform_config=self.platform_cfg,
                    real_img_path=self.helper.real_images_src.folder_path,
                    generated_img_path=generator_src.folder_path,
                    feature_extractor=self.eval_cfg.feature_extractor,
                    real_features=self.helper.cleanfid_features
                    if self.helper.cleanfid_features is not None
                    else None,
                    gen_features=self.helper.cleanfid_fake_features 
                    if self.helper.cleanfid_fake_features is not None
                    else None,
                )
                clean_kid = metric_clean_kid.calculate()
                if self.helper.cleankid_features is None:
                    self.helper.cleankid_features = metric_clean_kid.get_real_features()
                self.comparator_dict.update({name: clean_kid})
                print("[INFO]: Clean KID finished")

            if self.eval_cfg.ls:
                self.compute_ls(
                    real_img=real_img,
                    generator_src=generator_src,
                    real_to_real=real_to_real,
                )

            if self.eval_cfg.c2st_knn:
                self.compute_c2st_knn(
                    real_img=real_img,
                    generator_src=generator_src,
                    real_to_real=real_to_real,
                )

            if self.eval_cfg.prd:
                self.compute_prd(
                    real_img=real_img,
                    generator_src=generator_src,
                    real_to_real=real_to_real,
                )

            print(f"[FINISHED]: Calculating Metrics for {generator_src.source_name}")
            print(self.comparator_dict)
            self.out_dict.add(key=generator_src.source_name, value=self.comparator_dict)

        if self.eval_cfg.prd_plot:
            PRD.plot_prd(
                self.helper.prd_mappings.precision_recall_pairs,
                self.helper.prd_mappings.names,
                out_file_path="prd_plot.svg",
            )

        return self.out_dict

    def compute_prc(self, real_img: Dataset, generator_src: ImageSource) -> None:
        """
        PRC Computation
        """
        print(
            f"[INFO]: Start Calculation Improved PRC, Source = {generator_src.source_name}"
        )
        name = "Improved Precision Recall (PRC)"
        if self.helper.real_images_subsampled is None:
            metric_prc = PRC(
                name=name,
                eval_config=self.eval_cfg,
                platform_config=self.platform_cfg,
                real_img=real_img,
                generated_img=generator_src.get_dataset(),
                feature_extractor_flag=self.eval_cfg.feature_extractor,
            )
            self.helper.real_images_subsampled = metric_prc.get_real_subsampled_imgs()
        else:
            # reuse same subsampled real dataset for all comparison models
            metric_prc = PRC(
                name=name,
                eval_config=self.eval_cfg,
                platform_config=self.platform_cfg,
                real_img=self.helper.real_images_subsampled,
                generated_img=generator_src.get_dataset(),
                feature_extractor_flag=self.eval_cfg.feature_extractor,
            )
        precision, recall, f1 = metric_prc.calculate()
        self.comparator_dict.update(
            {"Precision": precision, "Recall": recall, "F1 Score": f1}
        )
        print("[INFO]: Improved PRC finished")

    def compute_ls(
        self, real_img: Dataset, generator_src: ImageSource, real_to_real: bool
    ) -> None:
        """
        Compute Likeliness Score
        """
        print(
            f"[INFO]: Start Calculation Likeliness Scores (LS), Source = {generator_src.source_name}"
        )
        name = "LS"
        if self.eval_cfg.ls_n_samples > 0:
            # single time calculation
            if self.helper.ls_real_subset is None:
                metric_ls = LS(
                    name=name,
                    eval_config=self.eval_cfg,
                    platform_config=self.platform_cfg,
                    real_img=real_img,
                    generated_img=generator_src.dataset,
                    real_to_real=real_to_real,
                    plot_title=f"ICDs and BCD, {generator_src.source_name} vs {self.helper.real_images_src.source_name}",
                )
                # set on first calculation
                self.helper.ls_real_subset = metric_ls.get_real_subset()
            else:
                metric_ls = LS(
                    name=name,
                    eval_config=self.eval_cfg,
                    platform_config=self.platform_cfg,
                    real_img=real_img,
                    generated_img=generator_src.dataset,
                    real_to_real=real_to_real,
                    plot_title=f"ICDs and BCD, {generator_src.source_name} vs {self.helper.real_images_src.source_name}",
                    real_down_t=self.helper.ls_real_subset,
                )
        else:
            # k fold
            metric_ls = LS(
                name=name,
                eval_config=self.eval_cfg,
                platform_config=self.platform_cfg,
                real_img=real_img,
                generated_img=generator_src.dataset,
                real_to_real=real_to_real,
                plot_title=f"ICDs and BCD, {generator_src.source_name} vs {self.helper.real_images_src.source_name}",
            )
        ls = metric_ls.calculate()
        self.comparator_dict.update({name: ls})
        print("[INFO]: LS finished")

    def compute_c2st_knn(
        self, real_img: Dataset, generator_src: ImageSource, real_to_real: bool
    ) -> None:
        """
        Compute Classifier Two-Sample Test KNN
        """
        print(
            f"[INFO]: Start Calculation C2ST-KNN, Source = {generator_src.source_name}"
        )
        if self.helper.feature_extractor is None:
            if self.eval_cfg.feature_extractor == FeatureExtractor.InceptionV3:
                self.helper.feature_extractor = InceptionV3FE(last_pool=True)
            elif self.eval_cfg.feature_extractor == FeatureExtractor.VGGFaceResNet50:
                self.helper.feature_extractor = VGGFaceFE()
        name = (
            f"C2ST-{self.eval_cfg.c2st_k}NN"
            if not self.eval_cfg.c2st_k_adaptive
            else "C2ST Adaptive KNN"
        )
        if self.helper.c2st_real_features is None:
            c2st_metric = C2STKNN(
                name=name,
                feature_extractor=self.helper.feature_extractor,
                eval_config=self.eval_cfg,
                platform_config=self.platform_cfg,
                real_img=real_img,
                generated_img=generator_src.dataset,
                real_to_real=real_to_real,
                real_features=None,
            )
            self.helper.c2st_real_features = c2st_metric.get_real_features()
        else:
            c2st_metric = C2STKNN(
                name=name,
                feature_extractor=self.helper.feature_extractor,
                eval_config=self.eval_cfg,
                platform_config=self.platform_cfg,
                real_img=real_img,
                generated_img=generator_src.dataset,
                real_to_real=real_to_real,
                real_features=self.helper.c2st_real_features,
            )
        c2st_acc = c2st_metric.calculate()
        self.comparator_dict.update({f"{name} Accuracy": c2st_acc})
        self.comparator_dict.update(
            {f"{name} Normalized": -np.abs(2 * c2st_acc - 1) + 1}
        )  # normalization as per Guan et al. 2021
        print("[INFO]: C2ST-KNN finished")

    def compute_prd(
        self, real_img: Dataset, generator_src: ImageSource, real_to_real: bool
    ) -> None:
        """
        Computer Precision-Recall (PRD) by Sajjadi et al. (2018)
        """
        print(
            f"[INFO]: Start Calculation Precision-Recall (PRD), Source = {generator_src.source_name}"
        )
        if self.helper.feature_extractor is None:
            if self.eval_cfg.feature_extractor == FeatureExtractor.InceptionV3:
                self.helper.feature_extractor = InceptionV3FE(last_pool=True)
            elif self.eval_cfg.feature_extractor == FeatureExtractor.VGGFaceResNet50:
                self.helper.feature_extractor = VGGFaceFE()
        name = "PRD"
        prd_metric = PRD(
            name=name,
            feature_extractor=self.helper.feature_extractor,
            eval_config=self.eval_cfg,
            platform_config=self.platform_cfg,
            real_img=real_img,
            generated_img=generator_src.dataset,
            real_to_real=real_to_real,
            real_features=self.helper.prd_real_features
            if self.helper.prd_real_features is not None
            else None,
        )
        self.helper.prd_real_features = (
            prd_metric.get_real_features()
            if self.helper.prd_real_features is None
            else self.helper.prd_real_features
        )
        f8_max_prec, f8_max_rec, prs = prd_metric.calculate()
        self.comparator_dict.update({f"{name} F8 Max Precision": f8_max_prec})
        self.comparator_dict.update({f"{name} F8 Max Recall": f8_max_rec})
        if self.eval_cfg.prd_plot:
            self.helper.prd_mappings.precision_recall_pairs.append(prs)
            self.helper.prd_mappings.names.append(generator_src.source_name)
        print("[INFO]: Precision-Recall (PRD) finished")

    def compute_cleanfid(self, generator_src: ImageSource):
        """
        Compute CleanFID
        """
        print(
            f"[INFO]: Start Calculation Clean FID, Source = {generator_src.source_name}"
        )
        name = "Clean FID"
        metric_clean_fid = CleanFID(
            name=name,
            eval_config=self.eval_cfg,
            platform_config=self.platform_cfg,
            real_img_path=self.helper.real_images_src.folder_path,
            generated_img_path=generator_src.folder_path,
            feature_extractor_flag=self.eval_cfg.feature_extractor,
            real_features=self.helper.cleanfid_features
            if self.helper.cleanfid_features is not None
            else None,
        )
        clean_fid = metric_clean_fid.calculate()
        if self.helper.cleanfid_features is None:
            self.helper.cleanfid_features = metric_clean_fid.get_real_features()
        self.helper.cleanfid_fake_features = (
            metric_clean_fid.get_gen_features()
        )
        self.comparator_dict.update({name: clean_fid})
        print("[INFO]: Clean FID finished")

    def compute_fid(self, generator_src : ImageSource, is_fid_kid_base : IsFidKidBase, real_img : Dataset, generated_img : Dataset):
        """
        FID computation
        """
        print(
            f"[INFO]: Start Calculation FID, Source = {generator_src.source_name}"
        )
        name = "Frechet Inception Distance"
        metric_fid = FID(
            name=name,
            inception_base=is_fid_kid_base,
            real_img=real_img,
            generated_img=generated_img,
        )
        fid = metric_fid.calculate()
        self.comparator_dict.update({name: fid})
        print("[INFO]: FID finished")

    def get_real_images_src(self) -> ImageSource:
        """
        Getter for real image Source
        """
        return self.helper.real_images_src

    def get_all_generator_src(self) -> List[ImageSource]:
        """
        Return all generator sources
        """
        return self.helper.generated_images_srcs

    def get_generator_src(self, src_name: str) -> Union[ImageSource, None]:
        """
        Get generator source by name
        Returns None is not found
        """
        if self.helper.contains_generator(src_name):
            return self.helper.get_generator_src(src_name)
