from dataclasses import dataclass
from typing import List, Union
import os
import json

import numpy as np
import torch
from torch.utils.data import Dataset

from framework.image_source import ImageSource
from framework.configs import EvalConfig, PlatformConfig
from metrics.is_fid_kid.is_fid_kid import IsFidKidBase
from metrics.is_fid_kid.metrics import IS, FID, KID
from metrics.prc_metric import PRC
from metrics.fid_infty_metric import FID_infty
from metrics.is_infty_metric import IS_infty
from metrics.clean_fid_metric import CleanFID
from metrics.clean_kid_metric import CleanKID
from metrics.ls_metric import LS


@dataclass
class ResultDict:
    """
    Custom Dict for saving results
    """

    data: dict

    def add(self, key: str, value: dict):
        """
        Add a metric for one generator
        """
        self.data[key] = value

    def print(self, round_scores: bool = False):
        """
        Custom print function
        """
        for generator in self.data:
            print("------------------------------------")
            print(f"Generator Name: {generator}")
            print("------------------------------------")
            for metric in self.data[generator]:
                score = (
                    np.round(self.data[generator][metric], decimals=3)
                    if round_scores
                    else self.data[generator][metric]
                )
                print(f"{metric}: {score}")

    def write_to_json(self, file="default.json"):
        """
        Write dict out to json
        """
        json_obj = json.dumps(self.data)
        with open(file, "w") as outf:
            outf.write(json_obj)
        
    def read_from_json(self, file="default.json"):
        """
        Read dict from json
        """
        with open(file, "r") as f:
            self.data = json.load(f)   

@dataclass
class ManagerHelper:
    """
    Manager helper class, holding auxiliary variables
    """

    real_images_src: ImageSource
    generated_images_srcs: List[ImageSource]
    real_images_subsampled: Union[Dataset, None] = None
    ls_real_subset: Union[torch.Tensor.type, None] = None

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
            comparator_dict = {}
            generated_img = generator_src.get_dataset()
            print(f"[START]: Calculating Metrics for {generator_src.source_name}")

            # check for IS, FID, KID
            if self.eval_cfg.inception_score or self.eval_cfg.fid or self.eval_cfg.kid:
                is_fid_kid_base = IsFidKidBase(self.eval_cfg, self.platform_cfg)

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
                comparator_dict.update({name + " Mean": mean, name + " Std": std})
                print("[INFO]: IS finished")

            if self.eval_cfg.fid:
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
                comparator_dict.update({name: fid})
                print("[INFO]: FID finished")

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
                comparator_dict.update({name + " Mean": mean, name + " Std": std})
                print("[INFO]: KID finished")

            if self.eval_cfg.prc:
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
                        generated_img=generated_img,
                    )
                    self.helper.real_images_subsampled = (
                        metric_prc.get_real_subsampled_imgs()
                    )
                else:
                    # reuse same subsampled real dataset for all comparison models
                    metric_prc = PRC(
                        name=name,
                        eval_config=self.eval_cfg,
                        platform_config=self.platform_cfg,
                        real_img=self.helper.real_images_subsampled,
                        generated_img=generated_img,
                    )
                precision, recall, f1 = metric_prc.calculate()
                comparator_dict.update(
                    {"Precision": precision, "Recall": recall, "F1 Score": f1}
                )
                print("[INFO]: Improved PRC finished")

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
                )
                fid_infty = metric_fid_infty.calculate()
                comparator_dict.update({name: fid_infty})
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
                )
                is_infty = metric_is_infty.calculate()
                comparator_dict.update({name: is_infty})
                print("[INFO]: IS infinity finished")

            if self.eval_cfg.clean_fid:
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
                )
                clean_fid = metric_clean_fid.calculate()
                comparator_dict.update({name: clean_fid})
                print("[INFO]: Clean FID finished")

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
                )
                clean_kid = metric_clean_kid.calculate()
                comparator_dict.update({name: clean_kid})
                print("[INFO]: Clean KID finished")

            if self.eval_cfg.ls:
                print(
                    f"[INFO]: Start Calculation Likeliness Scores (LS), Source = {generator_src.source_name}"
                )
                name = "LS"
                real_to_real = True if self.helper.real_images_src.source_name == generator_src.source_name else False
                if self.eval_cfg.ls_n_samples > 0:
                    # single time calculation
                    if self.helper.ls_real_subset is None:
                        metric_ls = LS(
                            name=name,
                            eval_config=self.eval_cfg,
                            platform_config=self.platform_cfg,
                            real_img=self.helper.real_images_src.dataset,
                            generated_img=generator_src.dataset,
                            real_to_real=real_to_real,
                            plot_title=f"ICDs and BCD, {generator_src.source_name} vs {self.helper.real_images_src.source_name}"
                        )
                        # set on first calculation
                        self.helper.ls_real_subset = metric_ls.get_real_subset()
                    else:
                        metric_ls = LS(
                            name=name,
                            eval_config=self.eval_cfg,
                            platform_config=self.platform_cfg,
                            real_img=self.helper.real_images_src.dataset,
                            generated_img=generator_src.dataset,
                            real_to_real=real_to_real,
                            plot_title=f"ICDs and BCD, {generator_src.source_name} vs {self.helper.real_images_src.source_name}",
                            real_down_t=self.helper.ls_real_subset
                        )
                else:
                    # k fold
                    metric_ls = LS(
                        name=name,
                        eval_config=self.eval_cfg,
                        platform_config=self.platform_cfg,
                        real_img=self.helper.real_images_src.dataset,
                        generated_img=generator_src.dataset,
                        real_to_real=real_to_real,
                        plot_title=f"ICDs and BCD, {generator_src.source_name} vs {self.helper.real_images_src.source_name}"
                    )
                ls = metric_ls.calculate()
                comparator_dict.update({name: ls})
                print("[INFO]: LS finished")

            print(f"[FINISHED]: Calculating Metrics for {generator_src.source_name}")
            print(comparator_dict)
            self.out_dict.add(key=generator_src.source_name, value=comparator_dict)

        return self.out_dict

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
