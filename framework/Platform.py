from dataclasses import dataclass
from typing import List, Union
import os

from framework.ImageSource import ImageSource


@dataclass
class ManagerHelper:
    """
    Manager helper class, holding auxiliary variables
    """

    real_images_src: ImageSource
    generated_images_srcs: List[ImageSource]

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
        generated_images_path: str = os.path.join(os.getcwd(), "generated_images"),
        real_images_path: str = os.path.join(os.getcwd(), "original_images"),
    ) -> None:
        """
        Init constructor
        """
        real_img_src_name = "CelebA_Original"
        real_images_src = ImageSource(real_images_path, real_img_src_name)
        print(f"Real images found, name:{real_img_src_name}")
        # create one source per sub-folder in generated images parent folder
        generator_srcs = []
        subfolders = next(os.walk(generated_images_path))[1]
        print(f"{len(subfolders)} different generator sources found! Names:")
        for generator_folder_name in subfolders:
            print(generator_folder_name)
            generator_folder_path = os.path.join(
                generated_images_path, generator_folder_name
            )
            generated_src = ImageSource(generator_folder_path, generator_folder_name)
            generator_srcs.append(generated_src)

        self.helper = ManagerHelper(real_images_src, generator_srcs)

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
