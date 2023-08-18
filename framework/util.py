import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import cv2


def img_to_64(path: str, out_path: str, filename: str = "celeba64_train.npz") -> None:
    """
    Preprocess image to 64x64 resolution
    Reference: https://github.com/forever208/DDPM-IP/blob/DDPM-IP/datasets/celeba64_npz.py#L35
    """
    npz = []

    for img in os.listdir(path):
        img_arr = cv2.imread(path + img)
        resized_img = cv2.resize(img_arr, (64, 64))
        npz.append(resized_img)
        cv2.imwrite(os.path.join(out_path, img), resized_img)

    output_npz = np.array(npz)
    np.savez(os.path.join(out_path,filename), output_npz)
    print(
        f"{output_npz.shape} size array saved into celeba64_train.npz"
    )  # (x, 64, 64, 3)


def show_images(path: str):
    """
    Reference: https://github.com/forever208/DDPM-IP/blob/DDPM-IP/datasets/celeba64_npz.py#L35
    """
    x = np.load(os.path.join(path, "celeba64_train.npz"))["arr_0"]
    plt.figure(figsize=(10, 10))
    for i in range(16):
        img = x[i, :, :, :]
        plt.subplot(4, 4, i + 1)
        plt.imshow(img)
        plt.axis("off")
    # plt.savefig('./imgnet32_samples_4.jpg')
    plt.show()
