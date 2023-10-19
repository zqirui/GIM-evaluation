import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import cv2
import torch
from tqdm import tqdm

def img_to_64(path: str, out_path: str, filename: str = "celeba64_train.npz", npz: bool = False) -> None:
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
    if npz:
        output_npz = np.array(npz)
        np.savez(os.path.join(out_path,filename), output_npz)
        print(
            f"{output_npz.shape} size array saved into celeba64_train.npz"
        )  # (x, 64, 64, 3)

def create_noise64_imgs(out_path: str, num_samples : int):
    """
    Create Normal Noise Images 64x64x3
    """
    noise = torch.randn(num_samples, 64, 64, 3)
    for i, img in enumerate(tqdm(noise, ascii=True, desc="[Create and Write Noise Imgs]:")):
        img = img.squeeze().numpy()
        img = cv2.normalize(img, None, alpha = 0, beta = 255, norm_type = cv2.NORM_MINMAX, dtype = cv2.CV_32F).astype(np.uint8)
        img = img
        cv2.imwrite(os.path.join(out_path, f"{i}.jpg"), img)



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
