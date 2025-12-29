import math
import os
import tensorflow as tf
import numpy as np

def downscale(original_image: np.array, ratio: int, scaling_snap: str = "round") -> np.array:
    """
    Given an image, downscales it using the specified ratio by averaging pixel colors in order to simulate a worse quality.

    Args:
        original_image (np.array): Original image
        ratio (int): Scaling ratio: result/original
        scaling_snap (str, optional): Determines whether the width and height are ceiled, floored or rounded if the result is not an integer. Defaults to 'round'.

    Returns:
        np.array: Image after downscaling
    """
    
    # Determine new shape
    if scaling_snap ==  "round":
        scaling_func = round
    elif scaling_snap == "ceil":
        scaling_func = math.ceil
    elif scaling_snap == "floor":
        scaling_func = math.floor
    else:
        raise KeyError(f'Invalid scaling_snap: "{scaling_snap}. Choose one of [round, ceil, floor]')

    h, w, d = original_image.shape

    new_h = scaling_func(h/ratio)
    new_w = scaling_func(w/ratio)

    # Recalculate de downscaled_image
    downscaled_image = np.zeros((new_h, new_w, d))

    for y in range(new_h):
        for x in range(new_w):
            y0 = y * ratio
            y1 = min((y + 1) * ratio, h)
            x0 = x * ratio
            x1 = min((x + 1) * ratio, w)

            chunk = original_image[y0:y1, x0:x1]
            downscaled_image[y, x] = chunk.mean(axis=(0, 1))

    # Convert the downscaled image back to the original data type
    downscaled_image = downscaled_image.astype(original_image.dtype)

    return downscaled_image


def load_image_paths(folder: str) -> list[str]:
    """
    Given a folder path, returns a list of image paths (only .jpg and .png files).

    Args:
        folder (str): Path to the folder containing images.

    Returns:
        list: List of image file paths in the folder.
    """
    return [
        os.path.join(folder, f)
        for f in os.listdir(folder)
        if f.endswith(".png") or f.endswith(".jpg")
    ]

def load_and_preprocess(path: str, hr_size: tuple, up_ratio: int) -> tuple:
    """
    Loads and preprocesses an image by reading the file, resizing to high resolution, and creating a low-resolution version.

    Args:
        path (str): File path of the image.
        hr_size (tuple): Desired high-resolution size (height, width).
        up_ratio (int): Upscaling ratio used to generate the low-resolution image.

    Returns:
        tuple: A tuple containing the low-resolution and high-resolution images.
    """
        
    img = tf.io.read_file(path)
    img = tf.image.decode_image(img, channels=3, expand_animations=False)
    img = tf.image.resize(img, hr_size)
    img = tf.cast(img, tf.float32) / 255.0

    lr = tf.image.resize(img, (hr_size[0] // up_ratio, hr_size[1] // up_ratio), method="area")

    return lr, img