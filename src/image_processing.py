import math
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
