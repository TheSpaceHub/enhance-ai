import numpy as np

def downscale(original_image: np.array, ratio: int, scaling_snap: str = "round") -> np.array:
    """Given an image, downscales it using the specified ratio by averaging pixel colors in order to simulate a worse quality.

    Args:
        original_image (np.array): Original image
        ratio (int): Scaling ratio: result/original
        scaling_snap (str, optional): Determines whether the width and height are ceiled, floored or rounded if the result is not an integer. Defaults to 'round'.

    Returns:
        np.array: Image after downscaling
    """
