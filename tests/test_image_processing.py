import numpy as np
import pytest
from PIL import Image
from src.image_processing import downscale

def test_downscaling():
    """
    Tests if image_processing.downscale works appropriately
    """
    images = ["Img1", "Img2"]
    ratios = [4, 11]
    scaling_snaps = ["floor", "round", "ceil"]

    for image in images:
        for ratio in ratios:
            for scaling_snap in scaling_snaps:

                # Load and downscale original image
                original_image = np.array(Image.open(f"./tests/images/image_processing/{image}.png"))
                downscaled_image = downscale(original_image, ratio, scaling_snap)

                # Load expected downscaled image
                expected_image = np.array(Image.open(f"./tests/images/image_processing/expected/{image}_{ratio}_{scaling_snap}.png"))

                # Compare
                assert expected_image.shape == downscaled_image.shape
                assert np.allclose(expected_image, downscaled_image)

    # Test scaling_snap KeyError
    with pytest.raises(KeyError):
        image_array = np.array(Image.open(f"./tests/images/image_processing/{images[0]}.png"))
        downscale(image_array, 20, "foo")