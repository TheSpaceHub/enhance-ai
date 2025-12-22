import architectures as archs
import keras
import numpy as np
import matplotlib.pyplot as plt


def plot_results(in_image: np.array, out_image: np.array) -> None:
    """
    Plots the input and output images side by side.

    Args:
        in_image (np.array): Low-resolution input image batch.
        out_image (np.array): Upscaled output image batch.
    """
    plt.subplot(1, 2, 1)
    plt.title("Input (Small)")
    plt.imshow(in_image[0])
    
    plt.subplot(1, 2, 2)
    plt.title("Output (Upscaled)")
    plt.imshow(out_image[0])
    plt.show()

def test_avg(up_ratio: float) -> None:
    avg = archs.Average(up_ratio=up_ratio)

    avg.build((None, None, None, 3))
    avg.compile(
        optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
    )

    img_small = np.random.rand(1, 400, 640, 3).astype("float32")

    out_small = avg(img_small)

    plot_results(img_small, out_small.numpy())

def test_cnn(up_ratio: float):
    cnnupscale = archs.CNNUpscaler(up_ratio=up_ratio) 

    cnnupscale.build((None, None, None, 3))
    cnnupscale.compile(
        optimizer="adam", loss="mse", metrics=["accuracy"]
        )

    img_small = np.random.rand(1, 64, 98, 3).astype("float32")

    out_small = cnnupscale(img_small)

    print(f"Forma original: {img_small.shape}")
    print(f"Forma de salida: {out_small.shape}")

    plot_results(img_small, out_small.numpy())

def main():
    print("Main loop running")
    test_cnn(1.5)

if __name__ == "__main__":
    main()
