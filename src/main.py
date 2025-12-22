import architectures as archs
import keras
import numpy as np
import matplotlib.pyplot as plt


def main():
    print("Main loop running")
    avg = archs.Average(up_ratio=4.5)

    avg.build((None, None, None, 3))
    avg.compile(
        optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
    )

    img_small = np.random.rand(1, 400, 640, 3).astype("float32")

    out_small = avg(img_small)

    plt.imshow(img_small[0])
    plt.axis('off')
    plt.show()
    plt.imshow(out_small.numpy()[0])
    plt.axis('off')
    plt.show()
    


if __name__ == "__main__":
    main()
