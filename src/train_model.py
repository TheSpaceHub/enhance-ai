import tensorflow as tf
import matplotlib.pyplot as plt
from architectures import ESPCN
from image_processing import load_and_preprocess, load_image_paths
from keras.callbacks import TensorBoard
import keras
import datetime
import sys
from pathlib import Path


def build_dataset(
    image_paths: list[str],
    hr_size: tuple,
    up_ratio: int,
    batch_size: int,
    training=True,
) -> tf.data.Dataset:
    """Given the params, builds the dataset.

    Args:
        image_paths (list[str]): List of image paths.
        hr_size (tuple): Size of high-res images.
        up_ratio (int): Desired upscaling ratio.
        batch_size (int): Batch size for the dataset.
        training (bool, optional): Determines if the dataset will be used for training. Defaults to True.

    Returns:
        tf.data.Dataset: The TensorFlow dataset.
    """
    ds = tf.data.Dataset.from_tensor_slices(image_paths)

    if training:
        # If this dataset is used for training, we shuffle the data to randomize split
        ds = ds.shuffle(buffer_size=len(image_paths))

    ds = ds.map(
        lambda p: load_and_preprocess(p, hr_size, up_ratio),
        num_parallel_calls=tf.data.AUTOTUNE,
    )

    # Split data into batches
    ds = ds.batch(batch_size, drop_remainder=True)
    ds = ds.prefetch(tf.data.AUTOTUNE)

    return ds


def show_example(
    model: keras.Model,
    dataset: tf.data.Dataset,
    image_path: str = None,
    hr_size: tuple = (256, 256),
    up_ratio=4,
):
    """Shows an example of the prediction, along with the lr and hr versions of the image.

    Args:
        model (keras.Model): Keras model to use for prediction.
        dataset (tf.data.Dataset): TensorFlow dataset from which to pick image (if path was not provided).
        image_path (str, optional): Path of image to test. Defaults to None.
        hr_size (tuple, optional): Size of high res image. Defaults to (256, 256).
        up_ratio (int, optional): Upscaling ratio. Defaults to 4.
    """

    if image_path is not None:
        # Load image
        lr, hr = load_and_preprocess(image_path, hr_size, up_ratio)
        # Add dim for tf
        lr = tf.expand_dims(lr, axis=0)

    else:
        # Pick random image
        for lr_batch, hr_batch in dataset.take(1):
            lr = lr_batch[0:1]
            hr = hr_batch[0]

    # Predict
    sr = model(lr, training=False)[0]

    # To numpy
    lr_np = lr[0].numpy()
    sr_np = sr.numpy()
    hr_np = hr.numpy()

    plt.figure(figsize=(12, 4))

    plt.subplot(1, 3, 1)
    plt.title("Low Resolution (Input)")
    plt.imshow(lr_np)
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.title("Super Resolution (Prediction)")
    plt.imshow(sr_np)
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.title("High Resolution (Target)")
    plt.imshow(hr_np)
    plt.axis("off")

    plt.show()


def set_up_logging(model_name: str = "srmodel") -> TensorBoard:
    """Creates logging folder and TensorBoard object.

    Args:
        name (str, optional): Name of the model. Defaults to "srmodel".

    Returns:
        TensorBoard: Callback to pass to fit.
    """
    PROJECT_ROOT = Path.cwd().parent
    sys.path.insert(0, str(PROJECT_ROOT))

    # Create a unique folder for this run
    log_dir = (
        "logs/train/" + model_name + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    )
    tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)

    # Launch it
    tensorboard_url = tensorboard_callback.launch()

    # Let the user know how to access
    print(f"Get real-time statistics at {tensorboard_url}")

    return tensorboard_callback


def main():
    # Constants
    UP_RATIO = 4
    BATCH_SIZE = 32
    EPOCHS = 30
    HR_SIZE = (256, 256)
    DATA_FOLDER = "data/DIV2K_train_HR/DIV2K_train_HR/"
    MODEL_NAME = "ESPCN"

    # Create, build and compile model
    model = ESPCN(up_ratio=UP_RATIO)
    model.build((None, HR_SIZE[0], HR_SIZE[1], 3))
    model.compile(optimizer="adam", loss="mae", jit_compile=False)

    # Show summary
    model.summary()

    # Get dataset
    image_paths = load_image_paths(DATA_FOLDER)
    train_ds = build_dataset(image_paths, HR_SIZE, UP_RATIO, BATCH_SIZE, training=True)

    # Set up logs and real-time visualization
    tensorboard_callback = set_up_logging(MODEL_NAME)

    history = model.fit(
        train_ds,
        epochs=EPOCHS,
        callbacks=[tensorboard_callback],
    )

    # Save
    model.save(f"models/{MODEL_NAME}")


if __name__ == "__main__":
    main()
