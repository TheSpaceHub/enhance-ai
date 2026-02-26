import os

os.environ["TF_USE_LEGACY_KERAS"] = "1"  # work with quantization
import tensorflow as tf
from architectures import quantizable_archs as qarchs
from image_processing import load_and_preprocess, load_image_paths
import tf2onnx
import onnx
import tensorflow_model_optimization as tfmot
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


def main():
    # Constants
    UP_RATIO = 4
    BATCH_SIZE = 16
    FILTERS = 16
    NUM_BLOCKS = 4
    EPOCHS = 1
    HR_SIZE = (256, 256)
    DATA_FOLDER = "data/DIV2K_train_HR/DIV2K_train_HR/"
    MODEL_NAME = "cnnu"
    TFLITE_PATH = f"models/tflite/{MODEL_NAME}_strict_int8.tflite"

    # Create, build and compile model

    model = qarchs.build_cnn_upscaler(
        up_ratio=UP_RATIO,
        input_shape=(HR_SIZE[0] // UP_RATIO, HR_SIZE[1] // UP_RATIO, 3),
    )
    model.compile(optimizer="adam", loss="mae", jit_compile=False)

    # Make it quantization aware
    quantize_model = tfmot.quantization.keras.quantize_model
    q_aware_model = quantize_model(model)

    q_aware_model.compile(optimizer="adam", loss="mae")  # recompile

    # Show summary
    q_aware_model.summary()

    # Get dataset
    image_paths = load_image_paths(DATA_FOLDER)
    train_ds = build_dataset(image_paths, HR_SIZE, UP_RATIO, BATCH_SIZE, training=True)

    history = q_aware_model.fit(
        train_ds,
        epochs=EPOCHS,
    )

    # Save
    converter = tf.lite.TFLiteConverter.from_keras_model(q_aware_model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]

    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]

    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8

    def representative_dataset():
        for lr, hr in train_ds.unbatch().batch(1).take(100):
            yield [tf.cast(lr, tf.float32)]

    converter.representative_dataset = representative_dataset

    tflite_quant_model = converter.convert()

    with open(TFLITE_PATH, "wb") as f:
        f.write(tflite_quant_model)

    print(f"Quantized model saved successfully to: {TFLITE_PATH}")


if __name__ == "__main__":
    main()
