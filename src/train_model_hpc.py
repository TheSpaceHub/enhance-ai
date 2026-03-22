import os
import tensorflow as tf
import hpc_archs
from image_processing import load_and_preprocess, load_image_paths

# Initialize the strategy
strategy = tf.distribute.MirroredStrategy()
print(f"Number of GPUs in sync: {strategy.num_replicas_in_sync}")
GLOBAL_BATCH_SIZE = 64 * 4


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

    def map_fn(p):
        return load_and_preprocess(p, hr_size, up_ratio)

    ds = ds.map(
        map_fn,
        num_parallel_calls=tf.data.AUTOTUNE,
    )

    # Split data into batches
    ds = ds.batch(batch_size, drop_remainder=True)
    ds = ds.prefetch(tf.data.AUTOTUNE)

    return ds


def get_model(name: str):
    if name == "CNNU_x2":
        return hpc_archs.CNNUpscaler(2)
    elif name == "CNNU_x4":
        return hpc_archs.CNNUpscaler(4)
    elif name == "ESPCN_x2":
        return hpc_archs.ESPCN(2)
    elif name == "ESPCN_x4":
        return hpc_archs.ESPCN(4)
    elif name == "SRRN_x2":
        return hpc_archs.SRRN(up_ratio=2, num_blocks=8, filters=64)
    elif name == "SRRN_x4":
        return hpc_archs.SRRN(up_ratio=4, num_blocks=8, filters=64)
    elif name == "SRGAN_x2":
        return hpc_archs.SRGAN(2, filters=8, num_blocks=4)
    elif name == "SRGAN_x4":
        return hpc_archs.SRGAN(4, filters=64, num_blocks=16)
    raise ValueError(f"Unknown model name: {name}")


def train_model(name: str, ratio: int):

    EPOCHS = 100
    HR_SIZE = (256, 256)
    DATA_FOLDER = "data/DIV2K_train_HR/DIV2K_train_HR/"
    MODEL_PATH = f"models/keras_models/{name}.keras"
    # Get dataset
    image_paths = load_image_paths(DATA_FOLDER)
    train_ds = build_dataset(
        image_paths, HR_SIZE, ratio, GLOBAL_BATCH_SIZE, training=True
    )

    with strategy.scope():
        model = get_model(name)
        
        if "SRGAN" in name:
            model.compile(
                gen_optimizer=tf.keras.optimizers.Adam(),
                disc_optimizer=tf.keras.optimizers.Adam()
            )
        else:
            model.compile(optimizer="adam", loss="mae")

        # Execute one forward pass to build the model properly
        for lr, hr in train_ds.take(1):
            model(lr)
            if "SRGAN" in name:
                model.discriminator(hr)

    # Show summary
    model.summary()

    history = model.fit(
        train_ds,
        epochs=EPOCHS,
    )

    # Save
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    model.save(MODEL_PATH)
    print(f"Model saved successfully to: {MODEL_PATH}")


def main():
    model_names = [
        #"CNNU_x2",
        #"CNNU_x4",
        #"ESPCN_x2",
        #"ESPCN_x4",
        #"SRRN_x2",
        #"SRRN_x4",
        "SRGAN_x2",
        #"SRGAN_x4",
    ]

    for name in model_names:
        up_ratio = int(name.split("_x")[1])

        print(f"\n--- Training {name} with up_ratio {up_ratio} ---")
        train_model(name, up_ratio)

        tf.keras.backend.clear_session()


if __name__ == "__main__":
    main()
