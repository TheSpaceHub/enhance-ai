import keras
import tensorflow as tf
import architectures as archs
from image_processing import load_image_paths, load_and_preprocess
import keras.applications.vgg19 as vgglib
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import time
import math
import csv
from pathlib import Path

def load_vgg(path: str) -> keras.Model:
    """Loads VGG19 model from local memory if it exists; otherwise downloads it from the Internet.

    Args:
        path (str): Path to read / download.

    Returns:
        keras.Model: VGG19 model.
    """
    if os.path.exists(path):
        return keras.models.load_model(path)
    else:
        vgg = vgglib.VGG19(
            include_top=False, weights="imagenet", input_shape=(None, None, 3)
        )
        vgg.save(path)
        return vgg


@tf.function(reduce_retracing=True)
def get_mae_loss(y_true, y_pred) -> float:
    """Returns Mean Average Error.

    Args:
        y_true: True image.
        y_pred: Prediction.

    Returns:
        float: MAE.
    """
    return float(tf.reduce_mean(tf.abs(y_true - y_pred)))


@tf.function(reduce_retracing=True)
def get_perceptual_loss(y_true, y_pred, vgg_features) -> float:
    """Returns MAE of a deep layer of the VGG.

    Args:
        y_true: True image.
        y_pred: Prediction.
        vgg_features: Part of VGG which outputs feature maps.

    Returns:
        float: Perceptual loss.
    """
    # Scale to 255
    y_true_scaled = y_true * 255.0
    y_pred_scaled = y_pred * 255.0

    # Preprocess for vgg
    y_true_pre = vgglib.preprocess_input(y_true_scaled)
    y_pred_pre = vgglib.preprocess_input(y_pred_scaled)

    # Get features
    features_true = vgg_features(y_true_pre)
    features_pred = vgg_features(y_pred_pre)

    # Get MAE
    return float(
        tf.cast(tf.reduce_mean(tf.abs(features_true - features_pred)), dtype=tf.float32)
    )

@tf.function
def get_metrics(
    models: list[keras.Model],
    lr_images: list,
    images: list,
    vgg_features,
) -> tuple[list, list, list]:
    """Given a list of models and images, returns three metrics:
    - Total MAE
    - Total perceptual loss
    - Total runtime

    Args:
        models (list[keras.Model]): List of keras models to evaluate.
        lr_images (list): List of low resolution images.
        images (list): List of high resolution images.
        vgg_features (_type_): Part of VGG which outputs feature maps.

    Returns:
        tuple[list, list, list]: Tuple containing total MAE, perceptual loss and runtime for each model.
    """
    # Set metrics
    mae_loss = []
    perceptual_loss = []
    running_times = []

    # Calculate for every model
    for i, model in enumerate(models):
        print("  " * 2, f"Predicting with model {i}")

        model_mae = 0
        model_perc = 0

        # Set up time tracking
        runtime = 0
        start_time = time.time()  # Keep it in proper scope

        # Run through every image
        print("  " * 2, "0/" + str(len(images)), end="")
        for j, img in enumerate(lr_images):
            # Add "batch dimension" for model
            tensor_lr_image = tf.expand_dims(img, axis=0)
            tensor_image = tf.expand_dims(images[j], axis=0)

            # Get prediction and calculate losses
            start_time = time.time()
            prediction = model(tensor_lr_image)
            runtime += time.time() - start_time

            model_mae += get_mae_loss(
                y_pred=prediction,
                y_true=tensor_image,
            )
            model_perc += get_perceptual_loss(
                y_pred=prediction,
                y_true=tensor_image,
                vgg_features=vgg_features,
            )
            print("\r", "  " * 2, f"{str(j + 1)}/{str(len(images))}", end="")

            # Delete generated images, as they are not needed and simply occupy memory
            del prediction
            del tensor_lr_image
            del tensor_image
        print()

        # Add metrics
        mae_loss.append(model_mae)
        perceptual_loss.append(model_perc)
        running_times.append(runtime)

    return mae_loss, perceptual_loss, running_times


def plot_metrics(
    mae_losses: list,
    perceptual_losses: list,
    runtimes: list[float],
    names: list[str],
):
    """Given the metrics, plots them in a 2x2 grid (with an additional Runtime vs Perceptual loss plot).

    Args:
        mae_losses (list): Average MAE for each model.
        perceptual_losses (list): Average perceptual loss for each model.
        runtimes (list[float]): Average runtime for each model.
        names (list[str]): Model names.
    """
    # Create subplots
    _, axs = plt.subplots(2, 2, figsize=(10, 8))

    axs[0, 0].bar(names, mae_losses, color="skyblue")
    axs[0, 0].set_title("MAE")

    axs[0, 1].bar(names, perceptual_losses, color="salmon")
    axs[0, 1].set_title("Perceptual loss (MAE)")

    axs[1, 0].bar(names, runtimes, color="lightgreen")
    axs[1, 0].set_title("Average runtime")

    axs[1, 1].scatter(runtimes, perceptual_losses, color="gold")
    axs[1, 1].set_title("Runtime vs Perceptual loss (MAE)")
    for x, y, name in zip(runtimes, perceptual_losses, names):
        # Add names to points
        plt.annotate(name, (x, y), textcoords="offset points", xytext=(5, 5), ha="left")

    plt.savefig("results/model_metrics.png")
    plt.show()

def save_metrics_csv(
    mae_losses,
    perceptual_losses,
    runtimes,
    names: list[str],
):
    """Saves a csv with all the metrics given by the input.

    Args:
        mae_losses (list): Average MAE for each model.
        perceptual_losses (list): Average perceptual loss for each model.
        runtimes (list[float]): Average runtime for each model.
        names (list[str]): Model names.
    """

    # Set the path
    output_path = Path("results/model_metrics.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["model", "mae", "perceptual_loss", "runtime_sec"])

        for i, name in enumerate(names):
            writer.writerow([
                name,
                float(mae_losses[i]),
                float(perceptual_losses[i]),
                float(runtimes[i]),
            ])
            
def main():
    # Define constants
    BATCH_SIZE = 64
    IMAGES_PATH = "data/DIV2K_train_HR/"
    MODEL_FOLDER_PATH = "models/"
    MODEL_NAMES = [
        "cnnu_e30_sc4.keras",
        "espcn_e30_sc4.keras",
        "srrn_e30_sc4_rb8f64.keras",
        "srgan_e30_sc4_rb8f64_l005.keras",
        ]
    MAX_DATASET_SIZE = 1

    # Set matplotlib font to be small
    plt.rcParams.update({"font.size": 8})

    # Load VGG
    vgg = load_vgg(os.path.join(MODEL_FOLDER_PATH, "vgg19.keras"))
    vgg_features = keras.Model(
        inputs=vgg.input, outputs=vgg.get_layer("block5_conv4").output
    )

    # Load models
    models = []
    model_labels = []
    for name in MODEL_NAMES:
        try:
            model = keras.models.load_model(os.path.join(MODEL_FOLDER_PATH, name))
            models.append(model)
            model_labels.append(model.name)
            
        except Exception as e:
            print(f"Could not load model {name}: {str(e)}")

    # Batch processing (holding all images in RAM is not viable for a large dataset)
    img_paths = load_image_paths(IMAGES_PATH)[:MAX_DATASET_SIZE]

    mae_losses = np.zeros(len(models))
    perceptual_losses = np.zeros(len(models))
    runtimes = np.zeros(len(models))

    batch_count = 0
    while batch_count * BATCH_SIZE < len(img_paths):
        batch_paths = img_paths[
            batch_count * BATCH_SIZE : (batch_count + 1) * BATCH_SIZE
        ]
        batch_count += 1
        print(f"Batch {batch_count}/{math.ceil(len(img_paths) / BATCH_SIZE)}")

        # Load images in /4 and original size
        lr_images = []
        images = []
        print("  ", "Loading images")
        print("  ", "0/" + str(len(batch_paths)), end="")
        for i, path in enumerate(batch_paths):
            lr_img, img = load_and_preprocess(path, up_ratio=4)
            lr_images.append(lr_img)
            images.append(img)
            print("\r  ", f"{str(i + 1)}/{str(len(batch_paths))}", end="")
        print()

        # Get predictions and times
        mae, perceptual, running_times = get_metrics(
            models=models,
            lr_images=lr_images,
            images=images,
            vgg_features=vgg_features,
        )

        mae_losses += np.array(mae)
        perceptual_losses += np.array(perceptual)
        runtimes += np.array(running_times)

        # Free memory
        del images

    # Average information
    mae_losses /= len(img_paths)
    perceptual_losses /= len(img_paths)
    runtimes /= len(img_paths)
    
    # Ensure the output directory for the results exists
    output_dir = 'results'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir) 

    # Plot and save metrics
    plot_metrics(mae_losses, perceptual_losses, runtimes, model_labels)
    
    # Save in csv
    save_metrics_csv(
        names=model_labels,
        mae_losses=mae_losses,
        perceptual_losses=perceptual_losses,
        runtimes=runtimes,
    )


if __name__ == "__main__":
    main()
