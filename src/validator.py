import keras
import tensorflow as tf
import architectures as archs
import keras.applications.vgg19 as vgglib
import numpy as np
import os


def load_vgg(path: str) -> keras.Model:
    if os.path.exists(path):
        return keras.models.load_model(path)
    else:
        vgg = vgglib.VGG19(
            include_top=False, weights="imagenet", input_shape=(None, None, 3)
        )
        vgg.save(path)
        return vgg


def pixel_mae_loss(y_true, y_pred) -> float:
    return float(tf.reduce_mean(tf.abs(y_true - y_pred)))


def calculate_perceptual_loss(y_true, y_pred, vgg_features) -> float:
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


def main():
    print("main")
    vgg = load_vgg("models/vgg19.keras")
    vgg_features = keras.Model(
        inputs=vgg.input, outputs=vgg.get_layer("block5_conv4").output
    )
    i1 = np.random.rand(1, 128, 128, 3)
    i2 = i1 + np.random.rand(1, 128, 128, 3) / 255
    print(calculate_perceptual_loss(i1, i2, vgg_features))


if __name__ == "__main__":
    main()
