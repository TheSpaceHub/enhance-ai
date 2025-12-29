import keras
import tensorflow as tf
from keras import layers
from .average import Upscaler

@keras.saving.register_keras_serializable()
class CNNUpscaler(keras.Model):
    """Convolutional Neural Network for image upscaling. It's just a simple model: upscales first, then applies convolution corrections."""

    def __init__(self, up_ratio: float, name="cnnupscaler", **kwargs):
        super().__init__(name=name, **kwargs)
        self.up_ratio = up_ratio
        
        self.upscaler = Upscaler(up_ratio)
        
        self.conv1 = layers.Conv2D(64, (3, 3), activation="relu", padding="same")
        self.conv2 = layers.Conv2D(32, (3, 3), activation="relu", padding="same")
        self.conv3 = layers.Conv2D(3, (3, 3), padding="same")

    def call(self, inputs):
        # Upscale first
        x_up = self.upscaler(inputs)

        # Calculate the correction factor
        x = self.conv1(x_up)
        x = self.conv2(x)
        correction = self.conv3(x)
        return x_up + correction

    def get_config(self):
        config = super().get_config()
        config.update({"up_ratio": self.up_ratio})
        return config