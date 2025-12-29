import numpy as np
import keras
import tensorflow as tf
from keras import ops
from keras import layers

@keras.saving.register_keras_serializable()
class Upscaler(layers.Layer):
    """Upscales images by superposing grids and averaging colors."""

    def __init__(self, up_ratio: float, name="upscaler", **kwargs):
        super().__init__(name=name, **kwargs)
        self.up_ratio = up_ratio

    def call(self, inputs):
        shape = tf.shape(inputs)
        height = shape[1]
        width = shape[2]

        # We round the new width and height casting twice (tf is weird)
        new_height = tf.cast(height, tf.float32) * self.up_ratio
        new_width = tf.cast(width, tf.float32) * self.up_ratio
        
        new_height = tf.cast(new_height, tf.int32)
        new_width = tf.cast(new_width, tf.int32)

        # Resize
        return tf.image.resize(inputs, [new_height, new_width], method='area')

    def get_config(self):
        config = super().get_config()
        config.update({"up_ratio": self.up_ratio})
        return config

@keras.saving.register_keras_serializable()
class Average(keras.Model):
    """Defines a model which upscales images by averaging. Training does not modify its behavior"""

    def __init__(self, up_ratio=2.0, name="average", **kwargs):
        super().__init__(name=name, **kwargs)
        self.up_ratio = up_ratio
        self.upscaler = Upscaler(up_ratio)

    def call(self, inputs):
        return self.upscaler(inputs)

    def get_config(self):
        config = super().get_config()
        config.update({"up_ratio": self.up_ratio})
        return config
