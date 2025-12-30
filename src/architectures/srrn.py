import keras
import tensorflow as tf
from keras import layers

@keras.saving.register_keras_serializable()
class ResidualBlock(layers.Layer):
    """Basic residual block: Conv -> ReLU"""
    def __init__(self, filters, **kwargs):
        super().__init__(**kwargs)
        self.conv1 = layers.Conv2D(filters, 3, padding='same')
        self.relu = layers.ReLU()
        self.conv2 = layers.Conv2D(filters, 3, padding='same')

    def call(self, x, training=False):
        residual = x
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        return x + residual

@keras.saving.register_keras_serializable()
class SRRN(keras.Model):
    """
    Super-Resolution Residual Network. A more sofisticated post-upscaler Network.
    Uses residual blocks to refine features in LR space, then performs upsampling via PixelShuffle.
    """

    def __init__(self, up_ratio, filters=64, num_blocks=8, name="SRRN", **kwargs):
        super().__init__(name=name, **kwargs)
        self.up_ratio = up_ratio

        # Intial conv
        self.conv_in = layers.Conv2D(filters, 9, padding='same')
        self.relu = layers.ReLU()

        # Residual blocks
        self.res_blocks = [ResidualBlock(filters) for _ in range(num_blocks)]

        # Post residual
        self.conv_post_res = layers.Conv2D(filters, 3, padding='same')

        # Upsampling
        self.upsample = layers.Conv2D(filters * (up_ratio**2), 3, padding='same')
        self.pixel_shuffle = layers.Lambda(lambda x: tf.nn.depth_to_space(x, up_ratio))

        # Final conv
        self.conv_final = layers.Conv2D(3, 9, padding='same', activation='sigmoid')

    def call(self, x, training=False):
        x = self.conv_in(x)
        x = self.relu(x)

        # Residual blocks
        res = x
        for block in self.res_blocks:
            x = block(x, training=training)
        x = self.conv_post_res(x)
        x = layers.add([res, x])

        # Upsampling
        x = self.upsample(x)
        x = self.pixel_shuffle(x)

        return self.conv_final(x)

    def get_config(self):
        config = super().get_config()
        config.update({"up_ratio": self.up_ratio})
        return config