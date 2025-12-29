import tensorflow as tf
import keras
from keras import layers


@keras.saving.register_keras_serializable()
class ChannelAttention(layers.Layer):
    """
    Channel attention https://arxiv.org/pdf/1807.02758
    """

    def __init__(self, channels: int, reduction: int = 16, **kwargs):
        super().__init__(**kwargs)
        self.channels = channels
        self.reduction = reduction

        # Global Average Pooling
        self.gap = layers.GlobalAveragePooling2D()

        # Little bit of dense
        self.dense1 = layers.Dense(channels // reduction, activation="relu")
        self.dense2 = layers.Dense(channels, activation="sigmoid")

        # Reshape
        self.reshape = layers.Reshape((1, 1, channels))

    def call(self, inputs):
        x = self.gap(inputs)
        x = self.dense1(x)
        x = self.dense2(x)
        x = self.reshape(x)

        return inputs * x

    def get_config(self):
        config = super().get_config()
        config.update({"channels": self.channels, "reduction": self.reduction})
        return config


@keras.saving.register_keras_serializable()
class ReswishLayer(layers.Layer):
    """
    A ReswishLayer is a residual block which aims to combine convolutions with trainable activation layers in order to produce complex models capable of learning different behaviors.
    """

    def __init__(self, filters: int, **kwargs):
        """Generates the layers in a ReswishLayer.

        Args:
            filters (int): Amount of filters wanted after convoluting
        """
        super().__init__(**kwargs)

        # Save stuff for config
        self.filters = filters

        # 2D Convolution
        self.conv = layers.Conv2D(filters, (3, 3), padding="same")
        # Swish it
        self.sw1 = layers.Activation("swish")

        # Conv again
        self.conv2 = layers.Conv2D(filters, (3, 3), padding="same")
        
        # Channel attention
        self.ca = ChannelAttention(filters, reduction=16)

    def call(self, inputs):
        res = inputs

        x = self.conv(inputs)
        x = self.sw1(x)
        x = self.conv2(x)
        x = self.ca(x)

        return layers.add([res, x])

    def get_config(self):
        config = super().get_config()
        config.update({"filters": self.filters})
        return config


@keras.saving.register_keras_serializable()
class PixelShuffleUpscale(layers.Layer):
    """
    Upscales by rearranging additional channels into pixels
    """

    def __init__(self, shuffle_num: int, filters: int, **kwargs):
        super().__init__(**kwargs)
        # Save stuff for config
        self.filters = filters
        self.shuffle_num = shuffle_num

        # Actual shuffle layers
        self.conv = layers.Conv2D(
            filters * shuffle_num * shuffle_num, (3, 3), padding="same"
        )

    def call(self, inputs):
        x = self.conv(inputs)
        x = tf.nn.depth_to_space(x, block_size=self.shuffle_num)
        return x

    def get_config(self):
        config = super().get_config()
        config.update({"filters": self.filters, "shuffle_num": self.shuffle_num})
        return config


@keras.saving.register_keras_serializable()
class ColorConstraint(layers.Layer):
    """
    Calculates the color consistency error and adds it to the model's loss
    without modifying the image pixels directly.
    """

    def __init__(self, up_ratio: int, weight: float = 0.5, **kwargs):
        super().__init__(**kwargs)
        self.up_ratio = up_ratio
        self.weight = weight

    def call(self, inputs):
        # inputs must be [gen, og]
        gen, og = inputs

        # Downscale to compare
        gen_downscaled = tf.nn.avg_pool2d(
            gen, ksize=self.up_ratio, strides=self.up_ratio, padding="VALID"
        )

        # Get error
        color_error = tf.reduce_mean(tf.abs(og - gen_downscaled))

        # Add loss
        self.add_loss(color_error * self.weight)

        return gen

    def compute_output_shape(self, input_shape):
        return input_shape[0]

    def get_config(self):
        config = super().get_config()
        config.update({"up_ratio": self.up_ratio, "weight": self.weight})
        return config


@keras.saving.register_keras_serializable()
class Reswish(keras.Model):
    """Defines a model using ReswishLayer architecture."""

    def __init__(
        self,
        filters: int = 64,
        blocks: int = 4,
        up_ratio: int = 5,
        name="Reswish",
        **kwargs
    ):
        # Save stuff for config
        self.filters = filters
        self.blocks = blocks
        self.up_ratio = up_ratio

        # Declare input (RGB images)
        inputs = layers.Input(shape=(None, None, 3))

        # Expand to #filters
        x = layers.Conv2D(filters, (3, 3), padding="same")(inputs)

        # Keep residual
        res = x

        # Add ReswishLayers
        for _ in range(blocks):
            x = ReswishLayer(filters)(x)

        # Extra conv to keep it on its toes
        x = layers.Conv2D(filters, (3, 3), padding="same")(x)

        # Add res back
        x = layers.add([x, res])

        # Upscale
        x = PixelShuffleUpscale(filters=filters, shuffle_num=up_ratio)(x)

        # Collapse to RGB
        x = layers.Conv2D(3, (3, 3), padding="same", activation=None)(x)

        # Correct
        outputs = ColorConstraint(up_ratio)([x, inputs])

        # Create whole model
        super().__init__(inputs, outputs, name=name, **kwargs)

    def get_config(self):
        # This allows model.save() to work
        return {
            "filters": self.filters,
            "blocks": self.blocks,
            "up_ratio": self.up_ratio,
        }
