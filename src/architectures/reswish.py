import tensorflow as tf
import keras
from keras import layers


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
        self.conv = layers.Conv2D(filters, (3, 3), padding="same", use_bias=False)
        # Normalize
        self.bn1 = layers.BatchNormalization()
        # Swish it
        self.sw1 = layers.Activation("swish")

        # Weight
        self.dnn1 = layers.Conv2D(filters, (1, 1), padding="valid", use_bias=False)
        # Normalize
        self.bn2 = layers.BatchNormalization()
        # Swish it
        self.sw2 = layers.Activation("swish")

        # Repeat
        self.dnn2 = layers.Conv2D(filters, (1, 1), padding="valid", use_bias=False)
        self.bn3 = layers.BatchNormalization()
        self.sw3 = layers.Activation("swish")

    def call(self, inputs):
        res = inputs

        x = self.conv(inputs)
        x = self.bn1(x)
        x = self.sw1(x)

        x = self.dnn1(x)
        x = self.bn2(x)
        x = self.sw2(x)

        x = self.dnn2(x)
        x = self.bn3(x)
        x = self.sw3(x)

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
        self.sw = layers.Activation("swish")

    def call(self, inputs):
        x = self.conv(inputs)
        x = self.sw(x)
        return tf.nn.depth_to_space(x, block_size=self.shuffle_num)

    def get_config(self):
        config = super().get_config()
        config.update({"filters": self.filters, "shuffle_num": self.shuffle_num})
        return config


@keras.saving.register_keras_serializable()
class Reswish(keras.Model):
    """Defines a model using ReswishLayer architecture."""

    def __init__(
        self,
        filters: int = 32,
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

        # Add res back
        x = layers.add([x, res])

        # Upscale
        x = PixelShuffleUpscale(filters=filters, shuffle_num=up_ratio)(x)

        # Collapse to RGB
        outputs = layers.Conv2D(3, (3, 3), padding="same", activation="sigmoid")(x)

        # Create whole model
        super().__init__(inputs, outputs, name=name, **kwargs)

    def get_config(self):
        # This allows model.save() to work
        return {
            "filters": self.filters,
            "blocks": self.blocks,
            "up_ratio": self.up_ratio,
        }
