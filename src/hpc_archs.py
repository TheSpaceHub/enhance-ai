import numpy as np
import keras
import tensorflow as tf
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

        new_height = tf.cast(height, tf.float32) * self.up_ratio
        new_width = tf.cast(width, tf.float32) * self.up_ratio
        
        new_height = tf.cast(new_height, tf.int32)
        new_width = tf.cast(new_width, tf.int32)

        return tf.image.resize(inputs, [new_height, new_width], method='bilinear')

    def get_config(self):
        config = super().get_config()
        config.update({"up_ratio": self.up_ratio})
        return config

@keras.saving.register_keras_serializable()
class Average(keras.Model):
    """Defines a model which upscales images by averaging."""

    def __init__(self, up_ratio, name="average", **kwargs):
        super().__init__(name=name, **kwargs)
        self.up_ratio = up_ratio
        self.upscaler = Upscaler(up_ratio)

    def call(self, inputs):
        return self.upscaler(inputs)

    def get_config(self):
        config = super().get_config()
        config.update({"up_ratio": self.up_ratio})
        return config

@keras.saving.register_keras_serializable()
class CNNUpscaler(keras.Model):
    """Convolutional Neural Network for image upscaling."""

    def __init__(self, up_ratio: float, name="cnnupscaler", **kwargs):
        super().__init__(name=name, **kwargs)
        self.up_ratio = up_ratio
        
        self.upscaler = Upscaler(up_ratio)
        
        self.conv1 = layers.Conv2D(64, (3, 3), activation="relu", padding="same")
        self.conv2 = layers.Conv2D(32, (3, 3), activation="relu", padding="same")
        self.conv3 = layers.Conv2D(3, (3, 3), padding="same")

    def call(self, inputs):
        x_up = self.upscaler(inputs)
        x = self.conv1(x_up)
        x = self.conv2(x)
        correction = self.conv3(x)
        return x_up + correction

    def get_config(self):
        config = super().get_config()
        config.update({"up_ratio": self.up_ratio})
        return config

@keras.saving.register_keras_serializable()
class ESPCN(keras.Model):
    """Efficient Sub-Pixel Convolutional Neural Network."""
    
    def __init__(self, up_ratio, name="espcn", **kwargs):
        super().__init__(name=name, **kwargs)
        self.up_ratio = up_ratio

        self.conv1= layers.Conv2D(64, 3, padding='same', activation='relu')
        self.conv2 = layers.Conv2D(64, 3, padding='same', activation='relu')
        self.conv3 = layers.Conv2D(32, 3, padding='same', activation='relu')
        self.conv4 = layers.Conv2D(3 * (self.up_ratio ** 2), 3, padding='same')
        
        self.pixel_shuffle = layers.Lambda(lambda x: tf.nn.depth_to_space(x, block_size=self.up_ratio))

    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.pixel_shuffle(x)
        return tf.clip_by_value(x, 0.0, 1.0)

    def get_config(self):
        config = super().get_config()
        config.update({"up_ratio": self.up_ratio})
        return config

@keras.saving.register_keras_serializable()
class ResidualBlock(layers.Layer):
    def __init__(self, filters, **kwargs):
        super().__init__(**kwargs)
        self.filters = filters
        self.conv1 = layers.Conv2D(filters, 3, padding="same")
        self.relu = layers.ReLU()
        self.conv2 = layers.Conv2D(filters, 3, padding="same")

    def call(self, x, training=False):
        residual = x
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        return x + residual

    def get_config(self):
        config = super().get_config()
        config.update({"filters": self.filters})
        return config

@keras.saving.register_keras_serializable()
class SRRN(keras.Model):
    def __init__(self, up_ratio, filters=64, num_blocks=8, name="SRRN", **kwargs):
        super().__init__(name=name, **kwargs)
        self.up_ratio = up_ratio
        self.filters = filters
        self.num_blocks = num_blocks

        self.conv_in = layers.Conv2D(filters, 9, padding="same")
        self.relu = layers.ReLU()

        self.res_blocks = [ResidualBlock(filters) for _ in range(num_blocks)]

        self.conv_post_res = layers.Conv2D(filters, 3, padding="same")

        self.upsample = layers.Conv2D(filters * (up_ratio**2), 3, padding="same")
        self.pixel_shuffle = layers.Lambda(lambda x: tf.nn.depth_to_space(x, up_ratio))

        self.conv_final = layers.Conv2D(3, 9, padding="same", activation="sigmoid")

    def call(self, x, training=False):
        x = self.conv_in(x)
        x = self.relu(x)

        res = x
        for block in self.res_blocks:
            x = block(x, training=training)
        x = self.conv_post_res(x)
        x = layers.add([res, x])

        x = self.upsample(x)
        x = self.pixel_shuffle(x)

        return self.conv_final(x)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "up_ratio": self.up_ratio,
                "filters": self.filters,
                "num_blocks": self.num_blocks,
            }
        )
        return config

@keras.saving.register_keras_serializable()
class DiscBlock(layers.Layer):
    def __init__(self, filters, stride=1, name="DiscBlock"):
        super().__init__(name=name)
        self.filters = filters
        self.stride = stride
        self.conv = layers.Conv2D(filters, 3, strides=stride, padding="same")
        self.bn = layers.BatchNormalization()
        self.act = layers.LeakyReLU(0.2)

    def call(self, x, training=False):
        x = self.conv(x)
        x = self.bn(x, training=training)
        return self.act(x)

    def get_config(self):
        config = super().get_config()
        config.update({"filters": self.filters, "stride": self.stride})
        return config

@keras.saving.register_keras_serializable()
class Discriminator(keras.Model):
    def __init__(self, name="discriminator", **kwargs):
        super().__init__(name=name, **kwargs)

        self.blocks = [
            DiscBlock(64, stride=1),
            DiscBlock(64, stride=2),
            DiscBlock(128, stride=1),
            DiscBlock(128, stride=2),
            DiscBlock(256, stride=1),
            DiscBlock(256, stride=2),
            DiscBlock(512, stride=1),
            DiscBlock(512, stride=2),
        ]

        self.final_conv = layers.Conv2D(1, 3, padding="same")

    def call(self, inputs, training=False):
        x = inputs
        for block in self.blocks:
            x = block(x, training=training)
        x = self.final_conv(x)
        return x

    def get_config(self):
        config = super().get_config()
        return config

@keras.saving.register_keras_serializable()
class SRGAN(keras.Model):
    def __init__(
        self,
        up_ratio,
        filters=64,
        num_blocks=8,
        lambda_adv=1e-3,
        name="srgan",
        **kwargs
    ):
        super().__init__(name=name, **kwargs)
        self.up_ratio = up_ratio
        self.filters = filters
        self.num_blocks = num_blocks

        self.generator = SRRN(
            up_ratio=up_ratio,
            filters=filters,
            num_blocks=num_blocks,
        )

        self.discriminator = Discriminator()

        self.lambda_adv = lambda_adv

        self.pixel_loss_fn = tf.keras.losses.MeanAbsoluteError()

        self.gen_optimizer = None
        self.disc_optimizer = None

    def discriminator_loss(self, real_logits, fake_logits):
        real_loss = tf.reduce_mean((real_logits - 1.0) ** 2)
        fake_loss = tf.reduce_mean((fake_logits - 0.0) ** 2)
        return real_loss + fake_loss

    def _generator_adversarial_loss(self, fake_logits):
        return tf.reduce_mean((fake_logits - 1.0) ** 2)

    def generator_total_loss(self, sr, hr, fake_logits):
        pixel_loss = self.pixel_loss_fn(hr, sr)
        adv_loss = self._generator_adversarial_loss(fake_logits)
        return pixel_loss + self.lambda_adv * adv_loss

    def compile(self, gen_optimizer, disc_optimizer, **kwargs):
        super().compile(**kwargs)

        self.gen_optimizer = gen_optimizer
        self.disc_optimizer = disc_optimizer

    def call(self, inputs):
        return self.generator(inputs)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "up_ratio": self.up_ratio,
                "lambda_adv": self.lambda_adv,
                "filters": self.filters,
                "num_blocks": self.num_blocks,
            }
        )
        return config

    @tf.function
    def train_step(self, data):
        lr, hr = data

        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            sr = self.generator(lr, training=True)

            real_logits = self.discriminator(hr, training=True)
            fake_logits = self.discriminator(sr, training=True)

            disc_loss = self.discriminator_loss(real_logits, fake_logits)
            gen_loss = self.generator_total_loss(sr, hr, fake_logits)

        gen_grads = gen_tape.gradient(gen_loss, self.generator.trainable_variables)
        disc_grads = disc_tape.gradient(
            disc_loss, self.discriminator.trainable_variables
        )

        self.gen_optimizer.apply_gradients(
            zip(gen_grads, self.generator.trainable_variables)
        )
        self.disc_optimizer.apply_gradients(
            zip(disc_grads, self.discriminator.trainable_variables)
        )

        psnr_val = tf.image.psnr(hr, sr, max_val=1.0)
        ssim_val = tf.image.ssim(hr, sr, max_val=1.0)

        return {
            "gen_loss": gen_loss,
            "disc_loss": disc_loss,
            "psnr": tf.reduce_mean(psnr_val),
            "ssim": tf.reduce_mean(ssim_val),
        }
