import os

os.environ["TF_USE_LEGACY_KERAS"] = "1"  # work with quantization
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow_model_optimization as tfmot


def pixel_shuffle_fn(tensor, block_size):
    return tf.nn.depth_to_space(tensor, block_size=block_size)


def build_cnn_upscaler(up_ratio: float, input_shape=(64, 64, 3)):
    """Functional CNNUpscaler"""
    inputs = keras.Input(shape=input_shape, name="input_lr")

    x_up = layers.UpSampling2D(
        size=(int(up_ratio), int(up_ratio)), interpolation="bilinear", name="upscaler"
    )(inputs)

    x = layers.Conv2D(64, (3, 3), activation="relu", padding="same")(x_up)
    x = layers.Conv2D(32, (3, 3), activation="relu", padding="same")(x)
    correction = layers.Conv2D(3, (3, 3), padding="same")(x)

    outputs = layers.Add(name="add_correction")([x_up, correction])

    return keras.Model(inputs=inputs, outputs=outputs, name="CNNUpscaler")


def build_espcn(up_ratio: int, input_shape=(64, 64, 3)):
    """Functional ESPCN"""
    inputs = keras.Input(shape=input_shape, name="input_lr")

    x = layers.Conv2D(64, 3, padding="same", activation="relu")(inputs)
    x = layers.Conv2D(64, 3, padding="same", activation="relu")(x)
    x = layers.Conv2D(32, 3, padding="same", activation="relu")(x)
    x = layers.Conv2D(3 * (up_ratio**2), 3, padding="same")(x)

    x = layers.Lambda(
        pixel_shuffle_fn, arguments={"block_size": up_ratio}, name="pixel_shuffle"
    )(x)

    outputs = layers.ReLU(max_value=1.0)(x)

    return keras.Model(inputs=inputs, outputs=outputs, name="ESPCN")


def residual_block(inputs, filters):
    """Helper function to build a residual block"""
    x = layers.Conv2D(filters, 3, padding="same")(inputs)
    x = layers.ReLU()(x)
    x = layers.Conv2D(filters, 3, padding="same")(x)
    return layers.Add()([inputs, x])


def build_srrn(up_ratio: int, filters=64, num_blocks=8, input_shape=(64, 64, 3)):
    """Functional SRRN"""
    inputs = keras.Input(shape=input_shape, name="input_lr")

    x = layers.Conv2D(filters, 9, padding="same")(inputs)
    x = layers.ReLU()(x)

    res = x
    for _ in range(num_blocks):
        x = residual_block(x, filters)

    x = layers.Conv2D(filters, 3, padding="same")(x)
    x = layers.Add()([res, x])

    x = layers.Conv2D(filters * (up_ratio**2), 3, padding="same")(x)

    x = layers.Lambda(
        pixel_shuffle_fn, arguments={"block_size": up_ratio}, name="pixel_shuffle"
    )(x)

    outputs = layers.Conv2D(3, 9, padding="same", activation="sigmoid")(x)

    return keras.Model(inputs=inputs, outputs=outputs, name="SRRN")


def disc_block(inputs, filters, stride=1):
    """Helper function for Discriminator block"""
    x = layers.Conv2D(filters, 3, strides=stride, padding="same")(inputs)
    x = layers.BatchNormalization()(x)
    return layers.LeakyReLU(0.2)(x)


def build_discriminator(input_shape=(256, 256, 3)):
    """Functional Discriminator"""
    inputs = keras.Input(shape=input_shape, name="input_hr")
    x = inputs

    block_configs = [
        (64, 1),
        (64, 2),
        (128, 1),
        (128, 2),
        (256, 1),
        (256, 2),
        (512, 1),
        (512, 2),
    ]
    for filters, stride in block_configs:
        x = disc_block(x, filters, stride)

    outputs = layers.Conv2D(1, 3, padding="same")(x)
    return keras.Model(inputs=inputs, outputs=outputs, name="discriminator")


@keras.saving.register_keras_serializable()
class QAT_SRGAN(keras.Model):
    """We define the model itself"""

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
        self.lambda_adv = lambda_adv
        self.filters = filters
        self.num_blocks = num_blocks

        self.generator = build_srrn(
            up_ratio=up_ratio, filters=filters, num_blocks=num_blocks
        )

        # apply qat
        self.generator = tfmot.quantization.keras.quantize_model(self.generator)

        self.discriminator = build_discriminator()

        self.pixel_loss_fn = tf.keras.losses.MeanAbsoluteError()

        # Optimizers (they are assigned in the compile fase)
        self.gen_optimizer = None
        self.disc_optimizer = None

    def discriminator_loss(self, real_logits, fake_logits):
        # Uses Least Squares instead of Binary Cross-Entropy
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
            # Generate SR image
            sr = self.generator(lr, training=True)

            # Discriminator predictions
            real_logits = self.discriminator(hr, training=True)
            fake_logits = self.discriminator(sr, training=True)

            # Losses
            disc_loss = self.discriminator_loss(real_logits, fake_logits)
            gen_loss = self.generator_total_loss(sr, hr, fake_logits)

        # Gradients
        gen_grads = gen_tape.gradient(gen_loss, self.generator.trainable_variables)
        disc_grads = disc_tape.gradient(
            disc_loss, self.discriminator.trainable_variables
        )

        # Apply gradients
        self.gen_optimizer.apply_gradients(
            zip(gen_grads, self.generator.trainable_variables)
        )
        self.disc_optimizer.apply_gradients(
            zip(disc_grads, self.discriminator.trainable_variables)
        )

        # PSNR y SSIM metri
        psnr_val = tf.image.psnr(hr, sr, max_val=1.0)
        ssim_val = tf.image.ssim(hr, sr, max_val=1.0)

        return {
            "gen_loss": gen_loss,
            "disc_loss": disc_loss,
            "psnr": tf.reduce_mean(psnr_val),
            "ssim": tf.reduce_mean(ssim_val),
        }
