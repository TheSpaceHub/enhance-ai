import tensorflow as tf
import keras
from keras import layers

# import the generator from other modules (or create a custom one)
from .srrn import SRRN

@keras.saving.register_keras_serializable()
class DiscBlock(layers.Layer):
    """Basic discriminator block: Conv -> BatchNorm -> LeakyReLu"""
    def __init__(self, filters, stride=1, name="DiscBlock"):
        super().__init__(name=name)
        self.conv = layers.Conv2D(filters, 3, strides=stride, padding="same")
        self.bn = layers.BatchNormalization()
        self.act = layers.LeakyReLU(0.2)

    def call(self, x, training=False):
        x = self.conv(x)
        x = self.bn(x, training=training)
        return self.act(x)

@keras.saving.register_keras_serializable()
class Discriminator(keras.Model):
    """Discriminador tipo PatchGAN para SRGAN"""
    def __init__(self, name="discriminator", **kwargs):
        super().__init__(name=name, **kwargs)
        
        # Discriminator blocks
        self.blocks = [
            DiscBlock(64, stride=1),
            DiscBlock(64, stride=2),

            DiscBlock(128, stride=1),
            DiscBlock(128, stride=2),

            DiscBlock(256, stride=1),
            DiscBlock(256, stride=2),

            DiscBlock(512, stride=1),
            DiscBlock(512, stride=2)
        ]

        # Final conv
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
    """Super-Resolution Generator Adversarial Network. Includes the generator + discriminator + custom train step"""
    def __init__(self, up_ratio, filters=64, num_blocks=8, lambda_adv=1e-3, name="srgan", **kwargs):
        super().__init__(name=name, **kwargs)
        self.up_ratio = up_ratio

        # Generator (SRResNet)
        self.generator = SRRN(up_ratio=up_ratio, filters=filters, num_blocks=num_blocks,)

        # Discriminator
        self.discriminator = Discriminator()

        # Loss weights
        self.lambda_adv = lambda_adv

        # Loss functions
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
        config.update({"up_ratio": self.up_ratio, "lambda_adv": self.lambda_adv})
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
        disc_grads = disc_tape.gradient(disc_loss, self.discriminator.trainable_variables)

        # Apply gradients
        self.gen_optimizer.apply_gradients(zip(gen_grads, self.generator.trainable_variables))
        self.disc_optimizer.apply_gradients(zip(disc_grads, self.discriminator.trainable_variables))

        # PSNR y SSIM metri 
        psnr_val = tf.image.psnr(hr, sr, max_val=1.0)
        ssim_val = tf.image.ssim(hr, sr, max_val=1.0)

        return {
            "gen_loss": gen_loss,
            "disc_loss": disc_loss,
            "psnr": tf.reduce_mean(psnr_val),
            "ssim": tf.reduce_mean(ssim_val)
            }