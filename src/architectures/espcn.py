import keras
import tensorflow as tf
from keras import layers

@keras.saving.register_keras_serializable()
class ESPCN(keras.Model):
    """
    Efficient Sub-Pixel Convolutional Neural Network for image upscaling.
    Works in LR space, uses pixel shuffle at the end for fast and stable super-resolution.
    """
    
    def __init__(self, up_ratio, name="espcn", **kwargs):
        super().__init__(name=name, **kwargs)
        self.up_ratio = up_ratio

        self.conv1= layers.Conv2D(64, 3, padding='same', activation='relu')
        self.conv2 = layers.Conv2D(64, 3, padding='same', activation='relu')
        self.conv3 = layers.Conv2D(32, 3, padding='same', activation='relu')
        self.conv4 = layers.Conv2D(3 * (self.up_ratio ** 2), 3, padding='same')
        
        self.pixel_shuffle = layers.Lambda(lambda x: tf.nn.depth_to_space(x, block_size=self.up_ratio))

    def call(self, inputs):
        # Calculate the corrections
        x = self.conv1(inputs)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        
        # Upscale by pixel_suffle
        x = self.pixel_shuffle(x)

        return tf.clip_by_value(x, 0.0, 1.0)

    def get_config(self):
        config = super().get_config()
        config.update({"up_ratio": self.up_ratio})
        return config