import tensorflow as tf
import architectures as archs
from architectures import *
from image_processing import load_and_preprocess
from pathlib import Path
import matplotlib.pyplot as plt
import onnxruntime as ort
import numpy as np


def try_h5():
    # Define constants
    IMAGE_PATH = "data/Set5/bird.png"
    MODEL_PATH = "models/average_x4.weights.h5"

    # Load model
    model = archs.Average(up_ratio=4)

    model(tf.zeros([1, 64, 64, 3], tf.float32))

    model.load_weights(MODEL_PATH)

    lr_img, _ = load_and_preprocess(IMAGE_PATH, up_ratio=4)
    result = model([lr_img])

    plt.imshow(lr_img)
    plt.show()
    plt.imshow(result[0])
    plt.show()


def try_onnx():
    # Define constants
    IMAGE_PATH = "data/Set5/bird.png"
    MODEL_PATH = "app/assets/models/average_x4.onnx"
    session = ort.InferenceSession(MODEL_PATH)

    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    img, _ = load_and_preprocess(IMAGE_PATH, up_ratio=4)
    
    img = np.array(img)
    
    plt.imshow(img)
    plt.show()

    img = img.astype(np.float32)
    input_data = np.expand_dims(img, axis=0)

    predictions = session.run([output_name], {input_name: input_data})

    output_tensor = predictions[0]
    output_image = np.squeeze(output_tensor, axis=0)
    output_image = np.clip(output_image, 0.0, 1.0)

    plt.imshow(output_image)
    plt.show()


def main():
    try_onnx()


if __name__ == "__main__":
    main()
