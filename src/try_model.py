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
    MODEL_PATH = "models/cnnu10ep.weights.h5"

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
    MODEL_PATH = "models/onnx_models/cnnu_x4.onnx"
    session = ort.InferenceSession(MODEL_PATH)

    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    img, _ = load_and_preprocess(IMAGE_PATH, up_ratio=4)

    img = np.array(img)
    img = img[:64, :64, :]

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


def try_tflite():
    # Define constants
    IMAGE_PATH = "data/Set5/bird.png"
    MODEL_PATH = "bscmodels/cnnu.tflite"

    # 1. Initialize the TFLite Interpreter
    interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
    interpreter.allocate_tensors()

    # 2. Get input and output details
    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]

    # Extract the specific quantization scaling factors baked into your model
    input_scale, input_zero_point = input_details["quantization"]
    output_scale, output_zero_point = output_details["quantization"]

    # 3. Load the image
    lr_img, _ = load_and_preprocess(IMAGE_PATH, up_ratio=4)

    # Keras automatically handles batch sizes, but TFLite does not.
    # We must explicitly add the batch dimension so the shape becomes (1, H, W, 3)
    lr_img_batched = np.expand_dims(lr_img, axis=0)
    lr_img_batched = lr_img_batched[:, :64, :64, :]

    # 4. Quantize the Input (Float -> INT8)
    # The standard formula: integer_value = round(float_value / scale) + zero_point
    input_int8 = np.round((lr_img_batched / input_scale) + input_zero_point)
    input_int8 = np.clip(input_int8, -128, 127).astype(np.int8)

    # 5. Run Inference
    interpreter.set_tensor(input_details["index"], input_int8)
    interpreter.invoke()

    # 6. Extract the Output and Dequantize (INT8 -> Float)
    result_int8 = interpreter.get_tensor(output_details["index"])

    # The reverse formula: float_value = (integer_value - zero_point) * scale
    result_float = (result_int8.astype(np.float32) - output_zero_point) * output_scale

    # Remove the batch dimension for plotting: (1, H, W, 3) -> (H, W, 3)
    result_img = result_float[0]

    # Clip to [0, 1] (or [0, 255] depending on your preprocessing) so matplotlib doesn't complain
    result_img = np.clip(result_img, 0.0, 1.0)

    # 7. Plot the results
    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.title("Low Resolution Input")
    # If lr_img is a TensorFlow tensor, convert to numpy for plotting
    plt.imshow(np.array(lr_img))
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.title("TFLite INT8 Super Resolution Output")
    plt.imshow(result_img)
    plt.axis("off")

    plt.show()


def main():
    try_tflite()


if __name__ == "__main__":
    main()
