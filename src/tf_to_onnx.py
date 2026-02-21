import tensorflow as tf
import tf2onnx
import onnx
import os

from architectures import Average, CNNUpscaler, ESPCN, SRRN, SRGAN

# Config

MODEL_PATH = "../models/"
OUTPUT_DIR = "../models/onnx_models"

MODEL_FILES = {
    "Average": {
        2: MODEL_PATH + "average_x2.weights.h5",
        4: MODEL_PATH + "average_x4.weights.h5",
    },
    "CNNU": {
        2: MODEL_PATH + "cnnu_e100_x2.weights.h5",
        4: MODEL_PATH + "cnnu_e100_x4.weights.h5",
    },
    "ESPCN": {
        2: MODEL_PATH + "espcn_e100_x2.weights.h5",
        4: MODEL_PATH + "espcn_e100_x4.weights.h5",
    },
    "SRRN": {
        2: MODEL_PATH + "srrn_e100_b16f64_x2.weights.h5",
        4: MODEL_PATH + "srrn_e100_b16f64_x4.weights.h5",
    },
    "SRGAN": {
        2: MODEL_PATH + "srgan_e100_b8f64_l005_x2.h5",  # still WIP
        4: MODEL_PATH + "srgan_e100_b8f64_l005_x4.h5",  # still WIP
    },
}

# Custom objects mapping for Keras models
CUSTOM_OBJECTS = {
    "Average": Average,
    "CNNUpscaler": CNNUpscaler,
    "ESPCN": ESPCN,
    "SRGAN": SRGAN,
    "SRRN": SRRN,
}

PARAMETERS = {
    "Average": {2: {}, 4: {}},
    "CNNU": {2: {}, 4: {}},
    "ESPCN": {2: {}, 4: {}},
    "SRRN": {
        2: {"num_blocks": 16, "filters": 64},
        4: {"num_blocks": 16, "filters": 64},
    },
    "SRGAN": {
        2: {"num_blocks": 8, "filters": 64},
        4: {"num_blocks": 8, "filters": 64},
    },
}


def build_model(model: tf.keras.Model):
    """Runs a dummy forward pass to initialize model weights and shapes."""
    model(tf.zeros([1, 64, 64, 3], tf.float32))


def load_model(
    path: str, model_class, scale: int, params: dict[str, int]
) -> tf.keras.Model:
    """Loads a Keras model or weights file into a complete model instance.

    Args:
        path (str): Path to the model or weights.
        model_class (class, optional): Required if loading only weights.
        scale (int, optional): Upscaling factor for the model.
        params (dict[str, int]): Parameters needed to initialize the model.

    Returns:
        tf.keras.Model: Loaded and initialized model.
    """
    if model_class is None:
        raise ValueError("model_class must be provided when loading weights")
    model = model_class(up_ratio=scale, **params) if scale else model_class()
    build_model(model)
    print(model.summary())
    model.load_weights(path)
    return model


def convert_to_onnx(model: tf.keras.Model, output_path: str):
    """Converts a Keras model to ONNX format and saves it to disk.

    Args:
        model (tf.keras.Model): Model to convert.
        output_path (str): Destination file path for ONNX model.
    """
    spec = (tf.TensorSpec((None, None, None, 3), tf.float32, name="input"),)
    onnx_model_proto, _ = tf2onnx.convert.from_keras(
        model, input_signature=spec, opset=13
    )

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    onnx.save_model(onnx_model_proto, output_path)
    print(f"✅ Saved ONNX model: {output_path}")


def convert_all_models_to_onnx():
    """Iterates through all configured models and scales, converting them to ONNX."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for name, scales in MODEL_FILES.items():
        for scale, path in scales.items():
            tf.keras.backend.clear_session()
            print(f"🔄 Converting {name} x{scale} to ONNX...")
            try:
                # Load the appropriate model class
                if name == "Average":
                    model = load_model(path, Average, scale, PARAMETERS[name][scale])
                elif name == "CNNU":
                    model = load_model(
                        path, CNNUpscaler, scale, PARAMETERS[name][scale]
                    )
                elif name == "ESPCN":
                    model = load_model(path, ESPCN, scale, PARAMETERS[name][scale])
                elif name == "SRRN":
                    model = load_model(path, SRRN, scale, PARAMETERS[name][scale])
                elif name == "SRGAN":
                    model = load_model(path, SRGAN, scale, PARAMETERS[name][scale])
                else:
                    raise ValueError(f"Unknown model: {name}")

                # Output ONNX path
                onnx_out = os.path.join(OUTPUT_DIR, f"{name}_x{scale}.onnx").lower()
                convert_to_onnx(model, onnx_out)

            except Exception as e:
                print(f"❌ Error converting {name} x{scale}")
                print(e)
                print("-" * 60)


if __name__ == "__main__":
    convert_all_models_to_onnx()
