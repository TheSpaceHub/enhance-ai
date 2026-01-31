from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import tensorflow as tf
import numpy as np
import time
import io
import base64
import logging
from PIL import Image
from architectures import *

# Logging configuration for observability and debugging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("API")

# FastAPI application entry point
app = FastAPI(title="Enhance AI")

# CORS configuration to allow frontend communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# GPU detection and memory configuration
# Enables memory growth to avoid TensorFlow pre-allocating all VRAM
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logger.info(f"Detected {len(gpus)} GPU(s). Memory growth enabled.")
    except RuntimeError as e:
        logger.error(f"GPU configuration error: {e}")


# In-memory cache for loaded models to avoid repeated disk loads
loaded_models = {}


# Model registry: architecture name -> scale factor -> model file path
MODEL_PATH = "../models/"
MODEL_FILES = {
    "Average":{
        2: MODEL_PATH + "average_x2.keras",
        4: MODEL_PATH + "average_x4.keras"
    },
    "CNNU": {
        2: MODEL_PATH + "cnnu_e100_x2.keras",
        4: MODEL_PATH + "cnnu_e100_x4.keras",
    },
    "ESPCN": {
        2: MODEL_PATH + "espcn_e100_x2.keras",
        4: MODEL_PATH + "espcn_e100_x4.keras",
    },
    "SRGAN": {
        2: MODEL_PATH + "srgan_e100_b8f64_l005_x2.keras",
        4: MODEL_PATH + "srgan_e100_b8f64_l005_x4.keras",
    },
    "SRResNet": {
        2: MODEL_PATH + "srrn_e100_b8f64_x2.keras",
        4: MODEL_PATH + "srrn_e100_b8f64_x4.keras",
    },
}

# Model loader with caching and scale validation
def get_model(model_name: str, scale: int):
    """
    Loads and caches a TensorFlow super-resolution model
    for a given architecture and scale factor.
    """
    if model_name not in MODEL_FILES:
        raise HTTPException(
            status_code=404,
            detail=f"Architecture '{model_name}' is not configured.",
        )

    if scale not in MODEL_FILES[model_name]:
        raise HTTPException(
            status_code=404,
            detail=f"Model '{model_name}' x{scale} is not available.",
        )

    cache_key = f"{model_name}_x{scale}"

    if cache_key not in loaded_models:
        model_path = MODEL_FILES[model_name][scale]
        logger.info(f"Loading model {cache_key} from {model_path}")
        try:
            loaded_models[cache_key] = tf.keras.models.load_model(
                model_path,
                compile=False,
            )
        except Exception as e:
            logger.error(f"Failed to load model {model_path}: {e}")
            raise HTTPException(
                status_code=500,
                detail=f"Error loading model file: {e}",
            )

    return loaded_models[cache_key]

def predict(
    input_img: np.ndarray,
    model_name: str,
    up_ratio: int,
    device_type: str
) -> tuple[tf.Tensor, float]:
    """
    Receives a tensor image and upscales it using a model with an up_ratio.
    Returns the prediction tensor and runtime in seconds.
    """

    # Select model(s)
    if up_ratio == 8:
        models = [
            get_model(model_name, 2),
            get_model(model_name, 4),
        ]
    else:
        print(model_name, up_ratio)
        models = [get_model(model_name, up_ratio)]

    # Inference with runtime measurement
    with tf.device(device_type):
        start_time = time.perf_counter()
        prediction = tf.convert_to_tensor(input_img)

        for model in models:
            prediction = model(prediction, training=False)

        _ = prediction.shape  # Forces execution
        runtime = time.perf_counter() - start_time

    return prediction, runtime

# Image upscaling endpoint
@app.post("/upscale")
async def upscale(
    file: UploadFile = File(...),
    model_name: str = Form(...),
    scale: str = Form("4"),
    device: str = Form("GPU"),
):
    """
    Receives an image and returns an upscaled version generated
    by the selected model, scale factor, and execution device.
    """
    try:
        print("A")
        scale_factor = int(float(scale))

        # Select execution device based on availability and user request
        if device.upper() == "GPU" and len(gpus) < 1:
            raise HTTPException(
                status_code=400, 
                detail="GPU device is selected but no GPU is detected!"
                )
        
        device_type = ("/GPU:0" if device.upper() == "GPU" else "/CPU:0")
        
        print("B")
        logger.info(
            f"Request received: {model_name} x{scale_factor} | "
            f"Device: {device_type} | File: {file.filename}"
        )

        # Input preprocessing
        contents = await file.read()
        pil_img = Image.open(io.BytesIO(contents)).convert("RGB")
        in_w, in_h = pil_img.size

        img_array = np.array(pil_img).astype(np.float32) / 255.0
        input_tensor = np.expand_dims(img_array, axis=0)

        # Upscale image
        prediction, runtime = predict(input_tensor, model_name, scale_factor, device_type)

        # Post-processing
        output_tensor = tf.clip_by_value(tf.squeeze(prediction), 0.0, 1.0)
        output_array = (output_tensor.numpy() * 255).astype(np.uint8)

        out_pil = Image.fromarray(output_array)
        out_w, out_h = out_pil.size

        buffer = io.BytesIO()
        out_pil.save(buffer, format="PNG")
        img_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

        # Structured response for frontend visualization
        return {
            "status": "success",
            "image": img_base64,
            "inference_time": f"{runtime:.3f}s",
            "metrics": {
                "Input Res": f"{in_w}x{in_h}",
                "Output Res": f"{out_w}x{out_h}",
                "Scale": f"x{scale_factor}",
                "Device Used": device_type.replace("/", ""),
            },
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Upscale error: {e}")
        return {
            "status": "error",
            "message": str(e),
        }

# Development entry point
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
