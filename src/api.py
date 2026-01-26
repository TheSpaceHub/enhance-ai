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
logger = logging.getLogger("SR-LAB-API")

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
MODEL_FILES = {
    "CNNU": {
        2: "../models/cnnu_x2.keras", # future x2 model
        4: "../models/cnnu_e30_sc4.keras",
    },
    "ESPCN": {
        2: "../models/espcn_x2.keras",
        4: "../models/espcn_e30_sc4.keras",
    },
    "SRGAN": {
        4: "../models/srgan_e30_sc4_rb8f64_l005.keras",
    },
    "SRResNet": {
        4: "../models/srrn_e30_sc4_rb8f64.keras",
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
        # Temporary fallback to x4 if requested scale is unavailable
        if 4 in MODEL_FILES[model_name]:
            logger.warning(
                f"Requested x{scale} not found for {model_name}, falling back to x4."
            )
            scale_key = 4
        else:
            raise HTTPException(
                status_code=404,
                detail=f"Model '{model_name}' x{scale} is not available.",
            )
    else:
        scale_key = scale

    cache_key = f"{model_name}_x{scale_key}"

    if cache_key not in loaded_models:
        model_path = MODEL_FILES[model_name][scale_key]
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
        scale_factor = int(float(scale))

        # Select execution device based on availability and user request
        device_type = (
            "/GPU:0"
            if device.upper() == "GPU" and len(gpus) > 0
            else "/CPU:0"
        )

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

        # Model loading
        model = get_model(model_name, scale_factor)

        # Inference with runtime measurement
        with tf.device(device_type):
            start_time = time.perf_counter()
            input_tensor = tf.convert_to_tensor(input_tensor)

            prediction = model(input_tensor, training=False)

            _ = prediction.shape  # Forces execution
            runtime = time.perf_counter() - start_time

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

    except Exception as e:
        logger.error(f"Upscale error: {e}")
        return {
            "status": "error",
            "message": str(e),
        }

# Development entry point
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
