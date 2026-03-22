import os
import tensorflow as tf
import hpc_archs  # Import architectures to register custom layers/models

def convert_to_tflite(keras_model_path, tflite_model_path):
    """
    Converts a Keras model to a TFLite model.
    """
    print(f"Loading Keras model from {keras_model_path}...")
    # Loading the model. hpc_archs components are automatically registered
    # via the @keras.saving.register_keras_serializable() decorators.
    model = tf.keras.models.load_model(keras_model_path)
    model.build((None,None,None,3))
    
    print("Converting to TFLite...")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    
    # You can uncomment the following line to enable default quantization/optimizations:
    # converter.optimizations = [tf.lite.Optimize.DEFAULT]
    
    tflite_model = converter.convert()
    
    with open(tflite_model_path, 'wb') as f:
        f.write(tflite_model)
        
    print(f"TFLite model saved to {tflite_model_path}")

if __name__ == "__main__":
    # List the names of the models you want to convert
    model_names = [
        "CNNU_x2",
        "CNNU_x4",
        "ESPCN_x2",
        "ESPCN_x4",
        "SRRN_x2",
        "SRRN_x4",
        "SRGAN_x2",
        "SRGAN_x4",
    ]

    # Create the output directory if it doesn't exist
    os.makedirs("models/tflite_models", exist_ok=True)

    for name in model_names:
        keras_path = f"bscmodels/{name}.keras"
        tflite_path = f"models/tflite/{name}.tflite"

        if os.path.exists(keras_path):
            convert_to_tflite(keras_path, tflite_path)
            print("-" * 50)
        else:
            print(f"Model file {keras_path} not found. Skipping...")
            print("-" * 50)
