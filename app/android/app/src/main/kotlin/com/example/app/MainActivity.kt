package com.example.app

import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Color
import io.flutter.embedding.android.FlutterActivity
import io.flutter.embedding.engine.FlutterEngine
import io.flutter.plugin.common.MethodChannel
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.nnapi.NnApiDelegate
import java.io.ByteArrayOutputStream
import java.io.File
import java.io.FileInputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.channels.FileChannel
import kotlin.concurrent.thread
import kotlin.math.roundToInt

class MainActivity: FlutterActivity() {
    private val CHANNEL = "com.enhanceai.superres/nnapi"
    
    // We keep the interpreter globally so it stays warm and cached in RAM
    private var tfliteInterpreter: Interpreter? = null

    override fun configureFlutterEngine(flutterEngine: FlutterEngine) {
        super.configureFlutterEngine(flutterEngine)

        MethodChannel(flutterEngine.dartExecutor.binaryMessenger, CHANNEL).setMethodCallHandler { call, result ->
            if (call.method == "upscaleImage") {
                val imageBytes = call.argument<ByteArray>("imageBytes")
                val modelName = call.argument<String>("modelName")
                val factor = call.argument<Int>("factor")

                if (imageBytes == null || modelName == null || factor == null) {
                    result.error("INVALID_ARGS", "Missing required arguments.", null)
                    return@setMethodCallHandler
                }

                // Push heavy math to a background thread so the Flutter UI doesn't freeze
                thread {
                    try {
                        val outputBytes = runNativeInference(imageBytes, modelName, factor)
                        
                        // Pass the result back to the main UI thread
                        runOnUiThread { result.success(outputBytes) }
                    } catch (e: Exception) {
                        runOnUiThread { result.error("NATIVE_ERROR", e.message, null) }
                    }
                }
            } else {
                result.notImplemented()
            }
        }
    }

    private fun runNativeInference(imageBytes: ByteArray, modelName: String, factor: Int): ByteArray {
        // 1. Load Interpreter & trigger NNAPI Caching (Only happens once)
        if (tfliteInterpreter == null) {
            val cacheDir = File(context.cacheDir, "nnapi_cache")
            if (!cacheDir.exists()) cacheDir.mkdirs()

            val nnApiOptions = NnApiDelegate.Options().apply {
                setExecutionPreference(NnApiDelegate.Options.EXECUTION_PREFERENCE_SUSTAINED_SPEED)
                setCacheDir(cacheDir.absolutePath)
                setModelToken("${modelName}_v1") // Cache ID
            }

            val tfliteOptions = Interpreter.Options().apply {
                addDelegate(NnApiDelegate(nnApiOptions))
                setNumThreads(4) // Backup if NPU fails
            }

            val modelBuffer = loadModelFromAssets(modelName)
            tfliteInterpreter = Interpreter(modelBuffer, tfliteOptions)
        }

        val interpreter = tfliteInterpreter!!

        // 2. Extract Quantization Parameters
        val inputQuant = interpreter.getInputTensor(0).quantizationParams()
        val outputQuant = interpreter.getOutputTensor(0).quantizationParams()
        
        val inScale = inputQuant.scale
        val inZero = inputQuant.zeroPoint
        val outScale = outputQuant.scale
        val outZero = outputQuant.zeroPoint

        // 3. Decode Image & Setup Geometry
        val inputBitmap = BitmapFactory.decodeByteArray(imageBytes, 0, imageBytes.size)
        val inW = inputBitmap.width
        val inH = inputBitmap.height
        val outW = inW * factor
        val outH = inH * factor

        val outputBitmap = Bitmap.createBitmap(outW, outH, Bitmap.Config.ARGB_8888)

        // 4. Pre-allocate flat ByteBuffers (The bullet-train tracks)
        val chunkSize = 64
        val outChunkSize = chunkSize * factor
        
        // 64 * 64 * 3 channels (1 byte per pixel for INT8)
        val inputBuffer = ByteBuffer.allocateDirect(chunkSize * chunkSize * 3)
        inputBuffer.order(ByteOrder.nativeOrder())
        
        val outputBuffer = ByteBuffer.allocateDirect(outChunkSize * outChunkSize * 3)
        outputBuffer.order(ByteOrder.nativeOrder())

        // 5. The Chunking Loop
        for (startY in 0 until inH step chunkSize) {
            for (startX in 0 until inW step chunkSize) {
                
                inputBuffer.rewind()
                outputBuffer.rewind()

                // A. Populate input buffer (Quantize)
                for (y in 0 until chunkSize) {
                    for (x in 0 until chunkSize) {
                        val imgY = startY + y
                        val imgX = startX + x

                        if (imgY < inH && imgX < inW) {
                            val pixel = inputBitmap.getPixel(imgX, imgY)
                            
                            // Convert [0-255] color to float [0.0-1.0], then to INT8
                            val r = (((Color.red(pixel) / 255f) / inScale) + inZero).roundToInt().coerceIn(-128, 127).toByte()
                            val g = (((Color.green(pixel) / 255f) / inScale) + inZero).roundToInt().coerceIn(-128, 127).toByte()
                            val b = (((Color.blue(pixel) / 255f) / inScale) + inZero).roundToInt().coerceIn(-128, 127).toByte()
                            
                            inputBuffer.put(r); inputBuffer.put(g); inputBuffer.put(b)
                        } else {
                            // Out of bounds padding
                            inputBuffer.put(inZero.toByte()); inputBuffer.put(inZero.toByte()); inputBuffer.put(inZero.toByte())
                        }
                    }
                }

                // B. Execute the model on the NPU
                interpreter.run(inputBuffer, outputBuffer)

                // C. Read output buffer (Dequantize & Stitch)
                outputBuffer.rewind()
                for (y in 0 until outChunkSize) {
                    for (x in 0 until outChunkSize) {
                        val outImgY = (startY * factor) + y
                        val outImgX = (startX * factor) + x

                        val rInt = outputBuffer.get().toInt()
                        val gInt = outputBuffer.get().toInt()
                        val bInt = outputBuffer.get().toInt()

                        if (outImgY < outH && outImgX < outW) {
                            val r = ((rInt - outZero) * outScale * 255).roundToInt().coerceIn(0, 255)
                            val g = ((gInt - outZero) * outScale * 255).roundToInt().coerceIn(0, 255)
                            val b = ((bInt - outZero) * outScale * 255).roundToInt().coerceIn(0, 255)

                            outputBitmap.setPixel(outImgX, outImgY, Color.rgb(r, g, b))
                        }
                    }
                }
            }
        }

        // 6. Compress final image back to bytes for Flutter
        val stream = ByteArrayOutputStream()
        outputBitmap.compress(Bitmap.CompressFormat.PNG, 100, stream)
        return stream.toByteArray()
    }

    private fun loadModelFromAssets(modelName: String): ByteBuffer {
        // MODERN FLUTTER WAY: Use FlutterInjector to find the asset
        val flutterLoader = io.flutter.FlutterInjector.instance().flutterLoader()
        val flutterAssetPath = flutterLoader.getLookupKeyForAsset("assets/models/$modelName")

        val assetManager = context.assets
        val fd = assetManager.openFd(flutterAssetPath)
        val inputStream = FileInputStream(fd.fileDescriptor)
        val fileChannel = inputStream.channel
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, fd.startOffset, fd.declaredLength)
    }
}