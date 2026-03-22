package com.example.app // Change this to your actual package name!

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
    
    private var tfliteInterpreter: Interpreter? = null
    private var currentModelName: String? = null

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

                thread {
                    try {
                        val outputBytes = runNativeInference(imageBytes, modelName, factor)
                        runOnUiThread { result.success(outputBytes) }
                    } catch (e: Exception) {
                        runOnUiThread { result.error("NATIVE_ERROR", e.message ?: "Unknown error", null) }
                    }
                }
            } else {
                result.notImplemented()
            }
        }
    }

    private fun runNativeInference(imageBytes: ByteArray, modelName: String, factor: Int): ByteArray {
        
        if (tfliteInterpreter == null || currentModelName != modelName) {
            tfliteInterpreter?.close()
            
            val cacheDir = File(context.cacheDir, "nnapi_cache")
            if (!cacheDir.exists()) cacheDir.mkdirs()

            val nnApiOptions = NnApiDelegate.Options().apply {
                setExecutionPreference(NnApiDelegate.Options.EXECUTION_PREFERENCE_SUSTAINED_SPEED)
                setCacheDir(cacheDir.absolutePath)
                setModelToken("${modelName}_float32_v1")
            }

            val tfliteOptions = Interpreter.Options().apply {
                addDelegate(NnApiDelegate(nnApiOptions))
                setNumThreads(4)
            }

            val modelBuffer = loadModelFromAssets(modelName)
            val interpreter = Interpreter(modelBuffer, tfliteOptions)
            
            interpreter.resizeInput(0, intArrayOf(1, 256, 256, 3))
            interpreter.allocateTensors()
            
            tfliteInterpreter = interpreter
            currentModelName = modelName
        }

        val interpreter = tfliteInterpreter!!

        val inputBitmap = BitmapFactory.decodeByteArray(imageBytes, 0, imageBytes.size)
        val inW = inputBitmap.width
        val inH = inputBitmap.height
        val outW = inW * factor
        val outH = inH * factor

        val outputBitmap = Bitmap.createBitmap(outW, outH, Bitmap.Config.ARGB_8888)

        // TILING CONFIGURATION
        val chunkSize = 256
        val overlap = 4
        // We step by the core size (256 - 8 = 248) to ensure the 4px padding naturally overlaps
        val stepSize = chunkSize - (overlap * 2) 
        val outChunkSize = chunkSize * factor
        val outOverlap = overlap * factor
        
        val inputBuffer = ByteBuffer.allocateDirect(chunkSize * chunkSize * 3 * 4)
        inputBuffer.order(ByteOrder.nativeOrder())
        
        val outputBuffer = ByteBuffer.allocateDirect(outChunkSize * outChunkSize * 3 * 4)
        outputBuffer.order(ByteOrder.nativeOrder())

        for (startY in 0 until inH step stepSize) {
            for (startX in 0 until inW step stepSize) {
                
                inputBuffer.rewind()
                outputBuffer.rewind()

                // A. Populate the 256x256 input buffer
                for (y in 0 until chunkSize) {
                    for (x in 0 until chunkSize) {
                        val imgY = startY + y
                        val imgX = startX + x

                        if (imgY < inH && imgX < inW) {
                            val pixel = inputBitmap.getPixel(imgX, imgY)
                            inputBuffer.putFloat(Color.red(pixel) / 255f)
                            inputBuffer.putFloat(Color.green(pixel) / 255f)
                            inputBuffer.putFloat(Color.blue(pixel) / 255f)
                        } else {
                            inputBuffer.putFloat(0f)
                            inputBuffer.putFloat(0f)
                            inputBuffer.putFloat(0f)
                        }
                    }
                }

                // B. Run inference
                interpreter.run(inputBuffer, outputBuffer)

                // C. Calculate the safe "core" zone to avoid edge artifacts
                // If we are at the absolute edge of the image, keep the border. 
                // Otherwise, crop the overlap out.
                val writeStartY = if (startY == 0) 0 else outOverlap
                val writeStartX = if (startX == 0) 0 else outOverlap
                val writeEndY = if (startY + chunkSize >= inH) outChunkSize else outChunkSize - outOverlap
                val writeEndX = if (startX + chunkSize >= inW) outChunkSize else outChunkSize - outOverlap

                // D. Read buffer and stitch safe pixels
                outputBuffer.rewind()
                for (y in 0 until outChunkSize) {
                    for (x in 0 until outChunkSize) {
                        // We must always read the floats to keep the buffer advancing properly
                        val rFloat = outputBuffer.float
                        val gFloat = outputBuffer.float
                        val bFloat = outputBuffer.float

                        // Only paint the pixel if it falls inside the safe core zone
                        if (y in writeStartY until writeEndY && x in writeStartX until writeEndX) {
                            val outImgY = (startY * factor) + y
                            val outImgX = (startX * factor) + x

                            if (outImgY < outH && outImgX < outW) {
                                val r = (rFloat * 255f).roundToInt().coerceIn(0, 255)
                                val g = (gFloat * 255f).roundToInt().coerceIn(0, 255)
                                val b = (bFloat * 255f).roundToInt().coerceIn(0, 255)

                                outputBitmap.setPixel(outImgX, outImgY, Color.rgb(r, g, b))
                            }
                        }
                    }
                }
            }
        }

        val stream = ByteArrayOutputStream()
        outputBitmap.compress(Bitmap.CompressFormat.PNG, 100, stream)
        return stream.toByteArray()
    }

    private fun loadModelFromAssets(modelName: String): ByteBuffer {
        val flutterLoader = io.flutter.FlutterInjector.instance().flutterLoader()
        val flutterAssetPath = flutterLoader.getLookupKeyForAsset("assets/models/$modelName")

        val assetManager = context.assets
        val fd = assetManager.openFd(flutterAssetPath)
        val inputStream = FileInputStream(fd.fileDescriptor)
        val fileChannel = inputStream.channel
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, fd.startOffset, fd.declaredLength)
    }
}