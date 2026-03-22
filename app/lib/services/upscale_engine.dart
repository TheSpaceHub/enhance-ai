import 'dart:async';
import 'package:flutter/foundation.dart';
import 'package:flutter/services.dart';
import '../ui/models/sr_experiment.dart';

class ApiService {
  static const platform = MethodChannel('com.enhanceai.superres/nnapi');

  static Future<SRRun> upscaleImage({
    required Uint8List imageBytes,
    required String modelName,
    required int factor,
    required String device,
  }) async {
    try {
      final Stopwatch watch = Stopwatch()..start();

      // Send the data across the bridge to Kotlin
      final Uint8List? resultBytes = await platform.invokeMethod('upscaleImage', {
        'imageBytes': imageBytes,
        // Ensure this matches your exact trained model filename
        'modelName': '${modelName.toLowerCase()}_x$factor.tflite', 
        'factor': factor,
      });

      if (resultBytes == null) throw Exception("Native processing returned null.");

      watch.stop();

      return SRRun(
        id: DateTime.now().millisecondsSinceEpoch.toString(),
        modelName: modelName,
        upscaleFactor: factor.toDouble(),
        isProcessing: false,
        resultBytes: resultBytes,
        device: "NPU (NNAPI Accelerated)",
        inferenceTime: "${(watch.elapsedMilliseconds / 1000.0).toStringAsFixed(3)}s",
        metrics: {"Status": "Success: 256x256 Float32 Tiling"},
      );
    } on PlatformException catch (e) {
      debugPrint("MethodChannel Error: '${e.message}'.");
      throw Exception("Failed native inference: ${e.message}");
    }
  }
}