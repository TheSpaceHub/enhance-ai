import 'dart:async';
import 'dart:io';
import 'package:flutter/foundation.dart';
import 'package:flutter/services.dart' show rootBundle;
import 'package:path_provider/path_provider.dart';
import 'package:onnxruntime/onnxruntime.dart';
import 'package:image/image.dart' as img;
import '../ui/models/sr_experiment.dart';

/// Core engine for running ONNX-based super-resolution experiments
class ApiService {
  /// Upscales the given image using the specified model and factor
  static Future<SRRun> upscaleImage({
    required Uint8List imageBytes,
    required String modelName,
    required int factor,
    required String device,
  }) async {
    try {
      final String assetName = '${modelName.toLowerCase()}_x$factor.onnx';
      final String modelPath = await _prepareModelFile(assetName);

      // Run inference in a separate isolate
      return await compute(_runInferenceOnnx, {
        'modelPath': modelPath,
        'imageBytes': imageBytes,
        'modelName': modelName,
        'factor': factor,
        'device': device,
      });
    } catch (e) {
      debugPrint("Error in ONNX Pipeline: $e");
      throw Exception("Failed to run ONNX inference: $e");
    }
  }

  /// Ensures the ONNX model exists on device and returns its absolute path
  static Future<String> _prepareModelFile(String filename) async {
    final directory = await getApplicationSupportDirectory();
    final String fullPath =
        "${directory.path}${Platform.pathSeparator}$filename";
    final file = File(fullPath);

    // Copy from assets if missing or empty
    if (!await file.exists() || await file.length() == 0) {
      debugPrint("Copying model to: $fullPath");
      final data = await rootBundle.load('assets/models/$filename');
      final bytes = data.buffer.asUint8List();
      await file.writeAsBytes(bytes, flush: true);
    }

    return file.path;
  }

  /// Executes ONNX inference on the given image
  static Future<SRRun> _runInferenceOnnx(Map<String, dynamic> params) async {
    OrtEnv.instance.init();

    final String modelPath = params['modelPath'];
    final Uint8List imageBytes = params['imageBytes'];
    final String modelName = params['modelName'];
    final int factor = params['factor'];
    final String device = params['device'];

    final sessionOptions = OrtSessionOptions();

    // Load model bytes to avoid platform path issues
    final Uint8List modelBytes = File(modelPath).readAsBytesSync();
    final session = OrtSession.fromBuffer(modelBytes, sessionOptions);

    try {
      final Stopwatch watch = Stopwatch()..start();

      // Decode input image and ensure 3 channels
      img.Image? image = img.decodeImage(imageBytes);
      if (image == null) throw Exception("Invalid image");
      if (image.numChannels != 3) image = image.convert(numChannels: 3);

      final int inH = image.height;
      final int inW = image.width;

      // Preprocess: NHWC format, normalize [0,1]
      final inputFloats = Float32List(inH * inW * 3);
      int idx = 0;
      for (var pixel in image) {
        inputFloats[idx++] = pixel.r / 255.0;
        inputFloats[idx++] = pixel.g / 255.0;
        inputFloats[idx++] = pixel.b / 255.0;
      }

      final String inputKey = session.inputNames.first;
      final inputOrt = OrtValueTensor.createTensorWithDataList(inputFloats, [
        1,
        inH,
        inW,
        3,
      ]);

      final runOptions = OrtRunOptions();
      final Map<String, OrtValue> inputs = {inputKey: inputOrt};
      final List<OrtValue?> outputs = session.run(runOptions, inputs);

      // Postprocess: convert output tensor to image
      final List<double> outputData = _flattenTensor(outputs[0]?.value);
      final int outH = inH * factor;
      final int outW = inW * factor;
      final outImage = img.Image(width: outW, height: outH, numChannels: 3);

      int outIdx = 0;
      for (int y = 0; y < outH; y++) {
        for (int x = 0; x < outW; x++) {
          if (outIdx + 2 >= outputData.length) break;
          outImage.setPixelRgb(
            x,
            y,
            (outputData[outIdx++] * 255).toInt().clamp(0, 255),
            (outputData[outIdx++] * 255).toInt().clamp(0, 255),
            (outputData[outIdx++] * 255).toInt().clamp(0, 255),
          );
        }
      }

      final resultBytes = Uint8List.fromList(img.encodePng(outImage));
      watch.stop();

      // Release resources
      inputOrt.release();
      runOptions.release();
      for (var out in outputs) {
        out?.release();
      }

      return SRRun(
        id: DateTime.now().millisecondsSinceEpoch.toString(),
        modelName: modelName,
        upscaleFactor: factor.toDouble(),
        isProcessing: false,
        resultBytes: resultBytes,
        device: device,
        inferenceTime:
            "${(watch.elapsedMilliseconds / 1000.0).toStringAsFixed(3)}s",
        metrics: {"Resolution": "${inW}x$inH → ${outW}x$outH"},
      );
    } finally {
      session.release();
      sessionOptions.release();
      OrtEnv.instance.release();
    }
  }

  /// Recursively flattens nested tensors into a 1D list of doubles
  static List<double> _flattenTensor(dynamic tensor) {
    if (tensor is List) {
      return tensor.expand((e) => _flattenTensor(e)).toList();
    } else if (tensor is double) {
      return [tensor];
    } else if (tensor is num) {
      return [tensor.toDouble()];
    }
    return [];
  }
}
