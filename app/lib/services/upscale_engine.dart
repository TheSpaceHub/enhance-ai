import 'dart:async';
import 'dart:io';
import 'dart:isolate';
import 'dart:ui' as ui;
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
    Function(double)? onProgress,
  }) async {
    try {
      if (factor == 8) {
        final stopwatch = Stopwatch()..start();
        
        // Pass 1: 4x Scale
        final run4x = await upscaleImage(
          imageBytes: imageBytes,
          modelName: modelName,
          factor: 4,
          device: device,
          onProgress: (p) => onProgress?.call(p * 0.5),
        );
        
        // Pass 2: 2x Scale mapped over Pass 1 Buffer
        final run8x = await upscaleImage(
          imageBytes: run4x.resultBytes!,
          modelName: modelName,
          factor: 2,
          device: device,
          onProgress: (p) => onProgress?.call(0.5 + (p * 0.5)),
        );
        
        stopwatch.stop();

        return SRRun(
          id: DateTime.now().millisecondsSinceEpoch.toString(),
          modelName: modelName,
          upscaleFactor: 8.0,
          isProcessing: false,
          resultBytes: run8x.resultBytes,
          device: device,
          inferenceTime: "${(stopwatch.elapsedMilliseconds / 1000.0).toStringAsFixed(3)}s",
          metrics: {
            "Resolution": "${run4x.metrics['Resolution']?.split(' → ')[0] ?? '?' } → ${run8x.metrics['Resolution']?.split(' → ')[1] ?? '?'}",
            "Accelerated (Tiles)": "ONNX Chained (4x + 2x)",
          },
        );
      }

      String safeModelName = modelName.toLowerCase();
      if (safeModelName == 'srresnet') {
        safeModelName = 'srrn';
      }
      
      final String assetName = '${safeModelName}_x$factor.onnx';
      final String modelPath = await _prepareModelFile(assetName);

      final Stopwatch decodeWatch = Stopwatch()..start();
      
      // Fast Native Decoding avoiding Dart array bottlenecks
      final codec = await ui.instantiateImageCodec(imageBytes);
      final frame = await codec.getNextFrame();
      final uiImage = frame.image;
      final int inW = uiImage.width;
      final int inH = uiImage.height;
      
      final ByteData? byteData = await uiImage.toByteData(format: ui.ImageByteFormat.rawRgba);
      if (byteData == null) throw Exception("Failed to decode ui.Image to raw bytes.");
      final Uint8List nativePixels = byteData.buffer.asUint8List();
      
      decodeWatch.stop();
      debugPrint("Native Decode Time: ${decodeWatch.elapsedMilliseconds}ms");

      final receivePort = ReceivePort();
      receivePort.listen((message) {
        if (message is double) {
          onProgress?.call(message);
        }
      });

      // Run inference in a separate isolate
      try {
        return await compute(_runInferenceOnnx, {
          'modelPath': modelPath,
          'imagePixels': nativePixels,
          'inW': inW,
          'inH': inH,
          'modelName': modelName,
          'factor': factor,
          'device': device,
          'sendPort': receivePort.sendPort,
        });
      } finally {
        receivePort.close();
      }
    } catch (e) {
      debugPrint("Error in ONNX Pipeline: $e");
      throw Exception("Failed to run ONNX inference: $e");
    }
  }

  /// Ensures the ONNX model exists on device and returns its absolute path
  static Future<String> _prepareModelFile(String filename) async {
    final directory = await getApplicationSupportDirectory();
    final String fullPath = "${directory.path}${Platform.pathSeparator}$filename";
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
    final Uint8List imagePixels = params['imagePixels'];
    final int inW = params['inW'];
    final int inH = params['inH'];
    final String modelName = params['modelName'];
    final int factor = params['factor'];
    final String device = params['device'];
    final SendPort? sendPort = params['sendPort'];

    final sessionOptions = OrtSessionOptions()
      ..setInterOpNumThreads(2)
      ..setIntraOpNumThreads(4)
      ..setSessionGraphOptimizationLevel(GraphOptimizationLevel.ortEnableAll);
      
    // Append ML Hardware Execution Providers
    if (Platform.isAndroid) {
        try {
            sessionOptions.appendNnapiProvider(NnapiFlags.useNone);
        } catch (_) {}
    } else if (Platform.isIOS || Platform.isMacOS) {
        try {
            sessionOptions.appendCoreMLProvider(CoreMLFlags.useNone);
        } catch (_) {}
    }
    
    // CPU Fallback Optimization (XNNPACK for ARM Architecture)
    try {
        sessionOptions.appendXnnpackProvider();
    } catch (_) {}

    // Load model bytes to avoid platform path issues
    final Uint8List modelBytes = File(modelPath).readAsBytesSync();
    final session = OrtSession.fromBuffer(modelBytes, sessionOptions);

    try {
      final Stopwatch watch = Stopwatch()..start();

      final int outW = inW * factor;
      final int outH = inH * factor;

      final outImage = img.Image(width: outW, height: outH, numChannels: 3);

      // Tile parameters (user assigned double size)
      final int tileSize = 384;
      final int overlap = 16;
      final int step = tileSize - 2 * overlap;

      int totalChunks = ((inH + step - 1) ~/ step) * ((inW + step - 1) ~/ step);
      int chunksProcessed = 0;

      for (int y = 0; y < inH; y += step) {
        for (int x = 0; x < inW; x += step) {
          // Calculate tile source region with overlapping padding
          int startY = y - overlap;
          int startX = x - overlap;
          int endY = y + step + overlap;
          int endX = x + step + overlap;

          // Clamp region purely bounding source image
          int cropStartY = startY < 0 ? 0 : startY;
          int cropStartX = startX < 0 ? 0 : startX;
          int cropEndY = endY > inH ? inH : endY;
          int cropEndX = endX > inW ? inW : endX;

          int cropW = cropEndX - cropStartX;
          int cropH = cropEndY - cropStartY;

          final inputFloats = Float32List(cropH * cropW * 3);
          int idx = 0;
          for (int cy = cropStartY; cy < cropEndY; cy++) {
            int rowOffset = cy * inW;
            for (int cx = cropStartX; cx < cropEndX; cx++) {
              int pxOffset = (rowOffset + cx) * 4;
              inputFloats[idx++] = imagePixels[pxOffset] / 255.0;
              inputFloats[idx++] = imagePixels[pxOffset + 1] / 255.0;
              inputFloats[idx++] = imagePixels[pxOffset + 2] / 255.0;
            }
          }

          final String inputKey = session.inputNames.first;
          final inputOrt = OrtValueTensor.createTensorWithDataList(inputFloats, [
            1,
            cropH,
            cropW,
            3,
          ]);

          final runOptions = OrtRunOptions();
          final Map<String, OrtValue> inputs = {inputKey: inputOrt};
          final List<OrtValue?> outputs = session.run(runOptions, inputs);

          final List<double> outputData = _flattenTensor(outputs[0]?.value);

          int outCropW = cropW * factor;
          int outCropH = cropH * factor;

          // Source Padding mapping relative to destination
          int srcPadLeft = x - cropStartX;
          int srcPadTop = y - cropStartY;

          int dstPadLeft = srcPadLeft * factor;
          int dstPadTop = srcPadTop * factor;

          // Constraints to make sure we don't paste outer bound leftovers
          int targetWriteW = ((x + step) > inW ? inW - x : step) * factor;
          int targetWriteH = ((y + step) > inH ? inH - y : step) * factor;

          int destX = x * factor;
          int destY = y * factor;

          for (int ty = 0; ty < targetWriteH; ty++) {
            int copyY = dstPadTop + ty;
            if (copyY >= outCropH) continue; 
            for (int tx = 0; tx < targetWriteW; tx++) {
              int copyX = dstPadLeft + tx;
              if (copyX >= outCropW) continue;

              int outIdx = (copyY * outCropW + copyX) * 3;
              outImage.setPixelRgb(
                destX + tx,
                destY + ty,
                (outputData[outIdx] * 255).toInt().clamp(0, 255),
                (outputData[outIdx + 1] * 255).toInt().clamp(0, 255),
                (outputData[outIdx + 2] * 255).toInt().clamp(0, 255),
              );
            }
          }

          // Release partials iteratively
          inputOrt.release();
          runOptions.release();
          for (var out in outputs) {
            out?.release();
          }
          chunksProcessed++;
          sendPort?.send(chunksProcessed / totalChunks);
        }
      }

      final resultBytes = img.encodePng(outImage);
      watch.stop();

      return SRRun(
        id: DateTime.now().millisecondsSinceEpoch.toString(),
        modelName: modelName,
        upscaleFactor: factor.toDouble(),
        isProcessing: false,
        resultBytes: resultBytes,
        device: device,
        inferenceTime: "${(watch.elapsedMilliseconds / 1000.0).toStringAsFixed(3)}s",
        metrics: {
          "Resolution": "${inW}x$inH → ${outW}x$outH",
          "Accelerated (Tiles)": "ONNX ($chunksProcessed Chunks)",
        },
      );
    } finally {
      session.release();
      sessionOptions.release();
      OrtEnv.instance.release();
    }
  }

  /// Recursively flattens nested tensors into a 1D list of doubles
  static List<double> _flattenTensor(dynamic tensor) {
    List<double> flatList = [];

    void flatten(dynamic item) {
      if (item is List) {
        for (var element in item) {
          flatten(element);
        }
      } else if (item is num) {
        flatList.add(item.toDouble());
      }
    }

    flatten(tensor);
    return flatList;
  }
}