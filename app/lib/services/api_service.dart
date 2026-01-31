import 'dart:convert';
import 'package:flutter/foundation.dart';
import 'package:http/http.dart' as http;
import '../ui/models/sr_experiment.dart';
import 'dart:async';

/// Service responsible for communicating with the backend AI upscaling API
class ApiService {
  //static const String baseUrl = 'http://127.0.0.1:8000/upscale';
  static const String baseUrl = 'https://danvancea-EnhanceAI.hf.space/upscale';

  static Future<SRRun> upscaleImage({
    required Uint8List imageBytes,
    required String modelName,
    required double factor,
    required String device,
  }) async {
    try {
      var request = http.MultipartRequest('POST', Uri.parse(baseUrl));

      request.fields['model_name'] = modelName;
      request.fields['scale'] = factor.toString();
      request.fields['device'] = device; // Can be "GPU" or CPU

      request.files.add(
        http.MultipartFile.fromBytes(
          'file',
          imageBytes,
          filename: 'upload.png',
        ),
      );

      final streamedResponse = await request.send().timeout(
        const Duration(seconds: 60),
      );
      final response = await http.Response.fromStream(streamedResponse);

      if (response.statusCode == 200) {
        return await compute(_parseSuccessResponse, {
          'body': response.body,
          'modelName': modelName,
          'factor': factor,
          'device': device,
        });
      } else {
        throw Exception(
          "Server Error: ${response.statusCode} - ${response.body}",
        );
      }
    } on TimeoutException {
      throw Exception(
        "The server is taking too long to respond. Check your connection or backend status.",
      );
    } catch (e) {
      throw Exception(
        "Unable to connect to the AI server. Make sure the backend is running.",
      );
    }
  }

  /// Parses a successful backend response and converts it into an SRRun object
  static SRRun _parseSuccessResponse(Map<String, dynamic> params) {
    final body = params['body'] as String;
    final modelName = params['modelName'] as String;
    final factor = params['factor'] as double;
    final device = params['device'] as String;

    final data = jsonDecode(body);

    if (data['status'] == 'success') {
      final String base64Image = data['image'];
      final Uint8List resultBytes = base64Decode(base64Image);

      final String time = data['inference_time']?.toString() ?? "N/A";

      return SRRun(
        id: DateTime.now().millisecondsSinceEpoch.toString(),
        modelName: modelName,
        upscaleFactor: factor,
        isProcessing: false,
        resultBytes: resultBytes,
        metrics: data['metrics'] ?? {},
        device: device,
        inferenceTime: time,
      );
    } else {
      throw Exception(data['message']);
    }
  }
}
