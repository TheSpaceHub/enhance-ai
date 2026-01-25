// lib/services/api_service.dart
import 'dart:convert';
import 'dart:typed_data';
import 'package:flutter/material.dart';
import 'package:http/http.dart' as http;
import '../ui/models/sr_experiment.dart';

class ApiService {
  // CAMBIO IMPORTANTE:
  // Usa 'http://127.0.0.1:8000/upscale' para WEB.
  static const String baseUrl = 'http://127.0.0.1:8000/upscale';

  static Future<SRRun> upscaleImage({
    required Uint8List imageBytes,
    required String modelName,
    required double factor,
  }) async {
    try {
      var request = http.MultipartRequest('POST', Uri.parse(baseUrl));

      request.fields['model_name'] = modelName;
      request.fields['scale'] = factor.toString();

      request.files.add(
        http.MultipartFile.fromBytes(
          'file',
          imageBytes,
          filename: 'upload.png',
        ),
      );

      final streamedResponse = await request.send();
      final response = await http.Response.fromStream(streamedResponse);

      if (response.statusCode == 200) {
        final data = jsonDecode(response.body);

        if (data['status'] == 'success') {
          final String base64Image = data['image'];
          final Uint8List resultBytes = base64Decode(base64Image);

          return SRRun(
            id: DateTime.now().millisecondsSinceEpoch.toString(),
            modelName: modelName,
            upscaleFactor: factor,
            isProcessing: false,
            resultImage: Image.memory(resultBytes),
            metrics: data['metrics'],
          );
        } else {
          throw Exception(data['message']);
        }
      } else {
        throw Exception("Server Error: ${response.statusCode}");
      }
    } catch (e) {
      print("Error API: $e");
      rethrow; // Lanzamos el error para verlo en pantalla si ocurre
    }
  }
}
