import 'dart:convert';
import 'dart:typed_data';
import 'package:http/http.dart' as http;
import '../ui/models/sr_experiment.dart';

class ApiService {
  // Ajusta esta URL según donde corras la app.
  // Android Emulator: 'http://10.0.2.2:8000/upscale'
  // Linux/Web: 'http://127.0.0.1:8000/upscale'
  static const String baseUrl = 'http://127.0.0.1:8000/upscale';

  static Future<SRRun> upscaleImage({
    required Uint8List imageBytes, // Enviamos bytes puros
    required String modelName,
    required double factor,
  }) async {
    try {
      var request = http.MultipartRequest('POST', Uri.parse(baseUrl));

      // Añadir campos
      request.fields['model_name'] = modelName;
      request.fields['scale'] = factor
          .toString(); // Aunque el servidor Python ignora esto por ahora (usa el modelo fijo)

      // Añadir imagen
      request.files.add(
        http.MultipartFile.fromBytes(
          'file',
          imageBytes,
          filename: 'upload.png',
        ),
      );

      print("Enviando petición a Python...");
      final streamedResponse = await request.send();
      final response = await http.Response.fromStream(streamedResponse);

      if (response.statusCode == 200) {
        final data = jsonDecode(response.body);

        if (data['status'] == 'success') {
          // Decodificar la imagen Base64 que devuelve Python
          final String base64Image = data['image'];
          final Uint8List resultBytes = base64Decode(base64Image);

          return SRRun(
            id: DateTime.now().millisecondsSinceEpoch.toString(),
            modelName: modelName,
            upscaleFactor: factor,
            isProcessing: false,
            resultImage: Image.memory(
              resultBytes,
            ), // Creamos imagen desde bytes
            metrics: data['metrics'],
          );
        } else {
          throw Exception(data['message']);
        }
      } else {
        throw Exception("Error del servidor: ${response.statusCode}");
      }
    } catch (e) {
      print("Error en API: $e");
      // Retornamos un run con error (o podrías lanzar excepción)
      return SRRun(
        id: "error",
        modelName: modelName,
        upscaleFactor: factor,
        isProcessing: false,
        metrics: {'Error': e.toString().substring(0, 20)},
      );
    }
  }
}
