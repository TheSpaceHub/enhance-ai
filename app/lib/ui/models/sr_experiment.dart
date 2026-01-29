import 'dart:typed_data';

/// Represents a super-resolution project containing the original image and its runs
class SRProject {
  final String id;
  final String name;
  final DateTime timestamp;
  final Uint8List originalBytes;

  List<SRRun> runs;

  SRProject({
    required this.id,
    required this.name,
    required this.timestamp,
    required this.originalBytes,
    this.runs = const [],
  });

  /// Adds new run to project
  void addRun(SRRun run) {
    runs = [run, ...runs];
  }
}

/// Represents a single super-resolution inference run
class SRRun {
  final String id;
  final String modelName;
  final double upscaleFactor;
  final bool isProcessing;
  final Uint8List? resultBytes;
  final Map<String, dynamic> metrics;

  // Nuevos campos técnicos
  final String device; // 'CPU' o 'GPU'
  final String inferenceTime; // Ej: "120ms"

  SRRun({
    required this.id,
    required this.modelName,
    required this.upscaleFactor,
    this.isProcessing = false,
    this.resultBytes,
    this.metrics = const {},
    this.device = 'GPU', // Default
    this.inferenceTime = 'N/A',
  });

  SRRun copyWith({
    bool? isProcessing,
    Uint8List? resultBytes,
    Map<String, dynamic>? metrics,
    String? inferenceTime,
  }) {
    return SRRun(
      id: id,
      modelName: modelName,
      upscaleFactor: upscaleFactor,
      isProcessing: isProcessing ?? this.isProcessing,
      resultBytes: resultBytes ?? this.resultBytes,
      metrics: metrics ?? this.metrics,
      device: device,
      inferenceTime: inferenceTime ?? this.inferenceTime,
    );
  }
}
