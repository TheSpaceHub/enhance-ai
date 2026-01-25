// lib/ui/models/sr_experiment.dart
import 'dart:typed_data';
import 'package:flutter/material.dart';

// 1. EL PROYECTO (La "Caja" contenedora: Una imagen subida)
class SRProject {
  final String id;
  final String name;
  final DateTime timestamp;
  final Image originalImage;
  final Uint8List rawBytes;

  // Lista de intentos (Runs) sobre esta imagen
  final List<SRRun> runs;

  SRProject({
    required this.id,
    required this.name,
    required this.timestamp,
    required this.originalImage,
    required this.rawBytes,
    this.runs = const [],
  });

  // Helper para añadir un Run nuevo y devolver una COPIA del proyecto
  SRProject addRun(SRRun run) {
    return SRProject(
      id: id,
      name: name,
      timestamp: timestamp,
      originalImage: originalImage,
      rawBytes: rawBytes,
      runs: [run, ...runs], // Lo pone el primero de la lista
    );
  }
}

// 2. EL RUN (Un intento específico: Modelo + Params + Resultado)
class SRRun {
  final String id;
  final String modelName;
  final double upscaleFactor;
  final bool isProcessing;

  final Image? resultImage;
  final Map<String, dynamic> metrics;

  SRRun({
    required this.id,
    required this.modelName,
    required this.upscaleFactor,
    this.isProcessing = false,
    this.resultImage,
    this.metrics = const {},
  });

  SRRun copyWith({
    bool? isProcessing,
    Image? resultImage,
    Map<String, dynamic>? metrics,
  }) {
    return SRRun(
      id: id,
      modelName: modelName,
      upscaleFactor: upscaleFactor,
      isProcessing: isProcessing ?? this.isProcessing,
      resultImage: resultImage ?? this.resultImage,
      metrics: metrics ?? this.metrics,
    );
  }
}
