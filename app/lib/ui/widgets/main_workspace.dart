// lib/ui/widgets/main_workspace.dart
import 'package:flutter/material.dart';
import '../models/sr_experiment.dart';
import 'image_holder.dart';
import 'before_after_slider.dart';

class MainWorkspace extends StatelessWidget {
  final Image? originalImage;
  final SRRun? activeRun; // El run que estamos viendo ahora
  final VoidCallback onUpload;

  const MainWorkspace({
    super.key,
    required this.originalImage,
    required this.activeRun,
    required this.onUpload,
  });

  @override
  Widget build(BuildContext context) {
    // 1. Estado Vacío (Sin proyecto)
    if (originalImage == null) {
      return Center(
        child: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            const Icon(Icons.science_outlined, size: 64, color: Colors.white10),
            const SizedBox(height: 16),
            const Text(
              "No project selected",
              style: TextStyle(color: Colors.white24),
            ),
            const SizedBox(height: 16),
            OutlinedButton(onPressed: onUpload, child: const Text("Start New")),
          ],
        ),
      );
    }

    // 2. Si hay Run y se está procesando
    if (activeRun != null && activeRun!.isProcessing) {
      return Center(
        child: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            const CircularProgressIndicator(color: Colors.blueAccent),
            const SizedBox(height: 20),
            Text(
              "Upscaling with ${activeRun!.modelName}...",
              style: const TextStyle(color: Colors.white70),
            ),
          ],
        ),
      );
    }

    // 3. Si hay Run terminado -> Comparador
    if (activeRun != null && activeRun!.resultImage != null) {
      return Padding(
        padding: const EdgeInsets.all(24.0),
        child: ClipRRect(
          borderRadius: BorderRadius.circular(8),
          child: BeforeAfterSlider(
            beforeImage: originalImage!,
            afterImage: activeRun!.resultImage!,
          ),
        ),
      );
    }

    // 4. Solo imagen original (Proyecto nuevo sin runs)
    return ImageHolder(image: originalImage!);
  }
}
