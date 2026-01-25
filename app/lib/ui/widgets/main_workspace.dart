// lib/ui/widgets/main_workspace.dart
import 'package:flutter/material.dart';
import '../models/sr_experiment.dart';
import 'image_holder.dart';
import 'before_after_slider.dart';

class MainWorkspace extends StatelessWidget {
  final Image? originalImage;
  final SRRun? activeRun;
  final VoidCallback onUpload;

  const MainWorkspace({
    super.key,
    required this.originalImage,
    required this.activeRun,
    required this.onUpload,
  });

  @override
  Widget build(BuildContext context) {
    // 1. Start new button
    if (originalImage == null) {
      return Center(
        child: MouseRegion(
          cursor: SystemMouseCursors.click,
          child: GestureDetector(
            onTap: onUpload,
            child: Container(
              width: 500,
              height: 350,
              decoration: BoxDecoration(
                color: const Color(0xFF252526),
                borderRadius: BorderRadius.circular(20),
                border: Border.all(color: Colors.white12, width: 2),
                boxShadow: [
                  BoxShadow(
                    color: Colors.black.withOpacity(0.3),
                    blurRadius: 20,
                    offset: const Offset(0, 10),
                  ),
                ],
              ),
              child: Column(
                mainAxisAlignment: MainAxisAlignment.center,
                children: [
                  Container(
                    padding: const EdgeInsets.all(20),
                    decoration: BoxDecoration(
                      color: Colors.blueAccent.withOpacity(0.1),
                      shape: BoxShape.circle,
                    ),
                    child: const Icon(
                      Icons.add_photo_alternate_outlined,
                      size: 64,
                      color: Colors.blueAccent,
                    ),
                  ),
                  const SizedBox(height: 24),
                  const Text(
                    "Start a New Project",
                    style: TextStyle(
                      color: Colors.white,
                      fontSize: 22,
                      fontWeight: FontWeight.bold,
                    ),
                  ),
                  const SizedBox(height: 12),
                  const Text(
                    "Click here to upload an image from your device",
                    style: TextStyle(color: Colors.white54, fontSize: 14),
                  ),
                  const SizedBox(height: 32),
                  ElevatedButton.icon(
                    onPressed: onUpload, // <--- BOTÓN DENTRO TAMBIÉN FUNCIONA
                    icon: const Icon(Icons.upload_file),
                    label: const Text("Browse Files"),
                    style: ElevatedButton.styleFrom(
                      backgroundColor: Colors.blueAccent,
                      foregroundColor: Colors.white,
                      padding: const EdgeInsets.symmetric(
                        horizontal: 32,
                        vertical: 16,
                      ),
                      textStyle: const TextStyle(
                        fontSize: 16,
                        fontWeight: FontWeight.bold,
                      ),
                    ),
                  ),
                ],
              ),
            ),
          ),
        ),
      );
    }

    // 2. PROCESANDO
    if (activeRun != null && activeRun!.isProcessing) {
      return Center(
        child: Container(
          padding: const EdgeInsets.all(32),
          decoration: BoxDecoration(
            color: const Color(0xFF252526),
            borderRadius: BorderRadius.circular(16),
          ),
          child: Column(
            mainAxisSize: MainAxisSize.min,
            children: [
              const SizedBox(
                width: 60,
                height: 60,
                child: CircularProgressIndicator(
                  color: Colors.blueAccent,
                  strokeWidth: 6,
                ),
              ),
              const SizedBox(height: 24),
              Text(
                "Enhancing with ${activeRun!.modelName}...",
                style: const TextStyle(
                  color: Colors.white,
                  fontSize: 16,
                  fontWeight: FontWeight.w500,
                ),
              ),
              const SizedBox(height: 8),
              const Text(
                "This might take a few seconds",
                style: TextStyle(color: Colors.white38),
              ),
            ],
          ),
        ),
      );
    }

    // 3. RESULTADO (Slider)
    if (activeRun != null && activeRun!.resultImage != null) {
      return Padding(
        padding: const EdgeInsets.all(24.0),
        child: ClipRRect(
          borderRadius: BorderRadius.circular(12),
          child: BeforeAfterSlider(
            beforeImage: originalImage!,
            afterImage: activeRun!.resultImage!,
          ),
        ),
      );
    }

    // 4. SOLO IMAGEN ORIGINAL
    return ImageHolder(image: originalImage!);
  }
}
