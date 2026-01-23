// lib/ui/widgets/main_workspace.dart
import 'package:flutter/material.dart';
import 'image_holder.dart';
import '../widgets/before_after_slider.dart';

class MainWorkspace extends StatelessWidget {
  final bool imageLoaded;
  final Image? originalImage;
  final Image? srImage;
  final VoidCallback onUpload;
  final bool isProcessing;

  const MainWorkspace({
    super.key,
    required this.imageLoaded,
    required this.originalImage,
    required this.srImage,
    required this.onUpload,
    required this.isProcessing,
  });

  @override
  Widget build(BuildContext context) {
    // Stage 1: image is not loaded
    if (!imageLoaded) {
      return Center(
        child: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            const Icon(Icons.image_outlined, size: 80, color: Colors.white54),
            const SizedBox(height: 24),
            ElevatedButton(
              onPressed: onUpload,
              child: const Text('Upload Image'),
            ),
          ],
        ),
      );
    }

    // Stage 2: image loaded but not processed yet
    if (srImage == null && originalImage != null) {
      return ImageHolder(image: originalImage!);
    }

    // Stage 3: loading screen
    if (isProcessing) {
      return const Center(
        child: CircularProgressIndicator(color: Colors.greenAccent),
      );
    }

    // Stage 4: show results Before/After Slider
    if (srImage != null && originalImage != null) {
      return Padding(
        padding: const EdgeInsets.all(16.0),
        child: BeforeAfterSlider(
          beforeImage: originalImage!,
          afterImage: srImage!,
        ),
      );
    }

    return Center(child: originalImage);
  }
}
