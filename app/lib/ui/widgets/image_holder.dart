import 'dart:typed_data';
import 'package:flutter/material.dart';

/// Displays an image from memory while preserving aspect ratio and pixel accuracy
class ImageHolder extends StatelessWidget {
  final Uint8List? imageBytes;
  final double maxWidthFactor;
  final double maxHeightFactor;

  const ImageHolder({
    super.key,
    required this.imageBytes,
    this.maxWidthFactor = 0.85,
    this.maxHeightFactor = 0.85,
  });

  @override
  Widget build(BuildContext context) {
    if (imageBytes == null) return const SizedBox.shrink();

    return LayoutBuilder(
      builder: (context, constraints) {
        final targetWidth = constraints.maxWidth * maxWidthFactor;
        final targetHeight = constraints.maxHeight * maxHeightFactor;

        return Center(
          child: SizedBox(
            width: targetWidth,
            height: targetHeight,
            child: FittedBox(
              fit: BoxFit.contain,
              child: Image.memory(
                imageBytes!,
                gaplessPlayback: true,
                filterQuality: FilterQuality.none,
              ),
            ),
          ),
        );
      },
    );
  }
}
