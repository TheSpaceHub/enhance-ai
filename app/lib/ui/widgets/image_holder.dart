// lib/ui/widgets/image_holder.dart
import 'package:flutter/material.dart';

class ImageHolder extends StatelessWidget {
  final Image image;
  final double maxWidthFactor; // ej: 0.65 → 65% del workspace
  final double maxHeightFactor; // por si quieres limitar también en vertical

  const ImageHolder({
    super.key,
    required this.image,
    this.maxWidthFactor = 1.0,
    this.maxHeightFactor = 1.0,
  });

  @override
  Widget build(BuildContext context) {
    return LayoutBuilder(
      builder: (context, constraints) {
        final maxWidth = constraints.maxWidth * maxWidthFactor;
        final maxHeight = constraints.maxHeight * maxHeightFactor;

        return Center(
          child: ConstrainedBox(
            constraints: BoxConstraints(
              maxWidth: maxWidth,
              maxHeight: maxHeight,
            ),
            child: AspectRatio(
              aspectRatio: image.width != null && image.height != null
                  ? image.width! / image.height!
                  : 1.0,
              child: Image(image: image.image, fit: BoxFit.contain),
            ),
          ),
        );
      },
    );
  }
}
