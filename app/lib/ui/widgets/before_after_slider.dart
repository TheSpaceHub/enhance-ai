// lib/ui/widgets/before_after_slider.dart
import 'package:flutter/material.dart';
import 'package:before_after/before_after.dart';

class BeforeAfterImage extends StatelessWidget {
  final Image original;
  final Image processed;

  const BeforeAfterImage({
    super.key,
    required this.original,
    required this.processed,
  });

  @override
  Widget build(BuildContext context) {
    return BeforeAfter(
      beforeImage: original,
      afterImage: processed,
      sliderColor: Colors.greenAccent,
      thumbColor: Colors.greenAccent,
      isVertical: false, // horizontal slider
    );
  }
}
