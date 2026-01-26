import 'dart:typed_data';
import 'package:flutter/material.dart';
import 'package:before_after/before_after.dart';
import 'image_holder.dart';

/// Interactive slider widget that allows visual comparison between two images
class BeforeAfterSlider extends StatefulWidget {
  final Uint8List? beforeImage;
  final Uint8List? afterImage;

  const BeforeAfterSlider({
    super.key,
    required this.beforeImage,
    required this.afterImage,
  });

  @override
  State<BeforeAfterSlider> createState() => _BeforeAfterSliderState();
}

/// Manages the slider position state for before/after image comparison
class _BeforeAfterSliderState extends State<BeforeAfterSlider> {
  double value = 0.5;

  @override
  Widget build(BuildContext context) {
    return BeforeAfter(
      value: value,
      before: ImageHolder(imageBytes: widget.beforeImage),
      after: ImageHolder(imageBytes: widget.afterImage),
      onValueChanged: (v) {
        setState(() => value = v);
      },
    );
  }
}
