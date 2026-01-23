// lib/ui/widgets/before_after_slider.dart
import 'package:flutter/material.dart';
import 'package:before_after/before_after.dart';
import 'image_holder.dart';

class BeforeAfterSlider extends StatefulWidget {
  final Image beforeImage;
  final Image afterImage;

  const BeforeAfterSlider({
    super.key,
    required this.beforeImage,
    required this.afterImage,
  });

  @override
  State<BeforeAfterSlider> createState() => _BeforeAfterSliderState();
}

class _BeforeAfterSliderState extends State<BeforeAfterSlider> {
  double value = 0.5;

  @override
  Widget build(BuildContext context) {
    return BeforeAfter(
      value: value,
      before: ImageHolder(image: widget.beforeImage),
      after: ImageHolder(image: widget.afterImage),
      onValueChanged: (v) {
        setState(() => value = v);
      },
    );
  }
}
