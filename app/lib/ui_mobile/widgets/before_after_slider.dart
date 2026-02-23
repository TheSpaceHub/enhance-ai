import 'dart:typed_data';
import 'package:flutter/material.dart';
import 'image_holder.dart';

/// Interactive slider widget that allows visual comparison between two images
/// and propagates unused gestures to the parent InteractiveViewer.
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
  double _value = 0.5;
  final GlobalKey _containerKey = GlobalKey();

  void _onDragUpdate(DragUpdateDetails details) {
    final RenderBox? box =
        _containerKey.currentContext?.findRenderObject() as RenderBox?;
    if (box != null && box.hasSize) {
      setState(() {
        _value += details.primaryDelta! / box.size.width;
        _value = _value.clamp(0.0, 1.0);
      });
    }
  }

  @override
  Widget build(BuildContext context) {
    if (widget.beforeImage == null || widget.afterImage == null) {
      return const SizedBox.shrink();
    }

    return LayoutBuilder(
      builder: (context, constraints) {
        // Clamp the effective position so the slider never extends past the thumb's radius
        final double pixelPos = (constraints.maxWidth * _value).clamp(
          16.0,
          constraints.maxWidth - 16.0,
        );
        final double effectiveValue = pixelPos / constraints.maxWidth;

        return SizedBox(
          key: _containerKey,
          width: constraints.maxWidth,
          height: constraints.maxHeight,
          child: GestureDetector(
            onTapUp: (details) {
              setState(() {
                _value = (details.localPosition.dx / constraints.maxWidth)
                    .clamp(0.0, 1.0);
              });
            },
            child: Stack(
              fit: StackFit.expand,
              children: [
                // Before Image (Background)
                ImageHolder(imageBytes: widget.beforeImage),

                // After Image (Foreground, Clipped)
                ClipRect(
                  clipper: _RectClipper(effectiveValue),
                  child: ImageHolder(imageBytes: widget.afterImage),
                ),

                // Slider and Thumb overlay
                Align(
                  alignment: Alignment.centerLeft,
                  child: Container(
                    margin: EdgeInsets.only(left: pixelPos - 1),
                    width: 2,
                    color: Colors.white,
                  ),
                ),
                Align(
                  alignment: Alignment.centerLeft,
                  child: Container(
                    margin: EdgeInsets.only(left: pixelPos - 16),
                    child: GestureDetector(
                      // We only capture horizontal drags specifically on the thumb.
                      // This lets the rest of the Stack pass touches to InteractiveViewer.
                      onHorizontalDragUpdate: _onDragUpdate,
                      onHorizontalDragDown: (details) {}, // Claim the gesture
                      child: Container(
                        width: 32,
                        height: 32,
                        decoration: const BoxDecoration(
                          color: Colors.white,
                          shape: BoxShape.circle,
                          boxShadow: [
                            BoxShadow(color: Colors.black26, blurRadius: 4),
                          ],
                        ),
                        child: const Icon(
                          Icons.code,
                          color: Colors.black,
                          size: 18,
                        ),
                      ),
                    ),
                  ),
                ),
              ],
            ),
          ),
        );
      },
    );
  }
}

/// Clipper to slice the Before image based on the slider value
class _RectClipper extends CustomClipper<Rect> {
  final double value;

  _RectClipper(this.value);

  @override
  Rect getClip(Size size) {
    return Rect.fromLTWH(0, 0, size.width * value, size.height);
  }

  @override
  bool shouldReclip(_RectClipper oldClipper) => value != oldClipper.value;
}
