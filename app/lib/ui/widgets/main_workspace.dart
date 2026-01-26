import 'dart:typed_data';
import 'package:flutter/material.dart';
import '../../ui/models/sr_experiment.dart';
import 'image_holder.dart';
import 'before_after_slider.dart';

/// Central workspace responsible for image visualization and comparison
class MainWorkspace extends StatefulWidget {
  final Uint8List? originalImageBytes;
  final SRRun? activeRun; // Active Run (Right Slot)
  final SRRun? pinnedRun; // Pinned Run (Left Slot)
  final VoidCallback onUpload;

  const MainWorkspace({
    super.key,
    required this.originalImageBytes,
    required this.activeRun,
    required this.pinnedRun,
    required this.onUpload,
  });

  @override
  State<MainWorkspace> createState() => _MainWorkspaceState();
}

class _MainWorkspaceState extends State<MainWorkspace> {
  // Controls zoom and pan state across image changes
  final TransformationController _transformationController =
      TransformationController();

  @override
  void dispose() {
    _transformationController.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    // Initial state: prompt image upload
    if (widget.originalImageBytes == null) {
      return _buildEmptyState();
    }

    // Processing state: show technical loader
    if (widget.activeRun != null && widget.activeRun!.isProcessing) {
      return _buildProcessingState();
    }

    // Comparison slots
    final Uint8List bytesA =
        widget.pinnedRun?.resultBytes ?? widget.originalImageBytes!;
    final String labelA = widget.pinnedRun != null
        ? "Reference: ${widget.pinnedRun!.modelName}"
        : "Original Input";

    final Uint8List bytesB =
        widget.activeRun?.resultBytes ?? widget.originalImageBytes!;
    final String labelB = widget.activeRun != null
        ? "Active: ${widget.activeRun!.modelName}"
        : "Original Input";

    return Column(
      children: [
        // Workspace toolbar
        _buildTopBar(labelB, labelA),

        Expanded(
          child: Padding(
            padding: const EdgeInsets.all(24.0),
            child: ClipRRect(
              borderRadius: BorderRadius.circular(8),
              child: Container(
                color: Colors.black26,
                child: InteractiveViewer(
                  transformationController: _transformationController,
                  boundaryMargin: const EdgeInsets.all(100),
                  minScale: 0.5,
                  maxScale: 20.0,
                  child: Center(
                    child:
                        (widget.activeRun == null && widget.pinnedRun == null)
                        ? ImageHolder(imageBytes: widget.originalImageBytes)
                        : BeforeAfterSlider(
                            beforeImage: bytesA,
                            afterImage: bytesB,
                          ),
                  ),
                ),
              ),
            ),
          ),
        ),
        _buildBottomHints(),
      ],
    );
  }

  /// Renders the top toolbar with slot labels and zoom controls
  Widget _buildTopBar(String labelA, String labelB) {
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 24, vertical: 12),
      decoration: const BoxDecoration(
        color: Color(0xFF1E1E1E),
        border: Border(bottom: BorderSide(color: Colors.white10)),
      ),
      child: Row(
        mainAxisAlignment: MainAxisAlignment.spaceBetween,
        children: [
          _buildCompactBadge(labelA, isPinned: widget.pinnedRun != null),

          _buildCompactBadge(labelB, isActive: true),
        ],
      ),
    );
  }

  /// Empty workspace shown before any image is loaded.
  Widget _buildEmptyState() {
    return Center(
      child: InkWell(
        onTap: widget.onUpload,
        borderRadius: BorderRadius.circular(20),
        child: Container(
          width: 400,
          height: 300,
          decoration: BoxDecoration(
            gradient: LinearGradient(
              colors: [Color(0xFF2A2A2A), Color(0xFF1E1E1E)],
              begin: Alignment.topLeft,
              end: Alignment.bottomRight,
            ),
            borderRadius: BorderRadius.circular(20),
            border: Border.all(color: Colors.white10, width: 2),
            boxShadow: [
              BoxShadow(
                color: Colors.black45,
                blurRadius: 12,
                offset: Offset(0, 8),
              ),
            ],
          ),
          child: Column(
            mainAxisAlignment: MainAxisAlignment.center,
            children: [
              TweenAnimationBuilder<double>(
                tween: Tween(begin: 1.0, end: 1.0),
                duration: Duration(milliseconds: 300),
                builder: (context, scale, child) {
                  return Transform.scale(scale: scale, child: child);
                },
                child: Icon(
                  Icons.add_photo_alternate_rounded,
                  size: 80,
                  color: Colors.lightBlueAccent,
                ),
              ),
              SizedBox(height: 20),
              Text(
                "Start New Experiment",
                style: TextStyle(
                  color: Colors.white70,
                  fontSize: 20,
                  fontWeight: FontWeight.bold,
                  letterSpacing: 0.5,
                ),
              ),
              SizedBox(height: 10),
              Text(
                "Tap to upload an image",
                style: TextStyle(
                  color: Colors.blueGrey,
                  fontSize: 14,
                  fontStyle: FontStyle.italic,
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }

  /// Loading state displayed while an upscale run is executing.
  Widget _buildProcessingState() {
    return Center(
      child: Column(
        mainAxisSize: MainAxisSize.min,
        children: [
          const CircularProgressIndicator(color: Colors.blueAccent),
          const SizedBox(height: 24),
          Text(
            "Upscaling on ${widget.activeRun!.device}...",
            style: const TextStyle(
              color: Colors.white,
              fontWeight: FontWeight.bold,
            ),
          ),
          Text(
            widget.activeRun!.modelName,
            style: const TextStyle(color: Colors.white38),
          ),
        ],
      ),
    );
  }

  /// Small badge used to label comparison slots.
  Widget _buildCompactBadge(
    String text, {
    bool isPinned = false,
    bool isActive = false,
  }) {
    final color = isPinned
        ? Colors.orangeAccent
        : (isActive ? Colors.blueAccent : Colors.white24);
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 4),
      decoration: BoxDecoration(
        color: color.withOpacity(0.1),
        borderRadius: BorderRadius.circular(4),
        border: Border.all(color: color.withOpacity(0.5)),
      ),
      child: Row(
        children: [
          if (isPinned) Icon(Icons.push_pin, size: 12, color: color),
          if (isPinned) const SizedBox(width: 6),
          Text(
            text,
            style: TextStyle(
              color: color,
              fontSize: 11,
              fontWeight: FontWeight.bold,
            ),
          ),
        ],
      ),
    );
  }

  /// Displays interaction hints for zooming and comparison
  Widget _buildBottomHints() {
    return Container(
      padding: const EdgeInsets.symmetric(vertical: 8),
      child: Row(
        mainAxisAlignment: MainAxisAlignment.center,
        children: const [
          Icon(Icons.mouse_outlined, size: 14, color: Colors.white24),
          SizedBox(width: 8),
          Text(
            "Scroll to zoom • Right click to move image • Left click to move slider",
            style: TextStyle(color: Colors.white24, fontSize: 11),
          ),
        ],
      ),
    );
  }
}
