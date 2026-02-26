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
  final bool showLeftPanel;
  final VoidCallback onCloseLeftPanel;
  final bool showRightPanel;
  final VoidCallback onCloseRightPanel;
  final bool haveProject;

  const MainWorkspace({
    super.key,
    required this.originalImageBytes,
    required this.activeRun,
    required this.pinnedRun,
    required this.onUpload,
    required this.showLeftPanel,
    required this.onCloseLeftPanel,
    required this.showRightPanel,
    required this.onCloseRightPanel,
    required this.haveProject,
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
    // Comparison slots
    final Uint8List? bytesA =
        widget.activeRun?.resultBytes ?? widget.originalImageBytes;
    final String labelB = widget.activeRun != null
        ? "${widget.activeRun!.modelName} (x${widget.activeRun!.upscaleFactor.toInt()})"
        : "Original Input";

    final Uint8List? bytesB =
        widget.pinnedRun?.resultBytes ?? widget.originalImageBytes;
    final String labelA = widget.pinnedRun != null
        ? "${widget.pinnedRun!.modelName} (x${widget.pinnedRun!.upscaleFactor.toInt()})"
        : "Original Input";

    Widget mainContent;

    // Initial state: prompt image upload
    if (widget.originalImageBytes == null) {
      mainContent = _buildEmptyState();
    }
    // Processing state: show technical loader
    else if (widget.activeRun != null && widget.activeRun!.isProcessing) {
      mainContent = _buildProcessingState();
    }
    // Comparison view
    else {
      mainContent = Column(
        children: [
          Expanded(
            child: Padding(
              padding: const EdgeInsets.all(16.0),
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
        ],
      );
    }

    return Column(
      children: [
        // Workspace toolbar is always visible
        _buildTopBar(labelA, labelB),

        Expanded(child: SafeArea(top: false, child: mainContent)),
      ],
    );
  }

  /// Renders the top toolbar with slot labels and zoom controls
  Widget _buildTopBar(String labelA, String labelB) {
    return Container(
      padding: const EdgeInsets.only(left: 8, right: 8, bottom: 12),
      decoration: const BoxDecoration(
        color: Color(0xFF1E1E1E),
        border: Border(bottom: BorderSide(color: Colors.white10)),
      ),
      child: SafeArea(
        bottom: false,
        child: Padding(
          padding: const EdgeInsets.only(top: 8.0),
          child: Row(
            mainAxisAlignment: MainAxisAlignment.spaceBetween,
            children: [
              !widget.showLeftPanel
                  ? IconButton(
                      icon: const Icon(Icons.menu_open),
                      tooltip: "Open History",
                      onPressed: widget.onCloseLeftPanel,
                    )
                  : SizedBox.shrink(),
              //const SizedBox(width: 12),
              _buildCompactBadge(labelA, isPinned: widget.pinnedRun != null),
              const Spacer(),
              if (widget.haveProject)
                _buildCompactBadge(labelB, isActive: widget.activeRun != null),
              //const SizedBox(width: 12),
              if (!widget.showRightPanel && widget.haveProject)
                IconButton(
                  icon: const Icon(Icons.settings),
                  tooltip: "Open Enhance Config",
                  onPressed: widget.onCloseRightPanel,
                ),
            ],
          ),
        ),
      ),
    );
  }

  /// Empty workspace shown before any image is loaded.
  Widget _buildEmptyState() {
    return Center(
      child: Padding(
        padding: const EdgeInsets.all(8.0),
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
                  "Click here to upload an image",
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
          Text("${widget.activeRun!.progress.toStringAsFixed(0)}%"),
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
        color: color.withAlpha(32),
        borderRadius: BorderRadius.circular(4),
        border: Border.all(color: color.withAlpha(128)),
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

          if (isActive) const SizedBox(width: 6),
          if (isActive) Icon(Icons.auto_awesome, size: 12, color: color),
        ],
      ),
    );
  }
}
