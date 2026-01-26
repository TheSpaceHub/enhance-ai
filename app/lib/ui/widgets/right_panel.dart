import 'package:flutter/material.dart';
import '../models/sr_experiment.dart';
import '../models/model_info.dart';
import 'package:file_saver/file_saver.dart';
import 'dart:typed_data';

/// Configuration and results panel for running and inspecting upscale experiments
class RightPanel extends StatefulWidget {
  final bool hasProject;
  final List<SRRun> runsHistory;
  final SRRun? activeRun; // Active Run (Right Slot)
  final String? pinnedRunId; // Pinned Run (Left Slot)

  final Function(SRRun) onRunSelect;
  final Function(String) onTogglePin;
  final Function(String model, double factor) onUpscale;
  final String selectedDevice;
  final Function(String) onDeviceChanged;
  final VoidCallback onClose;

  const RightPanel({
    super.key,
    required this.hasProject,
    required this.runsHistory,
    required this.activeRun,
    required this.pinnedRunId,
    required this.onRunSelect,
    required this.onTogglePin,
    required this.onUpscale,
    required this.selectedDevice,
    required this.onDeviceChanged,
    required this.onClose,
  });

  @override
  State<RightPanel> createState() => _RightPanelState();
}

/// Manages local UI state for model selection, scaling, and downloads
class _RightPanelState extends State<RightPanel> {
  double upscaleFactor = 4;
  String selectedModel = 'CNNU';
  final List<String> models = ['CNNU', 'ESPCN', 'SRResNet', 'SRGAN'];

  /// TODO:
  void _showModelInfo(String modelKey) {
    final info = modelRegistry[modelKey];
    if (info == null) return;

    showDialog(
      context: context,
      builder: (context) => AlertDialog(
        backgroundColor: const Color(0xFF2D2D30),
        shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(16)),
        title: Row(
          children: [
            const Icon(Icons.auto_awesome, color: Colors.blueAccent),
            const SizedBox(width: 12),
            Text(info.name, style: const TextStyle(color: Colors.white)),
          ],
        ),
        content: Column(
          mainAxisSize: MainAxisSize.min,
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Text(
              info.description,
              style: const TextStyle(color: Colors.white70),
            ),
            const SizedBox(height: 20),
            const Text(
              "STRENGTHS",
              style: TextStyle(
                color: Colors.white30,
                fontSize: 10,
                fontWeight: FontWeight.bold,
              ),
            ),
            const SizedBox(height: 8),
            Wrap(
              spacing: 8,
              children: info.strengths
                  .map(
                    (s) => Chip(
                      label: Text(s, style: const TextStyle(fontSize: 11)),
                      backgroundColor: Colors.blueAccent.withValues(alpha: 0.1),
                      side: const BorderSide(
                        color: Colors.blueAccent,
                        width: 0.5,
                      ),
                    ),
                  )
                  .toList(),
            ),
            const SizedBox(height: 16),
            _buildInfoRow("Computational Complexity:", info.complexity),
          ],
        ),
        actions: [
          TextButton(
            onPressed: () => Navigator.pop(context),
            child: const Text(
              "CLOSE",
              style: TextStyle(color: Colors.blueAccent),
            ),
          ),
        ],
      ),
    );
  }

  /// Saves the generated image to disk
  Future<void> _downloadImage(Uint8List bytes, String name) async {
    try {
      await FileSaver.instance.saveFile(
        name: name,
        bytes: bytes,
        mimeType: MimeType.png,
      );
    } catch (e) {
      debugPrint("Error al descargar: $e");
    }
  }

  /// Displays a label-value row used inside info dialogs
  Widget _buildInfoRow(String label, String value) {
    return Row(
      children: [
        Text(
          label,
          style: const TextStyle(color: Colors.white38, fontSize: 12),
        ),
        const SizedBox(width: 8),
        Text(
          value,
          style: const TextStyle(
            color: Colors.orangeAccent,
            fontSize: 12,
            fontWeight: FontWeight.bold,
          ),
        ),
      ],
    );
  }

  @override
  Widget build(BuildContext context) {
    if (!widget.hasProject) return const SizedBox.shrink();

    final isProcessing = widget.activeRun?.isProcessing ?? false;

    return Container(
      width: 320,
      color: const Color(0xFF252526),
      child: Column(
        children: [
          Expanded(
            child: ListView(
              padding: const EdgeInsets.all(20),
              children: [
                Row(
                  mainAxisAlignment: MainAxisAlignment.spaceBetween,
                  children: [
                    _buildHeader("CONFIGURATION"),
                    IconButton(
                      icon: const Icon(Icons.close, size: 18),
                      color: Colors.white54,
                      tooltip: "Hide panel",
                      onPressed: widget.onClose,
                    ),
                  ],
                ),
                // Model selection
                Row(
                  mainAxisAlignment: MainAxisAlignment.spaceBetween,
                  children: [
                    _buildLabel("Model Architecture"),
                    IconButton(
                      icon: const Icon(
                        Icons.info_outline,
                        size: 18,
                        color: Colors.white30,
                      ),
                      onPressed: () => _showModelInfo(selectedModel),
                      padding: EdgeInsets.zero,
                      constraints: const BoxConstraints(),
                    ),
                  ],
                ),
                Container(
                  padding: const EdgeInsets.symmetric(horizontal: 12),
                  decoration: BoxDecoration(
                    color: Colors.black26,
                    borderRadius: BorderRadius.circular(4),
                    border: Border.all(color: Colors.white12),
                  ),
                  child: DropdownButtonHideUnderline(
                    child: DropdownButton<String>(
                      value: selectedModel,
                      dropdownColor: const Color(0xFF333333),
                      isExpanded: true,
                      style: const TextStyle(color: Colors.white),
                      items: models
                          .map(
                            (m) => DropdownMenuItem(value: m, child: Text(m)),
                          )
                          .toList(),
                      onChanged: isProcessing
                          ? null
                          : (v) => setState(() => selectedModel = v!),
                    ),
                  ),
                ),

                const SizedBox(height: 20),

                // Hardware Selectors
                _buildLabel("Processing Unit"),
                const SizedBox(height: 8),
                SegmentedButton<String>(
                  segments: const [
                    ButtonSegment(
                      value: 'CPU',
                      label: Text('CPU'),
                      icon: Icon(Icons.computer),
                    ),
                    ButtonSegment(
                      value: 'GPU',
                      label: Text('GPU'),
                      icon: Icon(Icons.bolt),
                    ),
                  ],
                  selected: {widget.selectedDevice},
                  onSelectionChanged: isProcessing
                      ? null
                      : (newSelection) =>
                            widget.onDeviceChanged(newSelection.first),
                  style: ButtonStyle(
                    backgroundColor: WidgetStateProperty.resolveWith<Color>((
                      Set<WidgetState> states,
                    ) {
                      if (states.contains(WidgetState.selected)) {
                        return Colors.blueAccent.withValues(alpha: 0.2);
                      }
                      return Colors.black12;
                    }),
                    foregroundColor: WidgetStateProperty.all(Colors.white),
                  ),
                ),

                const SizedBox(height: 20),

                // Slider "scale factor"
                Row(
                  mainAxisAlignment: MainAxisAlignment.spaceBetween,
                  children: [
                    _buildLabel("Scale Factor"),
                    Text(
                      "x${upscaleFactor.toInt()}",
                      style: const TextStyle(
                        color: Colors.blueAccent,
                        fontWeight: FontWeight.bold,
                      ),
                    ),
                  ],
                ),
                Slider(
                  value: upscaleFactor,
                  min: 2,
                  max: 8,
                  divisions: 3,
                  activeColor: Colors.blueAccent,
                  onChanged: isProcessing
                      ? null
                      : (v) => setState(() => upscaleFactor = v),
                ),

                const SizedBox(height: 24),

                // "Run Button"
                SizedBox(
                  width: double.infinity,
                  height: 50,
                  child: ElevatedButton(
                    onPressed: isProcessing
                        ? null
                        : () => widget.onUpscale(selectedModel, upscaleFactor),
                    style: ElevatedButton.styleFrom(
                      backgroundColor: Colors.blueAccent,
                      foregroundColor: Colors.white,
                    ),
                    child: isProcessing
                        ? const SizedBox(
                            width: 20,
                            height: 20,
                            child: CircularProgressIndicator(
                              strokeWidth: 2,
                              color: Colors.white,
                            ),
                          )
                        : const Text(
                            'RUN UPSCALE',
                            style: TextStyle(fontWeight: FontWeight.bold),
                          ),
                  ),
                ),

                const SizedBox(height: 40),

                // Show metrics
                if (widget.activeRun != null && !isProcessing) ...[
                  _buildHeader("CURRENT RUN METRICS"),
                  const SizedBox(height: 16),

                  _buildMetricRow(
                    "Device",
                    widget.activeRun!.device,
                    color: Colors.orangeAccent,
                  ),
                  _buildMetricRow(
                    "Time",
                    widget.activeRun!.inferenceTime,
                    color: Colors.greenAccent,
                  ),
                  const Divider(color: Colors.white10, height: 24),

                  ...widget.activeRun!.metrics.entries.map(
                    (e) => _buildMetricRow(e.key, e.value),
                  ),
                  const SizedBox(height: 16),
                  SizedBox(
                    width: double.infinity,
                    child: OutlinedButton.icon(
                      onPressed: () => _downloadImage(
                        widget.activeRun!.resultBytes!,
                        "SR_${widget.activeRun!.modelName}_${DateTime.now().millisecondsSinceEpoch}.png",
                      ),
                      icon: const Icon(Icons.download, size: 18),
                      label: const Text("DOWNLOAD RESULT"),
                      style: OutlinedButton.styleFrom(
                        foregroundColor: Colors.tealAccent,
                        side: const BorderSide(color: Colors.tealAccent),
                      ),
                    ),
                  ),
                ],
              ],
            ),
          ),

          // History selector
          Container(
            height: 250,
            decoration: const BoxDecoration(
              color: Color(0xFF1E1E1E),
              border: Border(top: BorderSide(color: Colors.white10)),
            ),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Padding(
                  padding: const EdgeInsets.all(16.0),
                  child: _buildHeader("RUN HISTORY"),
                ),
                Expanded(
                  child: ListView.builder(
                    itemCount: widget.runsHistory.length,
                    itemBuilder: (context, index) {
                      final run = widget.runsHistory[index];
                      final isActive = run.id == widget.activeRun?.id;
                      final isPinned = run.id == widget.pinnedRunId;

                      return Container(
                        color: isActive
                            ? Colors.white.withValues(alpha: 0.05)
                            : null,
                        child: ListTile(
                          dense: true,
                          onTap: () => widget.onRunSelect(run),
                          leading: run.isProcessing
                              ? const SizedBox(
                                  width: 16,
                                  height: 16,
                                  child: CircularProgressIndicator(
                                    strokeWidth: 2,
                                  ),
                                )
                              : Icon(
                                  Icons.check_circle,
                                  color: Colors.greenAccent,
                                  size: 18,
                                ),

                          title: Text(
                            "${run.modelName} (x${run.upscaleFactor.toInt()})",
                            style: TextStyle(
                              color: isActive ? Colors.white : Colors.white70,
                              fontWeight: isActive
                                  ? FontWeight.bold
                                  : FontWeight.normal,
                              fontSize: 13,
                            ),
                          ),
                          subtitle: Text(
                            "${run.device} • ${run.inferenceTime}",
                            style: const TextStyle(
                              color: Colors.white30,
                              fontSize: 11,
                            ),
                          ),
                          trailing: IconButton(
                            icon: Icon(
                              isPinned
                                  ? Icons.push_pin
                                  : Icons.push_pin_outlined,
                              color: isPinned
                                  ? Colors.orangeAccent
                                  : Colors.white24,
                              size: 18,
                            ),
                            tooltip: isPinned
                                ? "Unpin (Reset to Original)"
                                : "Pin to Compare (Slot A)",
                            onPressed: () => widget.onTogglePin(run.id),
                          ),
                        ),
                      );
                    },
                  ),
                ),
              ],
            ),
          ),
        ],
      ),
    );
  }

  /// Builds a section header label used to separate panel sections
  Widget _buildHeader(String text) {
    return Text(
      text,
      style: const TextStyle(
        color: Colors.white54,
        fontSize: 11,
        fontWeight: FontWeight.bold,
        letterSpacing: 1.0,
      ),
    );
  }

  /// Builds a standard form label for configuration controls
  Widget _buildLabel(String text) {
    return Text(
      text,
      style: const TextStyle(color: Colors.white, fontSize: 13),
    );
  }

  /// Renders a single metric row displaying a key-value pair
  Widget _buildMetricRow(String key, dynamic value, {Color? color}) {
    return Padding(
      padding: const EdgeInsets.only(bottom: 6.0),
      child: Row(
        mainAxisAlignment: MainAxisAlignment.spaceBetween,
        children: [
          Text(
            key,
            style: const TextStyle(color: Colors.white60, fontSize: 12),
          ),
          Text(
            value.toString(),
            style: TextStyle(
              color: color ?? Colors.white,
              fontWeight: FontWeight.bold,
              fontFamily: 'Monospace',
              fontSize: 12,
            ),
          ),
        ],
      ),
    );
  }
}
