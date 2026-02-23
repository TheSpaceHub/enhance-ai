import 'package:flutter/material.dart';
import '../../ui/models/sr_experiment.dart';
import '../../ui/models/model_info.dart';
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
  bool _modelInfoExpanded = false;
  double upscaleFactor = 4;
  String selectedModel = 'Average';
  final List<String> models = ['Average', 'CNNU', 'ESPCN', 'SRResNet', 'SRGAN'];

  /// Shows detailed information about the selected model
  Widget _showModelInfo(String modelKey) {
    final info = modelRegistry[modelKey];
    if (info == null) return SizedBox.shrink();

    return Container(
      padding: const EdgeInsets.all(12),
      margin: const EdgeInsets.only(top: 12, bottom: 16),
      decoration: BoxDecoration(
        color: Colors.black26,
        borderRadius: BorderRadius.circular(8),
        border: Border.all(color: Colors.white12),
      ),

      // Description
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Text(
            info.name,
            style: const TextStyle(
              fontWeight: FontWeight.bold,
              color: Colors.blueAccent,
            ),
          ),
          const SizedBox(height: 8),
          Text(
            info.description,
            style: const TextStyle(color: Colors.white70, fontSize: 12),
          ),
          const SizedBox(height: 12),
          const Text(
            "IDEAL FOR:",
            style: TextStyle(
              color: Colors.white30,
              fontSize: 10,
              fontWeight: FontWeight.bold,
            ),
          ),
          Text(
            info.recommendedFor,
            style: const TextStyle(color: Colors.tealAccent, fontSize: 11),
          ),
          const SizedBox(height: 12),

          // Strenghts
          Wrap(
            spacing: 6,
            runSpacing: 6,
            children: info.strengths
                .map(
                  (s) => Chip(
                    label: Text(
                      s,
                      style: const TextStyle(fontSize: 10, color: Colors.white),
                    ),
                    backgroundColor: Colors.blueAccent.withValues(alpha: 0.1),
                    padding: EdgeInsets.zero,
                    materialTapTargetSize: MaterialTapTargetSize.shrinkWrap,
                    side: const BorderSide(
                      color: Colors.blueAccent,
                      width: 0.5,
                    ),
                  ),
                )
                .toList(),
          ),
          const SizedBox(height: 16),

          // Speed bar
          Row(
            children: [
              const Text(
                "SPEED  ",
                style: TextStyle(
                  color: Colors.white30,
                  fontSize: 10,
                  fontWeight: FontWeight.bold,
                ),
              ),
              Expanded(
                child: ClipRRect(
                  borderRadius: BorderRadius.circular(2),
                  child: LinearProgressIndicator(
                    value: info.speedScore,
                    minHeight: 4,
                    backgroundColor: Colors.white10,
                    valueColor: AlwaysStoppedAnimation<Color>(
                      info.speedScore > 0.7
                          ? Colors.greenAccent
                          : info.speedScore > 0.3
                          ? Colors.orangeAccent
                          : Colors.redAccent,
                    ),
                  ),
                ),
              ),
            ],
          ),
        ],
      ),
    );
  }

  /// Toggles and displays the "About Model" section
  Widget aboutModel(String modelKey) {
    final info = modelRegistry[modelKey];
    final label = info != null ? 'About ${info.name}' : 'Model Details';

    return Padding(
      padding: const EdgeInsets.symmetric(vertical: 8.0),
      child: InkWell(
        onTap: () => setState(() => _modelInfoExpanded = !_modelInfoExpanded),
        child: Row(
          mainAxisSize: MainAxisSize.min,
          children: [
            Icon(
              _modelInfoExpanded
                  ? Icons.keyboard_arrow_up
                  : Icons.keyboard_arrow_down,
              size: 18,
              color: Colors.blueAccent,
            ),
            const SizedBox(width: 4),
            Text(
              label,
              style: const TextStyle(
                color: Colors.blueAccent,
                fontSize: 12,
                fontWeight: FontWeight.w600,
              ),
            ),
          ],
        ),
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

  @override
  Widget build(BuildContext context) {
    if (!widget.hasProject) return const SizedBox.shrink();

    final isProcessing = widget.activeRun?.isProcessing ?? false;

    return Container(
      width: 320,
      color: const Color.fromARGB(255, 37, 37, 37),
      child: SafeArea(
        child: Column(
          children: [
            Expanded(
              child: ListView(
                padding: const EdgeInsets.all(20),
                children: [
                  Padding(
                    padding: const EdgeInsets.only(bottom: 16),
                    child: Row(
                      mainAxisAlignment: MainAxisAlignment.spaceBetween,
                      children: [
                        _buildHeader("CONFIGURATION", 16),

                        // Botón cerrar solo si hay proyecto
                        if (widget.hasProject)
                          IconButton(
                            icon: const Icon(
                              Icons.close,
                              color: Colors.white30,
                            ),
                            tooltip: "Close Panel",
                            onPressed: widget.onClose,
                            padding: EdgeInsets.zero,
                            constraints: const BoxConstraints(),
                          ),
                      ],
                    ),
                  ),

                  // Model selection
                  Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      Row(
                        mainAxisAlignment: MainAxisAlignment.spaceBetween,
                        children: [_buildLabel("Model Architecture")],
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
                                  (m) => DropdownMenuItem(
                                    value: m,
                                    child: Text(m),
                                  ),
                                )
                                .toList(),
                            onChanged: isProcessing
                                ? null
                                : (v) => setState(() => selectedModel = v!),
                          ),
                        ),
                      ),

                      aboutModel(selectedModel),

                      ClipRect(
                        child: AnimatedSize(
                          duration: const Duration(milliseconds: 300),
                          curve: Curves.easeOutCubic,
                          alignment: Alignment.topCenter,
                          child: _modelInfoExpanded
                              ? _showModelInfo(selectedModel)
                              : const SizedBox(width: double.infinity),
                        ),
                      ),
                    ],
                  ),

                  const SizedBox(height: 20),

                  // Slider "scale factor"
                  Row(
                    mainAxisAlignment: MainAxisAlignment.spaceBetween,
                    children: [
                      _buildLabel("Scale Factor"),
                      Text(
                        upscaleFactor.toInt() == 8
                            ? "x${upscaleFactor.toInt()} (experimental)"
                            : "x${upscaleFactor.toInt()}",
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
                    divisions: 2,
                    label: upscaleFactor == 2
                        ? "x2"
                        : upscaleFactor == 4
                        ? "x4"
                        : "x8 (experimental)",
                    activeColor: Colors.blueAccent,
                    onChanged: isProcessing
                        ? null
                        : (v) {
                            double newValue;
                            if (v < 3) {
                              newValue = 2;
                            } else if (v < 6) {
                              newValue = 4;
                            } else {
                              newValue = 8;
                            }
                            setState(() => upscaleFactor = newValue);
                          },
                  ),

                  const SizedBox(height: 24),

                  // "Run Button"
                  SizedBox(
                    width: double.infinity,
                    height: 50,
                    child: ElevatedButton(
                      onPressed: isProcessing
                          ? null
                          : () =>
                                widget.onUpscale(selectedModel, upscaleFactor),
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

                  const SizedBox(height: 20),

                  // Show metrics
                  if (widget.activeRun != null && !isProcessing) ...[
                    _buildHeader("CURRENT RUN METRICS", 11),
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
              height: 275,
              decoration: const BoxDecoration(
                color: Color(0xFF1E1E1E),
                border: Border(top: BorderSide(color: Colors.white10)),
              ),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Padding(
                    padding: const EdgeInsets.all(16.0),
                    child: _buildHeader("RUN HISTORY", 12),
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
      ),
    );
  }

  /// Builds a section header label used to separate panel sections
  Widget _buildHeader(String text, double fontSize) {
    return Text(
      text,
      style: TextStyle(
        color: Colors.white54,
        fontSize: fontSize,
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
