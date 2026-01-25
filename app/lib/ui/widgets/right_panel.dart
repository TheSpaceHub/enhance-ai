// lib/ui/widgets/right_panel.dart
import 'package:flutter/material.dart';
import '../models/sr_experiment.dart';

class RightPanel extends StatefulWidget {
  final bool hasProject;
  final List<SRRun> runsHistory;
  final SRRun? activeRun;
  final Function(SRRun) onRunSelect;
  final Function(String model, double factor) onUpscale;

  const RightPanel({
    super.key,
    required this.hasProject,
    required this.runsHistory,
    required this.activeRun,
    required this.onRunSelect,
    required this.onUpscale,
  });

  @override
  State<RightPanel> createState() => _RightPanelState();
}

class _RightPanelState extends State<RightPanel> {
  double upscaleFactor = 4;
  String selectedModel = 'ESPCN';
  final List<String> models = ['CNNU', 'ESPCN', 'SRResNet', 'SRGAN'];

  @override
  Widget build(BuildContext context) {
    if (!widget.hasProject) return const SizedBox.shrink();

    final isProcessing = widget.activeRun?.isProcessing ?? false;

    return Container(
      width: 300,
      color: const Color(0xFF252526),
      child: Column(
        children: [
          Expanded(
            child: ListView(
              padding: const EdgeInsets.all(20),
              children: [
                // SECCIÓN 1: CONFIGURACIÓN
                _buildHeader("CONFIGURATION"),
                const SizedBox(height: 16),

                // Model Dropdown
                _buildLabel("Model Architecture"),
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

                const SizedBox(height: 24),

                // Slider Factor
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

                // Botón Upscale
                SizedBox(
                  width: double.infinity,
                  height: 45,
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
                        : const Text('RUN UPSCALE'),
                  ),
                ),

                const SizedBox(height: 40),

                // SECCIÓN 2: MÉTRICAS (Solo si hay resultado)
                if (widget.activeRun != null &&
                    !isProcessing &&
                    widget.activeRun!.metrics.isNotEmpty) ...[
                  _buildHeader("CURRENT RUN METRICS"),
                  const SizedBox(height: 16),
                  ...widget.activeRun!.metrics.entries.map(
                    (e) => _buildMetricRow(e.key, e.value),
                  ),
                ],
              ],
            ),
          ),

          // SECCIÓN 3: MINILISTADO DE HISTORIAL (Al fondo del panel)
          Container(
            height: 200,
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

                      return ListTile(
                        dense: true,
                        selected: isActive,
                        selectedTileColor: Colors.white.withOpacity(0.05),
                        onTap: () => widget.onRunSelect(run),
                        leading: Icon(
                          run.isProcessing
                              ? Icons.hourglass_empty
                              : Icons.check_circle,
                          color: run.isProcessing
                              ? Colors.orange
                              : Colors.greenAccent,
                          size: 16,
                        ),
                        title: Text(
                          "${run.modelName} (x${run.upscaleFactor.toInt()})",
                          style: TextStyle(
                            color: isActive ? Colors.white : Colors.white60,
                            fontSize: 13,
                          ),
                        ),
                        trailing: run.metrics.containsKey('MAE')
                            ? Text(
                                "MAE: ${run.metrics['MAE']}",
                                style: const TextStyle(
                                  fontSize: 10,
                                  color: Colors.white30,
                                ),
                              )
                            : null,
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

  Widget _buildHeader(String text) {
    return Text(
      text,
      style: const TextStyle(
        color: Colors.white54,
        fontSize: 11,
        fontWeight: FontWeight.bold,
      ),
    );
  }

  Widget _buildLabel(String text) {
    return Text(
      text,
      style: const TextStyle(color: Colors.white, fontSize: 13),
    );
  }

  Widget _buildMetricRow(String key, dynamic value) {
    return Padding(
      padding: const EdgeInsets.only(bottom: 8.0),
      child: Row(
        mainAxisAlignment: MainAxisAlignment.spaceBetween,
        children: [
          Text(key, style: const TextStyle(color: Colors.white60)),
          Text(
            value.toString(),
            style: const TextStyle(
              color: Colors.white,
              fontWeight: FontWeight.bold,
              fontFamily: 'Monospace',
            ),
          ),
        ],
      ),
    );
  }
}
