import 'package:flutter/material.dart';

class RightPanel extends StatefulWidget {
  final bool imageLoaded;
  final Future<void> Function() onUpscale;
  final bool isProcessing;

  const RightPanel({
    super.key,
    required this.imageLoaded,
    required this.onUpscale,
    required this.isProcessing,
  });

  @override
  State<RightPanel> createState() => _RightPanelState();
}

class _RightPanelState extends State<RightPanel> {
  double upscaleFactor = 4;
  double noiseReduction = 0.5;
  String selectedModel = 'ESPCN';
  final List<String> models = ['ESPCN', 'SRRN', 'MODELO_X'];

  @override
  Widget build(BuildContext context) {
    double panelWidth = 250;

    return AnimatedContainer(
      duration: const Duration(milliseconds: 300),
      width: widget.imageLoaded ? panelWidth : 0,
      color: Colors.grey[900],
      child: widget.imageLoaded
          ? Padding(
              padding: const EdgeInsets.all(16.0),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  const Text(
                    'Model Parameters',
                    style: TextStyle(
                      color: Colors.white,
                      fontSize: 18,
                      fontWeight: FontWeight.bold,
                    ),
                  ),
                  const SizedBox(height: 16),
                  const Text('Model', style: TextStyle(color: Colors.white70)),
                  DropdownButton<String>(
                    value: selectedModel,
                    dropdownColor: Colors.grey[800],
                    style: const TextStyle(color: Colors.white),
                    items: models
                        .map(
                          (model) => DropdownMenuItem(
                            value: model,
                            child: Text(model),
                          ),
                        )
                        .toList(),
                    onChanged: widget.isProcessing
                        ? null
                        : (value) {
                            setState(() {
                              selectedModel = value!;
                            });
                          },
                  ),
                  const SizedBox(height: 16),
                  const Text(
                    'Upscale Factor',
                    style: TextStyle(color: Colors.white70),
                  ),
                  Slider(
                    value: upscaleFactor,
                    min: 2,
                    max: 8,
                    divisions: 3,
                    label: '${upscaleFactor.toInt()}x',
                    onChanged: widget.isProcessing
                        ? null
                        : (value) {
                            setState(() {
                              upscaleFactor = value;
                            });
                          },
                  ),
                  const SizedBox(height: 16),
                  Center(
                    child: widget.isProcessing
                        ? const CircularProgressIndicator(
                            color: Colors.greenAccent,
                          )
                        : ElevatedButton(
                            onPressed: () async {
                              await widget.onUpscale();
                            },
                            child: const Text('Upscale'),
                          ),
                  ),
                ],
              ),
            )
          : const SizedBox.shrink(),
    );
  }
}
