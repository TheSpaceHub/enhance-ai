import 'package:flutter/material.dart';
import '../../controller.dart';
import '../widgets/left_panel.dart';
import '../widgets/right_panel.dart';
import '../widgets/main_workspace.dart';

class HomePage extends StatefulWidget {
  const HomePage({super.key});

  @override
  State<HomePage> createState() => _HomePageState();
}

/// Manages responsive layout and connects UI panels to the Controller
class _HomePageState extends State<HomePage> {
  final Controller _controller = Controller();
  bool _showLeftNav = true;
  bool _showRightPanel = true;

  @override
  Widget build(BuildContext context) {
    double width = MediaQuery.of(context).size.width;

    // Auto-collapse if the screen is too small
    if (width < 1100 && _showLeftNav) _showLeftNav = false;
    if (width < 800 && _showRightPanel) _showRightPanel = false;

    return ListenableBuilder(
      listenable: _controller,
      builder: (context, child) {
        return Scaffold(
          body: Row(
            children: [
              // Left Tab: project history
              AnimatedContainer(
                duration: const Duration(milliseconds: 300),
                width: _showLeftNav ? 260 : 0,
                child: _showLeftNav
                    ? LeftPanel(
                        projects: _controller.projects,
                        selectedId: _controller.selectedProjectId,
                        onSelect: (p) => _controller.selectProject(p.id),
                        onNewProject: _controller.createNewProject,
                        onClose: () => setState(() => _showLeftNav = false),
                      )
                    : const SizedBox.shrink(),
              ),

              // Central Workspace: image preview and results
              Expanded(
                child: Stack(
                  children: [
                    MainWorkspace(
                      originalImageBytes:
                          _controller.currentProject?.originalBytes,
                      activeRun: _controller.activeRun,
                      pinnedRun: _controller.pinnedRun,
                      onUpload: _controller.createNewProject,
                    ),
                    // Buttons to open tabs if closed
                    Positioned(
                      top: 85,
                      left: 20,
                      child: !_showLeftNav
                          ? IconButton(
                              icon: const Icon(Icons.menu_open),
                              tooltip: "Open History",
                              onPressed: () =>
                                  setState(() => _showLeftNav = true),
                            )
                          : const SizedBox.shrink(),
                    ),
                    Positioned(
                      top: 85,
                      right: 20,
                      child: !_showRightPanel
                          ? IconButton(
                              icon: const Icon(Icons.settings),
                              tooltip: "Open Enhance Config",
                              onPressed: () =>
                                  setState(() => _showRightPanel = true),
                            )
                          : const SizedBox.shrink(),
                    ),
                  ],
                ),
              ),

              // Right Tab: run configuration
              AnimatedContainer(
                duration: const Duration(milliseconds: 300),
                width: _showRightPanel ? 320 : 0,
                child: _showRightPanel
                    ? RightPanel(
                        hasProject: _controller.currentProject != null,
                        runsHistory: _controller.currentProject?.runs ?? [],
                        activeRun: _controller.activeRun,
                        pinnedRunId: _controller.pinnedRunId,
                        selectedDevice: _controller.selectedDevice,
                        onRunSelect: (run) => _controller.selectRun(run.id),
                        onTogglePin: (id) => _controller.togglePin(id),
                        onUpscale: (m, f) => _controller.runUpscale(m, f),
                        onDeviceChanged: (d) => _controller.setDevice(d),
                        onClose: () => setState(() => _showRightPanel = false),
                      )
                    : const SizedBox.shrink(),
              ),
            ],
          ),
        );
      },
    );
  }
}
