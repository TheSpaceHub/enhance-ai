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
  late final Controller _controller = Controller(
    stateSetter: () => setState(() {}),
  );
  bool _showLeftPanel = false;
  bool _showRightPanel = false;

  bool forcedCloseLeftPanel = false;
  bool forcedCloseRightPanel = false;

  Widget leftButton() {
    return IconButton(
      icon: const Icon(Icons.menu_open),
      tooltip: !_showLeftPanel ? "Open History" : "Hide History",
      onPressed: () => setState(() => _showLeftPanel = !_showLeftPanel),
    );
  }

  Widget rightButton() {
    return IconButton(
      icon: const Icon(Icons.settings),
      tooltip: !_showRightPanel ? "Open Enhance Config" : "Hide Enhance Config",
      onPressed: () => setState(() => _showRightPanel = !_showRightPanel),
    );
  }

  @override
  Widget build(BuildContext context) {
    double width = MediaQuery.of(context).size.width;

    bool hasProject = _controller.currentProject != null;

    // Auto-collapse if the screen is too small
    if (width < 1100 && !forcedCloseLeftPanel && _showLeftPanel) {
      _showLeftPanel = false;
      forcedCloseLeftPanel = true;
    }
    if (width < 800 &&
        !forcedCloseRightPanel &&
        _showRightPanel &&
        hasProject) {
      _showRightPanel = false;
      forcedCloseRightPanel = true;
    }
    if (width >= 1100 && forcedCloseLeftPanel) forcedCloseLeftPanel = false;
    if (width >= 800 && forcedCloseRightPanel) forcedCloseRightPanel = false;

    return ListenableBuilder(
      listenable: _controller,
      builder: (context, child) {
        if (_controller.currentProject == null) hasProject = false;
        if (_controller.errorMessage != null) {
          WidgetsBinding.instance.addPostFrameCallback((_) {
            ScaffoldMessenger.of(context).showSnackBar(
              SnackBar(
                content: Text(_controller.errorMessage!),
                backgroundColor: Colors.red,
                duration: const Duration(seconds: 20),
              ),
            );
            _controller.clearError();
          });
        }
        return Scaffold(
          body: Row(
            children: [
              // Left Tab: project history
              AnimatedContainer(
                duration: const Duration(milliseconds: 300),
                width: _showLeftPanel ? 260 : 0,
                child: ClipRect(
                  child: OverflowBox(
                    minWidth: 260,
                    maxWidth: 260,
                    alignment: Alignment.centerLeft,
                    child: _showLeftPanel
                        ? LeftPanel(
                            projects: _controller.projects,
                            selectedId: _controller.selectedProjectId,
                            onSelect: (p) => _controller.selectProject(p.id),
                            onNewProject: _controller.createNewProject,

                            onDeleteProject: (p) =>
                                _controller.deleteProject(p.id),
                            onClose: () =>
                                setState(() => _showLeftPanel = false),
                          )
                        : const SizedBox.shrink(),
                  ),
                ),
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
                      leftButton: leftButton(),
                      rightButton: rightButton(),
                    ),
                    (!hasProject && !_showLeftPanel)
                        ? Positioned(top: 20, left: 20, child: leftButton())
                        : const SizedBox.shrink(),
                  ],
                ),
              ),

              // Right Tab: run configuration
              AnimatedContainer(
                duration: const Duration(milliseconds: 300),
                width: (_showRightPanel && hasProject) ? 320 : 0,
                child: ClipRect(
                  child: OverflowBox(
                    minWidth: 320,
                    maxWidth: 320,
                    alignment: Alignment.centerLeft,
                    child: (_showRightPanel && hasProject)
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
                            onClose: () =>
                                setState(() => _showRightPanel = false),
                          )
                        : const SizedBox.shrink(),
                  ),
                ),
              ),
            ],
          ),
        );
      },
    );
  }
}
