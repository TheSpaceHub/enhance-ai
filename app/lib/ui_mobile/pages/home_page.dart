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

  bool forcedCloseLeftPanel = false;
  bool forcedCloseRightPanel = false;

  @override
  Widget build(BuildContext context) {
    double width = MediaQuery.of(context).size.width;
    // Auto-collapse if the screen is too small
    WidgetsBinding.instance.addPostFrameCallback((_) {
      if (width < 1100 && _controller.showLeftPanel && !forcedCloseLeftPanel) {
        _controller.toggleLeftPanel();
        forcedCloseLeftPanel = true;
      }

      if (width < 800 &&
          !forcedCloseRightPanel &&
          _controller.showRightPanel &&
          _controller.currentProject != null) {
        _controller.toggleRightPanel();
        forcedCloseRightPanel = true;
      }
      if (width >= 1100 && forcedCloseLeftPanel) forcedCloseLeftPanel = false;
      if (width >= 800 && forcedCloseRightPanel) forcedCloseRightPanel = false;
    });

    return ListenableBuilder(
      listenable: _controller,
      builder: (context, child) {
        bool hasProject = _controller.currentProject != null;
        if (_controller.errorMessage != null) {
          debugPrint(_controller.errorMessage);
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
          body: Stack(
            children: [
              // Central Workspace: image preview and results
              // Takes up the entire screen in the background
              Positioned.fill(
                child: MainWorkspace(
                  originalImageBytes: _controller.currentProject?.originalBytes,
                  activeRun: _controller.activeRun,
                  pinnedRun: _controller.pinnedRun,
                  onUpload: _controller.createNewProject,
                  showLeftPanel: _controller.showLeftPanel,
                  onCloseLeftPanel: _controller.toggleLeftPanel,
                  showRightPanel: _controller.showRightPanel,
                  onCloseRightPanel: _controller.toggleRightPanel,
                ),
              ),

              // Overlay block to handle touches if needed (optional)
              if (_controller.showLeftPanel || (_controller.showRightPanel && hasProject))
                Positioned.fill(
                  child: GestureDetector(
                    onTap: () {
                      if (_controller.showLeftPanel) _controller.toggleLeftPanel();
                      if (_controller.showRightPanel) _controller.toggleRightPanel();
                    },
                    child: Container(color: Colors.black45),
                  ),
                ),

              // Left Tab: project history
              AnimatedPositioned(
                duration: const Duration(milliseconds: 300),
                curve: Curves.easeInOut,
                left: _controller.showLeftPanel ? 0 : -260,
                top: 0,
                bottom: 0,
                width: 260,
                child: LeftPanel(
                  projects: _controller.projects,
                  selectedId: _controller.selectedProjectId,
                  onSelect: (p) => _controller.selectProject(p.id),
                  onNewProject: _controller.createNewProject,
                  onDeleteProject: (p) => _controller.deleteProject(p.id),
                  onClose: _controller.toggleLeftPanel,
                  hasProject: hasProject,
                ),
              ),

              // Right Tab: run configuration
              AnimatedPositioned(
                duration: const Duration(milliseconds: 300),
                curve: Curves.easeInOut,
                right: (_controller.showRightPanel && hasProject) ? 0 : -320,
                top: 0,
                bottom: 0,
                width: 320,
                child: RightPanel(
                  hasProject: _controller.currentProject != null,
                  runsHistory: _controller.currentProject?.runs ?? [],
                  activeRun: _controller.activeRun,
                  pinnedRunId: _controller.pinnedRunId,
                  selectedDevice: _controller.selectedDevice,
                  onRunSelect: (run) => _controller.selectRun(run.id),
                  onTogglePin: (id) => _controller.togglePin(id),
                  onUpscale: (m, f) => _controller.runUpscale(m, f),
                  onDeviceChanged: (d) => _controller.setDevice(d),
                  onClose: _controller.toggleRightPanel,
                ),
              ),
            ],
          ),
        );
      },
    );
  }
}
