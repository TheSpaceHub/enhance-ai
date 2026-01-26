import 'package:flutter/material.dart';
import 'package:file_picker/file_picker.dart';
import 'ui/models/sr_experiment.dart';
import 'services/api_service.dart';

/// State manager that coordinates projects, runs, and UI interactions
class Controller extends ChangeNotifier {
  final List<SRProject> _projects = [];
  String? _selectedProjectId;
  String? _activeRunId; // Active Run (Right Slot)
  String? _pinnedRunId; // Pinned Run (Left Slot)
  String _selectedDevice = 'GPU'; // 'CPU' or 'GPU'
  String? _errorMessage;
  List<SRProject> get projects => _projects;
  String? get selectedProjectId => _selectedProjectId;
  String get selectedDevice => _selectedDevice;
  String? get activeRunId => _activeRunId;
  String? get pinnedRunId => _pinnedRunId;

  SRProject? get currentProject {
    if (_selectedProjectId == null) return null;
    try {
      return _projects.firstWhere((p) => p.id == _selectedProjectId);
    } catch (_) {
      return null;
    }
  }

  SRRun? get activeRun => _getRunById(_activeRunId);
  SRRun? get pinnedRun => _getRunById(_pinnedRunId);

  SRRun? _getRunById(String? id) {
    if (currentProject == null || id == null) return null;
    try {
      return currentProject!.runs.firstWhere((r) => r.id == id);
    } catch (_) {
      return null;
    }
  }

  String? get errorMessage => _errorMessage;

  /// Clears the current error message and notifies listeners
  void clearError() {
    _errorMessage = null;
    notifyListeners();
  }

  /// Sets the computation device (CPU or GPU) for future runs
  void setDevice(String device) {
    _selectedDevice = device;
    notifyListeners();
  }

  /// Toggles pin state for a run to enable side-by-side comparison
  void togglePin(String runId) {
    if (_pinnedRunId == runId) {
      _pinnedRunId = null;
    } else {
      _pinnedRunId = runId;
    }
    notifyListeners();
  }

  /// Selects a project and initializes run selection state
  void selectProject(String projectId) {
    _selectedProjectId = projectId;
    final proj = currentProject;
    _pinnedRunId = null;
    if (proj != null && proj.runs.isNotEmpty) {
      _activeRunId = proj.runs.first.id;
    } else {
      _activeRunId = null;
    }
    notifyListeners();
  }

  /// Selects a run as the active result
  void selectRun(String runId) {
    _activeRunId = runId;
    notifyListeners();
  }

  /// Creates a new project by importing an image from disk
  Future<void> createNewProject() async {
    FilePickerResult? result = await FilePicker.platform.pickFiles(
      type: FileType.image,
      withData: true,
    );

    if (result != null && result.files.single.bytes != null) {
      final bytes = result.files.single.bytes!;
      final name = result.files.single.name;

      final newProject = SRProject(
        id: DateTime.now().millisecondsSinceEpoch.toString(),
        name: name,
        timestamp: DateTime.now(),
        originalBytes: bytes,
      );

      _projects.insert(0, newProject);
      selectProject(newProject.id);
    }
  }

  /// Runs an image upscaling task with optimistic UI updates and error rollback.
  Future<void> runUpscale(String model, double factor) async {
    if (currentProject == null) return;

    final tempRunId = DateTime.now().millisecondsSinceEpoch.toString();
    final tempRun = SRRun(
      id: tempRunId,
      modelName: model,
      upscaleFactor: factor,
      isProcessing: true,
      device: _selectedDevice,
    );

    // Show run as "processing"
    final projIndex = _projects.indexOf(currentProject!);
    _projects[projIndex] = currentProject!.addRun(tempRun);
    _activeRunId = tempRunId;
    _errorMessage = null;
    notifyListeners();

    try {
      final finishedRun = await ApiService.upscaleImage(
        imageBytes: currentProject!.originalBytes,
        modelName: model,
        factor: factor,
        device: _selectedDevice,
      );

      // Replace temporary run with final result
      final pIndex = _projects.indexWhere((p) => p.id == _selectedProjectId);
      if (pIndex != -1) {
        final proj = _projects[pIndex];
        final updatedRuns = proj.runs
            .map((r) => r.id == tempRunId ? finishedRun : r)
            .toList();

        _projects[pIndex] = SRProject(
          id: proj.id,
          name: proj.name,
          timestamp: proj.timestamp,
          originalBytes: proj.originalBytes,
          runs: updatedRuns,
        );

        _activeRunId = finishedRun.id;
        notifyListeners();
      }
    } catch (e) {
      // Rollback in case of error
      _errorMessage = e.toString().replaceAll("Exception: ", "");
      final pIndex = _projects.indexWhere((p) => p.id == _selectedProjectId);
      if (pIndex != -1) {
        final proj = _projects[pIndex];
        final updatedRuns = proj.runs.where((r) => r.id != tempRunId).toList();

        _projects[pIndex] = SRProject(
          id: proj.id,
          name: proj.name,
          timestamp: proj.timestamp,
          originalBytes: proj.originalBytes,
          runs: updatedRuns,
        );
        _activeRunId = null;
        notifyListeners();
      }
    }
  }
}
