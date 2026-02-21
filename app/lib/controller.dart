import 'package:app/storage_manager.dart';
import 'package:flutter/material.dart';
import 'package:file_picker/file_picker.dart';
import 'ui/models/sr_experiment.dart';
import 'services/upscale_engine.dart';

/// State manager that coordinates projects, runs, and UI interactions
class Controller extends ChangeNotifier {
  final List<SRProject> _projects;
  final StorageManager storageManager;
  bool _showRightPanel = true;
  bool _showLeftPanel = true;
  String? _selectedProjectId;
  String? _activeRunId; // Active Run (Right Slot)
  String? _pinnedRunId; // Pinned Run (Left Slot)
  String _selectedDevice = 'GPU'; // 'CPU' or 'GPU'
  String? _errorMessage;
  bool get showRightPanel => _showRightPanel;
  bool get showLeftPanel => _showLeftPanel;
  List<SRProject> get projects => _projects;
  String? get selectedProjectId => _selectedProjectId;
  String get selectedDevice => _selectedDevice;
  String? get activeRunId => _activeRunId;
  String? get pinnedRunId => _pinnedRunId;

  //initialize storage manager right after creating the projects in memory
  Controller({required VoidCallback stateSetter})
    : this._internal(stateSetter, []);
  Controller._internal(VoidCallback stateSetter, List<SRProject> list)
    : _projects = list,
      storageManager = StorageManager(list, stateSetter);

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

  /// Changes the "show right panel" state
  void toggleRightPanel() {
    _showRightPanel = !_showRightPanel;
    notifyListeners();
  }

  void closeRightPanel() {
    if (!_showRightPanel) return;
    _showRightPanel = false;
    notifyListeners();
  }

  /// Changes the "show left panel" state
  void toggleLeftPanel() {
    _showLeftPanel = !_showLeftPanel;
    notifyListeners();
  }

  void closeLeftPanel() {
    if (!_showLeftPanel) return;
    _showLeftPanel = false;
    notifyListeners();
  }

  /// Creates a new project by importing an image from disk
  Future<void> createNewProject() async {
    FilePickerResult? result = await FilePicker.platform.pickFiles(
      type: FileType.image,
      withData: true,
    );

    if (result != null && result.files.single.bytes != null) {
      //set bytess and name
      final bytes = result.files.single.bytes!;
      final name = result.files.single.name;

      //create SRProject object and insert
      final newProject = SRProject(
        id: DateTime.now().millisecondsSinceEpoch.toString(),
        name: name,
        timestamp: DateTime.now(),
        originalBytes: bytes,
      );

      _projects.insert(0, newProject);

      //store image file
      await storageManager.addProjectToStorage(newProject);

      //select project
      selectProject(newProject.id);
    }
  }

  Future<void> deleteProject(String id) async {
    //delete from state and storage

    //state
    final index = _projects.indexWhere((p) => p.id == id);
    _projects.removeAt(index);
    //storage
    storageManager.deleteProjectFromStorage(id);
    notifyListeners();
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
    _projects[projIndex].addRun(tempRun);
    _activeRunId = tempRunId;
    _errorMessage = null;
    notifyListeners();

    try {
      final finishedRun = await ApiService.upscaleImage(
        imageBytes: currentProject!.originalBytes,
        modelName: model,
        factor: factor.toInt(),
        device: _selectedDevice,
      );

      // Replace temporary run with final result
      final pIndex = _projects.indexWhere((p) => p.id == _selectedProjectId);
      if (pIndex != -1) {
        _projects[pIndex].updateRun(tempRunId, finishedRun);

        //store it
        storageManager.addRunToStorage(finishedRun, _projects[pIndex]);

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
