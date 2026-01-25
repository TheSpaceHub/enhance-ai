import 'dart:typed_data';
import 'package:flutter/material.dart';
import 'package:file_picker/file_picker.dart';
import '../../services/api_service.dart';
import '../models/sr_experiment.dart';
import '../widgets/left_nav.dart';
import '../widgets/main_workspace.dart';
import '../widgets/right_panel.dart';

class HomePage extends StatefulWidget {
  const HomePage({super.key});

  @override
  State<HomePage> createState() => _HomePageState();
}

class _HomePageState extends State<HomePage> {
  final List<SRProject> projects = [];
  SRProject? selectedProject;
  SRRun? selectedRun;

  // Loading Screen
  void resetToUploadScreen() {
    setState(() {
      selectedProject = null;
      selectedRun = null;
    });
  }

  // ACCIÓN 2: Botón central -> Abre el selector de archivos y crea el proyecto
  Future<void> pickAndCreateProject() async {
    print("Intentando abrir selector de archivos...");
    try {
      FilePickerResult? result = await FilePicker.platform.pickFiles(
        type: FileType.image,
        withData: true, // OBLIGATORIO para Web
      );

      if (result != null) {
        // En Web usamos 'bytes', no 'path'
        Uint8List? fileBytes = result.files.first.bytes;
        String fileName = result.files.first.name;

        if (fileBytes == null) {
          print("ERROR: No se pudieron leer los bytes (es null).");
          return;
        }

        print("Archivo seleccionado: $fileName");

        final newId = DateTime.now().millisecondsSinceEpoch.toString();

        final newProject = SRProject(
          id: newId,
          name: fileName,
          timestamp: DateTime.now(),
          originalImage: Image.memory(fileBytes), // Visualización
          rawBytes: fileBytes, // Datos para la API
          runs: [],
        );

        setState(() {
          projects.insert(0, newProject);
          selectedProject = newProject;
        });
      } else {
        print("El usuario canceló la selección.");
      }
    } catch (e) {
      print("CRASH al seleccionar archivo: $e");
    }
  }

  void selectProject(SRProject project) {
    setState(() {
      selectedProject = project;
      selectedRun = project.runs.isNotEmpty ? project.runs.first : null;
    });
  }

  void selectRun(SRRun run) {
    setState(() {
      selectedRun = run;
    });
  }

  Future<void> runUpscale(String model, double factor) async {
    if (selectedProject == null) return;

    final Uint8List originalBytes = selectedProject!.rawBytes;
    final runId = DateTime.now().millisecondsSinceEpoch.toString();

    final processingRun = SRRun(
      id: runId,
      modelName: model,
      upscaleFactor: factor,
      isProcessing: true,
    );

    setState(() {
      selectedProject = selectedProject!.addRun(processingRun);
      // Actualizamos la lista global para que se guarde el cambio
      final idx = projects.indexWhere((p) => p.id == selectedProject!.id);
      if (idx != -1) projects[idx] = selectedProject!;
      selectedRun = processingRun;
    });

    SRRun resultRun;
    try {
      resultRun = await ApiService.upscaleImage(
        imageBytes: originalBytes,
        modelName: model,
        factor: factor,
      );
    } catch (e) {
      print("Error Upscaling: $e");
      resultRun = processingRun.copyWith(
        isProcessing: false,
        metrics: {'Error': 'Failed', 'Msg': 'Server Error'},
      );
    }

    if (!mounted) return;

    setState(() {
      final currentProj = projects.firstWhere(
        (p) => p.id == selectedProject!.id,
      );
      final updatedRuns = currentProj.runs
          .map((r) => r.isProcessing ? resultRun : r)
          .toList();

      final finalProject = SRProject(
        id: currentProj.id,
        name: currentProj.name,
        timestamp: currentProj.timestamp,
        originalImage: currentProj.originalImage,
        rawBytes: currentProj.rawBytes,
        runs: updatedRuns,
      );

      final pIndex = projects.indexWhere((p) => p.id == finalProject.id);
      projects[pIndex] = finalProject;
      selectedProject = finalProject;
      selectedRun = resultRun;
    });
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: Row(
        children: [
          LeftNav(
            projects: projects,
            selectedId: selectedProject?.id,
            onSelect: selectProject,
            onNewProject: resetToUploadScreen, // <--- Botón Sidebar
          ),
          Expanded(
            child: MainWorkspace(
              originalImage: selectedProject?.originalImage,
              activeRun: selectedRun,
              onUpload: pickAndCreateProject, // <--- Botón Central
            ),
          ),
          RightPanel(
            hasProject: selectedProject != null,
            runsHistory: selectedProject?.runs ?? [],
            activeRun: selectedRun,
            onRunSelect: selectRun,
            onUpscale: runUpscale,
          ),
        ],
      ),
    );
  }
}
