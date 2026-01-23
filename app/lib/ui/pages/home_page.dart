// lib/ui/pages/home_page.dart
import 'package:flutter/material.dart';
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
  // Lista Maestra de Proyectos
  final List<SRProject> projects = [];

  // Estado Actual
  SRProject? selectedProject;
  SRRun? selectedRun;

  // 1. Crear Nuevo Proyecto (Upload)
  void createNewProject() {
    final newId = DateTime.now().millisecondsSinceEpoch.toString();
    final newProject = SRProject(
      id: newId,
      name: "Image_${newId.substring(newId.length - 4)}",
      timestamp: DateTime.now(),
      originalImage: Image.asset('assets/lr_sample.png'), // Tu imagen asset
      runs: [],
    );

    setState(() {
      projects.insert(0, newProject);
      selectedProject = newProject;
      selectedRun = null;
    });
  }

  // 2. Seleccionar Proyecto (Click en Sidebar)
  void selectProject(SRProject project) {
    setState(() {
      selectedProject = project;
      // Selecciona el último run automáticamente si existe
      selectedRun = project.runs.isNotEmpty ? project.runs.first : null;
    });
  }

  // 3. Seleccionar Run (Click en historial del Panel Derecho)
  void selectRun(SRRun run) {
    setState(() {
      selectedRun = run;
    });
  }

  // 4. Ejecutar Upscale (Click en botón RUN)
  Future<void> runUpscale(String model, double factor) async {
    if (selectedProject == null) return;

    final runId = DateTime.now().millisecondsSinceEpoch.toString();

    // A. Crear Run en estado "Cargando"
    final newRun = SRRun(
      id: runId,
      modelName: model,
      upscaleFactor: factor,
      isProcessing: true,
    );

    setState(() {
      // Añadimos el run al proyecto actual
      final updatedProject = selectedProject!.addRun(newRun);

      // Actualizamos la lista global
      final index = projects.indexWhere((p) => p.id == updatedProject.id);
      projects[index] = updatedProject;

      // Actualizamos selección
      selectedProject = updatedProject;
      selectedRun = newRun;
    });

    // B. Simulación Backend
    await Future.delayed(const Duration(seconds: 2));

    // C. Resultado Final
    if (!mounted) return;

    setState(() {
      // Recuperamos el proyecto actual (por seguridad)
      final currentProj = projects.firstWhere(
        (p) => p.id == selectedProject!.id,
      );

      // Creamos el run finalizado
      final finishedRun = newRun.copyWith(
        isProcessing: false,
        resultImage: Image.asset(
          'assets/sr_sample.png',
        ), // Tu imagen resultado asset
        metrics: {'MAE': 0.045, 'PSNR': '28.5dB', 'Time': '1.2s'},
      );

      // Reemplazamos el run viejo por el nuevo en la lista
      final updatedRuns = currentProj.runs
          .map((r) => r.id == runId ? finishedRun : r)
          .toList();

      final finalProject = SRProject(
        id: currentProj.id,
        name: currentProj.name,
        timestamp: currentProj.timestamp,
        originalImage: currentProj.originalImage,
        runs: updatedRuns,
      );

      final pIndex = projects.indexWhere((p) => p.id == finalProject.id);
      projects[pIndex] = finalProject;
      selectedProject = finalProject;
      selectedRun = finishedRun;
    });
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: Row(
        children: [
          // Sidebar
          LeftNav(
            projects: projects,
            selectedId: selectedProject?.id,
            onSelect: selectProject,
            onNewProject: createNewProject,
          ),

          // Workspace
          Expanded(
            child: MainWorkspace(
              originalImage: selectedProject?.originalImage,
              activeRun: selectedRun,
              onUpload: createNewProject,
            ),
          ),

          // Panel Derecho
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
