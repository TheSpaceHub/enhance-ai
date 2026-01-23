// lib/ui/widgets/left_nav.dart
import 'package:flutter/material.dart';
import '../models/sr_experiment.dart';

class LeftNav extends StatelessWidget {
  final List<SRProject> projects;
  final String? selectedId;
  final Function(SRProject) onSelect;
  final VoidCallback onNewProject;

  const LeftNav({
    super.key,
    required this.projects,
    required this.selectedId,
    required this.onSelect,
    required this.onNewProject,
  });

  @override
  Widget build(BuildContext context) {
    return Container(
      width: 260,
      color: const Color(0xFF1E1E1E),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          // Botón Nuevo Proyecto
          Padding(
            padding: const EdgeInsets.all(16.0),
            child: ElevatedButton.icon(
              onPressed: onNewProject,
              icon: const Icon(Icons.add_photo_alternate),
              label: const Text("New Project"),
              style: ElevatedButton.styleFrom(
                backgroundColor: Colors.blueAccent,
                foregroundColor: Colors.white,
                minimumSize: const Size(double.infinity, 50),
              ),
            ),
          ),
          const Divider(color: Colors.white12),
          const Padding(
            padding: EdgeInsets.symmetric(horizontal: 16, vertical: 8),
            child: Text(
              "PROJECTS",
              style: TextStyle(
                color: Colors.white38,
                fontSize: 11,
                fontWeight: FontWeight.bold,
                letterSpacing: 1.5,
              ),
            ),
          ),
          // Lista de Proyectos
          Expanded(
            child: ListView.builder(
              itemCount: projects.length,
              itemBuilder: (context, index) {
                final project = projects[index];
                final isSelected = project.id == selectedId;

                return Container(
                  color: isSelected ? Colors.white.withOpacity(0.05) : null,
                  child: ListTile(
                    contentPadding: const EdgeInsets.symmetric(
                      horizontal: 16,
                      vertical: 8,
                    ),
                    onTap: () => onSelect(project),
                    leading: Container(
                      width: 44,
                      height: 44,
                      decoration: BoxDecoration(
                        borderRadius: BorderRadius.circular(6),
                        border: Border.all(color: Colors.white12),
                      ),
                      child: ClipRRect(
                        borderRadius: BorderRadius.circular(5),
                        child: FittedBox(
                          fit: BoxFit.cover,
                          child: project.originalImage,
                        ),
                      ),
                    ),
                    title: Text(
                      project.name,
                      maxLines: 1,
                      overflow: TextOverflow.ellipsis,
                      style: TextStyle(
                        color: isSelected ? Colors.white : Colors.white70,
                        fontWeight: isSelected
                            ? FontWeight.w600
                            : FontWeight.normal,
                      ),
                    ),
                    subtitle: Text(
                      "${project.runs.length} runs",
                      style: const TextStyle(
                        color: Colors.white30,
                        fontSize: 12,
                      ),
                    ),
                  ),
                );
              },
            ),
          ),
        ],
      ),
    );
  }
}
