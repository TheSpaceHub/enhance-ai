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
          // Logo Area
          const Padding(
            padding: EdgeInsets.fromLTRB(20, 30, 20, 10),
            child: Text(
              "EnhanceAI",
              style: TextStyle(
                color: Colors.white,
                fontSize: 20,
                fontWeight: FontWeight.bold,
                letterSpacing: 1,
              ),
            ),
          ),

          // Botón Nuevo Proyecto
          Padding(
            padding: const EdgeInsets.all(16.0),
            child: ElevatedButton.icon(
              onPressed: onNewProject,
              icon: const Icon(Icons.add),
              label: const Text("New Project"),
              style: ElevatedButton.styleFrom(
                backgroundColor: const Color(0xFF333333),
                foregroundColor: Colors.white,
                elevation: 0,
                minimumSize: const Size(double.infinity, 50),
                shape: RoundedRectangleBorder(
                  borderRadius: BorderRadius.circular(8),
                  side: const BorderSide(color: Colors.white12),
                ),
              ),
            ),
          ),

          const Divider(color: Colors.white10),

          const Padding(
            padding: EdgeInsets.symmetric(horizontal: 16, vertical: 12),
            child: Text(
              "HISTORY",
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
            child: projects.isEmpty
                ? const Center(
                    child: Text(
                      "No projects yet",
                      style: TextStyle(color: Colors.white24),
                    ),
                  )
                : ListView.builder(
                    itemCount: projects.length,
                    itemBuilder: (context, index) {
                      final project = projects[index];
                      final isSelected = project.id == selectedId;

                      return Container(
                        margin: const EdgeInsets.symmetric(
                          horizontal: 8,
                          vertical: 2,
                        ),
                        decoration: BoxDecoration(
                          color: isSelected ? const Color(0xFF37373D) : null,
                          borderRadius: BorderRadius.circular(6),
                        ),
                        child: ListTile(
                          contentPadding: const EdgeInsets.symmetric(
                            horizontal: 12,
                            vertical: 4,
                          ),
                          onTap: () => onSelect(project),
                          leading: Container(
                            width: 40,
                            height: 40,
                            decoration: BoxDecoration(
                              borderRadius: BorderRadius.circular(4),
                              border: Border.all(color: Colors.white12),
                            ),
                            child: ClipRRect(
                              borderRadius: BorderRadius.circular(3),
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
                              fontSize: 13,
                              fontWeight: isSelected
                                  ? FontWeight.w600
                                  : FontWeight.normal,
                            ),
                          ),
                          subtitle: Text(
                            "${project.runs.length} enhancements",
                            style: const TextStyle(
                              color: Colors.white30,
                              fontSize: 11,
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
