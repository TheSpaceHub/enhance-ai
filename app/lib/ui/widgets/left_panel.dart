import 'package:flutter/material.dart';
import '../models/sr_experiment.dart';

/// Left navigation panel displaying project history and creation controls.
class LeftPanel extends StatelessWidget {
  final List<SRProject> projects;
  final String? selectedId;
  final Function(SRProject) onSelect;
  final VoidCallback onNewProject;
  final Function(SRProject) onDeleteProject;
  final VoidCallback onClose;

  const LeftPanel({
    super.key,
    required this.projects,
    required this.selectedId,
    required this.onSelect,
    required this.onNewProject,
    required this.onDeleteProject,
    required this.onClose,
  });

  @override
  Widget build(BuildContext context) {
    return Container(
      width: 260,
      color: const Color(0xFF1E1E1E),
      child: Column(
        children: [
          // Header Logo
          Padding(
            padding: EdgeInsets.fromLTRB(20, 30, 20, 20),
            child: Text(
              "Enhance AI",
              style: TextStyle(
                color: Colors.white,
                fontSize: 20,
                fontWeight: FontWeight.bold,
              ),
            ),
          ),

          // New project action
          Padding(
            padding: const EdgeInsets.symmetric(horizontal: 16),
            child: ElevatedButton.icon(
              onPressed: onNewProject,
              icon: const Icon(Icons.add),
              label: const Text("New Project"),
              style: ElevatedButton.styleFrom(
                backgroundColor: const Color(0xFF333333),
                minimumSize: const Size(double.infinity, 45),
              ),
            ),
          ),

          const SizedBox(height: 20),
          const Divider(color: Colors.white10),

          // Scrollable project list
          Expanded(
            child: ListView.builder(
              itemCount: projects.length,
              itemBuilder: (context, index) {
                final project = projects[index];
                final isSelected = project.id == selectedId;

                return Container(
                  color: isSelected ? const Color(0xFF37373D) : null,
                  child: ListTile(
                    onTap: () => onSelect(project),
                    leading: Container(
                      width: 40,
                      height: 40,
                      decoration: BoxDecoration(
                        border: Border.all(color: Colors.white12),
                        borderRadius: BorderRadius.circular(4),
                      ),
                      child: ClipRRect(
                        borderRadius: BorderRadius.circular(3),
                        child: Image.memory(
                          project.originalBytes,
                          fit: BoxFit.cover,
                          gaplessPlayback: true,
                        ),
                      ),
                    ),
                    title: Text(
                      project.name,
                      style: TextStyle(
                        color: isSelected ? Colors.white : Colors.white70,
                      ),
                    ),
                    subtitle: Text(
                      "${project.runs.length} runs",
                      style: const TextStyle(
                        color: Colors.white30,
                        fontSize: 11,
                      ),
                    ),
                    trailing: IconButton(
                      icon: Icon(Icons.delete),
                      onPressed: () => onDeleteProject(project),
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
