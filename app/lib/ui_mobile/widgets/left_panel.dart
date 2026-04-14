import 'dart:ui';
import 'package:flutter/material.dart';
import '../../ui/models/sr_experiment.dart';

/// Left navigation panel displaying project history and creation controls.
class LeftPanel extends StatelessWidget {
  final List<SRProject> projects;
  final String? selectedId;
  final Function(SRProject) onSelect;
  final VoidCallback onNewProject;
  final Function(SRProject) onDeleteProject;
  final VoidCallback onClose;
  final bool hasProject;

  const LeftPanel({
    super.key,
    required this.projects,
    required this.selectedId,
    required this.onSelect,
    required this.onNewProject,
    required this.onDeleteProject,
    required this.onClose,
    required this.hasProject,
  });

  @override
  Widget build(BuildContext context) {
    return ClipRRect(
      child: BackdropFilter(
        filter: ImageFilter.blur(sigmaX: 20.0, sigmaY: 20.0),
        child: Container(
          width: 260,
          color: Colors.black.withOpacity(0.4),
          child: SafeArea(
            child: Column(
        children: [
          // Header Logo
          Padding(
            padding: EdgeInsets.fromLTRB(20, 30, 20, 20),
            child: Row(
              mainAxisAlignment: MainAxisAlignment.spaceBetween,
              children: [
                Text(
                  "Enhance AI",
                  style: TextStyle(
                    color: Colors.white,
                    fontSize: 20,
                    fontWeight: FontWeight.bold,
                  ),
                ),
                hasProject
                    ? IconButton(
                        icon: const Icon(Icons.close),
                        color: Colors.white54,
                        tooltip: "Close panel",
                        onPressed: onClose,
                      )
                    : SizedBox.shrink(),
              ],
            ),
          ),

          // New project action
          Padding(
            padding: const EdgeInsets.symmetric(horizontal: 16),
            child: InkWell(
              onTap: onNewProject,
              borderRadius: BorderRadius.circular(12),
              child: Container(
                height: 48,
                decoration: BoxDecoration(
                  borderRadius: BorderRadius.circular(12),
                  color: Colors.blueAccent,
                  boxShadow: [
                    BoxShadow(
                      color: Colors.blueAccent.withOpacity(0.35),
                      blurRadius: 10,
                      offset: const Offset(0, 4),
                    ),
                  ],
                ),
                child: const Row(
                  mainAxisAlignment: MainAxisAlignment.center,
                  children: [
                    Icon(Icons.add, color: Colors.white, size: 22),
                    SizedBox(width: 8),
                    Text(
                      "New Project",
                      style: TextStyle(
                        color: Colors.white,
                        fontWeight: FontWeight.bold,
                        fontSize: 15,
                        letterSpacing: 0.5,
                      ),
                    ),
                  ],
                ),
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
      ),
    )));
  }
}
