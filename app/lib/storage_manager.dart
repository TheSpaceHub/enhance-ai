import 'dart:io';

import 'package:app/ui/models/sr_experiment.dart';
import 'package:path_provider/path_provider.dart';

class StorageManager {
  /*
  Storage structure:
  - Project 1
    Metadata
    Original image
    Run 1
    Run 2
  - Project 2
    Metadata
    Original image
  */

  late Directory storageDir;
  bool loaded = false;
  bool crashed = true;

  StorageManager(List<SRProject> controllerProjects) {
    initializeStorageManager(controllerProjects);
  }

  Future<void> initializeStorageManager(
    List<SRProject> controllerProjects,
  ) async {
    await setStorageDir();
    await loadProjects(controllerProjects);
    loaded = true;
  }

  Future<void> setStorageDir() async {
    //sets the directory where all data is stored
    try {
      final documentsDir = await getApplicationDocumentsDirectory();
      storageDir = Directory("${documentsDir.path}/EnhanceAI");
      if (!(await storageDir.exists())) {
        await storageDir.create();
      }
    } on Exception {
      loaded = true;
      crashed = true;
    }
    print(storageDir.path);
  }

  Future<void> loadProjects(List<SRProject> controllerProjects) async {
    //loads all projects found in the storage folder
    List<FileSystemEntity> projects = await storageDir
        .list(recursive: false)
        .toList();
    for (var project in projects) {
      if (project is Directory) {
        //load metadata
        final File metadataFile = File("${project.path}/metadata.txt");
        if (!(await metadataFile.exists())) {
          throw Exception(
            "Metadata for project in ${project.path} does not exist.",
          );
        }
        final String metadata = await metadataFile.readAsString();
      }
    }
  }

  Future<void> addProject(SRProject project) async {
    //create folder
    if (!loaded) return;

    final String storagePath = storageDir.path;
    final Directory projectDir = await Directory(
      "$storagePath/${project.id}",
    ).create(recursive: true);
  }
}
