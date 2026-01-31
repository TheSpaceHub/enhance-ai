import 'dart:io';
import 'dart:typed_data';
import 'package:flutter/material.dart';
import 'package:intl/intl.dart';
import 'package:app/ui/models/sr_experiment.dart';
import 'package:path_provider/path_provider.dart';

class StorageManager {
  /*
  Storage structure:
  - Project 1
    Metadata
    Original image
    - Run 1
      Data
      Metadata
    - Run 2
      Data
      Metadata
  - Project 2
    Metadata
    Original image
  */

  late Directory storageDir;
  bool loaded = false;
  bool crashed = true;
  final VoidCallback stateSetter;

  final String storageSeparator = "%";
  final String storageSeparator2 = "|"; //nested separating

  //storage manager initialization
  StorageManager(List<SRProject> controllerProjects, this.stateSetter) {
    initializeStorageManager(controllerProjects);
  }

  Future<void> initializeStorageManager(
    List<SRProject> controllerProjects,
  ) async {
    await setStorageDir();
    await loadAllProjects(controllerProjects);
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
  }

  Future<void> loadAllProjects(List<SRProject> controllerProjects) async {
    //loads all projects found in the storage folder
    List<FileSystemEntity> projects = await storageDir
        .list(recursive: false)
        .toList();
    //load them backwards
    for (int i = projects.length - 1; i >= 0; i--) {
      FileSystemEntity projectDir = projects[i];
      if (projectDir is Directory) {
        await loadProject(projectDir, controllerProjects);
      }
    }
    //once the projects are loaded, set state to update ui and let user access them
    stateSetter();
  }

  Future<void> loadProject(
    Directory projectDir,
    List<SRProject> controllerProjects,
  ) async {
    //load metadata
    final File metadataFile = File("${projectDir.path}/metadata.txt");
    if (!(await metadataFile.exists())) {
      throw Exception(
        "Metadata for project in ${projectDir.path} does not exist.",
      );
    }
    final Map<String, dynamic> projectMetadata = getProjectMetadata(
      await metadataFile.readAsString(),
    );

    //load original image
    final Uint8List projectOriginalImage = await File(
      "${projectDir.path}/originalImage",
    ).readAsBytes();

    //create empty project object
    final SRProject project = SRProject(
      id: projectMetadata["id"],
      name: projectMetadata["name"],
      timestamp: projectMetadata["timestamp"],
      originalBytes: projectOriginalImage,
      runs: [],
    );

    //add all runs
    //count how many runs there are
    final int runCount =
        (await projectDir.list(recursive: false).toList()).length -
        2; //remove metadata and original
    for (int i = 0; i < runCount; i++) {
      await loadRun("${projectDir.path}/run$i", project);
    }

    //add to controller
    controllerProjects.add(project);
  }

  Future<void> loadRun(String runPath, SRProject project) async {
    //load metadata
    final File metadataFile = File("$runPath/metadata.txt");
    if (!(await metadataFile.exists())) {
      throw Exception("Metadata for run in $runPath does not exist.");
    }
    final Map<String, dynamic> runMetadata = getRunMetadata(
      await metadataFile.readAsString(),
    );

    //load original image
    final Uint8List image = await File("$runPath/image").readAsBytes();

    final SRRun run = SRRun(
      id: runMetadata["id"],
      modelName: runMetadata["modelName"],
      upscaleFactor: runMetadata["upscaleFactor"],
      device: runMetadata["device"],
      inferenceTime: runMetadata["inferenceTime"],
      isProcessing: false,
      metrics: runMetadata["metrics"],
      resultBytes: image,
    );

    project.addRun(run);
  }

  Map<String, dynamic> getProjectMetadata(String metadataString) {
    //given a stored project metadata string, returns the structured metadata
    Map<String, dynamic> metadata = Map<String, dynamic>.from(
      stringToMap(metadataString, storageSeparator),
    );
    metadata["timestamp"] = DateFormat(
      'yyyy-MM-dd-kk:mm',
    ).parse(metadata["timestamp"]);

    return metadata;
  }

  Map<String, dynamic> getRunMetadata(String metadataString) {
    //given a stored run metadata string, returns the structured metadata
    Map<String, dynamic> metadata = Map<String, dynamic>.from(
      stringToMap(metadataString, storageSeparator),
    );
    //structure metrics
    metadata["metrics"] = stringToMap(metadata["metrics"], storageSeparator2);
    metadata["upscaleFactor"] = double.parse(metadata["upscaleFactor"]);

    return metadata;
  }

  //helper functions
  String mapToString(Map<String, dynamic> m, String separator) {
    String result = "";
    for (var key in m.keys) {
      try {
        result += key;
        result += separator;
        result += m[key].toString();
        result += separator;
      } on Exception {
        throw Exception("Could not cast to string the object with key $key.");
      }
    }
    return result;
  }

  Map<String, String> stringToMap(String s, String separator) {
    Map<String, String> map = {};
    List<String> items = s.split(separator);
    if (!(items.length % 2 == 1 && items[items.length - 1] == "")) {
      throw Exception("Map structure is not present in string.");
    }
    for (int i = 0; i < items.length / 2 - 1; i++) {
      map[items[2 * i]] = items[2 * i + 1];
    }
    return map;
  }

  //project creation
  Future<void> createProjectMetadata(
    SRProject project,
    String folderPath,
  ) async {
    //Given a project and a path, creates its metadata
    final File metadataFile = File("$folderPath/metadata.txt");
    await metadataFile.create();
    final String metadataString = mapToString({
      "id": project.id,
      "name": project.name,
      "timestamp": DateFormat(
        'yyyy-MM-dd-kk:mm',
      ).format(project.timestamp), //must cast datetime to string
    }, storageSeparator);
    await metadataFile.writeAsString(metadataString);
  }

  Future<void> addProjectToStorage(SRProject project) async {
    if (!loaded) {
      throw Exception(
        "Could not create new project. Metadata has not loaded yet",
      );
    }

    //create project folder
    final String storagePath = storageDir.path;
    final String projectPath = "$storagePath/${project.id}";
    await Directory(projectPath).create(recursive: true);

    //create metadata for project
    await createProjectMetadata(project, "$storagePath/${project.id}");

    //create original photo
    final File originalImageFile = await File(
      "$projectPath/originalImage",
    ).create();
    await originalImageFile.writeAsBytes(project.originalBytes);
  }

  //runs
  Future<void> createRunMetadata(SRRun run, String folderPath) async {
    //Given a run and a path, creates its metadata
    final File metadataFile = File("$folderPath/metadata.txt");
    await metadataFile.create();
    final String metadataString = mapToString({
      "id": run.id,
      "modelName": run.modelName,
      "upscaleFactor": run.upscaleFactor,
      "device": run.device,
      "inferenceTime": run.inferenceTime,
      "metrics": mapToString(run.metrics, storageSeparator2),
    }, storageSeparator);
    await metadataFile.writeAsString(metadataString);
  }

  Future<void> addRunToStorage(final SRRun run, final SRProject project) async {
    //Adds run to storage. Does NOT modify run or project

    //check if project exists in storage
    final String projectPath = "${storageDir.path}/${project.id}";
    final Directory projectDir = Directory(projectPath);
    if (!(await projectDir.exists())) {
      throw Exception("Project does not exist. Could not add new run.");
    }

    //count how many runs there are already
    final int runCount =
        (await projectDir.list(recursive: false).toList()).length -
        2; //remove metadata and original

    //create run dir
    final String runPath = "$projectPath/run$runCount";
    await Directory(runPath).create();

    //store metadata and image
    await createRunMetadata(run, runPath);
    final File imageFile = await File("$runPath/image").create();
    if (run.resultBytes == null) {
      throw Exception("Cannot store empty run");
    }
    imageFile.writeAsBytes(run.resultBytes!);
  }
}
