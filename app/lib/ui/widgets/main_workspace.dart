import 'package:flutter/material.dart';

class MainWorkspace extends StatelessWidget {
  final bool imageLoaded;
  final VoidCallback onUpload;

  const MainWorkspace({
    super.key,
    required this.imageLoaded,
    required this.onUpload,
  });

  @override
  Widget build(BuildContext context) {
    if (!imageLoaded) {
      return Center(
        child: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            const Icon(Icons.image_outlined, size: 80, color: Colors.white54),
            const SizedBox(height: 24),
            ElevatedButton(
              onPressed: onUpload,
              child: const Text('Upload Image'),
            ),
          ],
        ),
      );
    }

    return const Center(
      child: Text('Image preview here', style: TextStyle(fontSize: 18)),
    );
  }
}
