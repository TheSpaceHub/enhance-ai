// lib/ui/pages/home_page.dart
import 'package:flutter/material.dart';
import '../widgets/left_nav.dart';
import '../widgets/main_workspace.dart';
import '../widgets/right_panel.dart';

class HomePage extends StatefulWidget {
  const HomePage({super.key});

  @override
  State<HomePage> createState() => _HomePageState();
}

class _HomePageState extends State<HomePage> {
  bool imageLoaded = false;
  bool processing = false;
  Image? originalImage;
  Image? srImage;

  void loadImage() {
    setState(() {
      originalImage = Image.asset('assets/lr_sample.png');
      imageLoaded = true;
    });
  }

  Future<void> processImage() async {
    setState(() {
      processing = true;
    });

    await Future.delayed(const Duration(seconds: 3));

    setState(() {
      srImage = Image.asset('assets/sr_sample.png');
      processing = false;
    });
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: Row(
        children: [
          // Left tab
          LeftNav(),

          // Middle tab
          Expanded(
            child: MainWorkspace(
              imageLoaded: imageLoaded,
              originalImage: originalImage,
              srImage: srImage,
              onUpload: loadImage,
              isProcessing: processing,
            ),
          ),

          // Right tab
          RightPanel(
            imageLoaded: imageLoaded,
            onUpscale: () async => await processImage(),
            isProcessing: processing,
          ),
        ],
      ),
    );
  }
}
