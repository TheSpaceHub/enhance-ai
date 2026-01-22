import 'package:flutter/material.dart';
import '../widgets/right_panel.dart';

class HomePage extends StatefulWidget {
  const HomePage({super.key});

  @override
  State<HomePage> createState() => _HomePageState();
}

class _HomePageState extends State<HomePage> {
  bool imageLoaded = false; // Imagen cargada
  bool processing = false; // Procesando SR
  Image? originalImage;
  Image? srImage; // Imagen superresolucionada

  // Simula cargar una imagen
  void loadImage() {
    setState(() {
      originalImage = Image.asset('assets/lr_sample.png');
      imageLoaded = true;
    });
  }

  // Simula procesar imagen
  Future<void> processImage() async {
    setState(() {
      processing = true;
    });

    // Aquí iría tu llamada al ML
    await Future.delayed(const Duration(seconds: 3));

    setState(() {
      srImage = Image.asset('assets/sr_sample.png'); // Imagen SR simulada
      processing = false;
    });
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: Row(
        children: [
          // Sidebar Izquierda
          Container(
            width: 80,
            color: Colors.grey[850],
            child: Column(
              children: const [
                SizedBox(height: 50),
                Icon(Icons.history, color: Colors.white),
                SizedBox(height: 10),
                Icon(Icons.folder, color: Colors.white),
              ],
            ),
          ),

          // Área Central
          Expanded(
            child: Center(
              child: imageLoaded
                  ? (processing
                        ? const CircularProgressIndicator(
                            color: Colors.greenAccent,
                          )
                        : SizedBox(
                            width: 800, // ancho fijo
                            height: 800, // alto fijo
                            child: FittedBox(
                              fit: BoxFit.contain, // mantiene proporción
                              child: srImage ?? originalImage,
                            ),
                          ))
                  : ElevatedButton(
                      onPressed: loadImage,
                      child: const Text('Upload Image'),
                    ),
            ),
          ),

          // Right Panel
          RightPanel(
            imageLoaded: imageLoaded,
            onUpscale: () async {
              await processImage();
            },
            isProcessing: processing,
          ),
        ],
      ),
    );
  }
}
