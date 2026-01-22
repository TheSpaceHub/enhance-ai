import 'package:flutter/material.dart';

class LeftNav extends StatelessWidget {
  const LeftNav({super.key});

  @override
  Widget build(BuildContext context) {
    return Container(
      width: 80,
      color: Colors.black54,
      child: Column(
        children: const [
          SizedBox(height: 24),
          Icon(Icons.auto_fix_high, size: 32),
          SizedBox(height: 40),
          Icon(Icons.home),
          SizedBox(height: 20),
          Icon(Icons.history),
        ],
      ),
    );
  }
}
