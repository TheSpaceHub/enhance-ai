class ModelInfo {
  final String name;
  final String description;
  final String recommendedFor;
  final List<String> strengths;
  final double speedScore; // 0.0 (Slow) to 1.0 (Very Fase) - For UI speed bars

  ModelInfo({
    required this.name,
    required this.description,
    required this.recommendedFor,
    required this.strengths,
    required this.speedScore,
  });
}

final Map<String, ModelInfo> modelRegistry = {
  'CNNU': ModelInfo(
    name: 'CNNU',
    description:
        'The lightest model. Improves resolution by smoothing edges but does not add new details.',
    recommendedFor: 'Quick tests, older devices, simple icons.',
    strengths: ['Ultra-fast', 'Low resource usage'],
    speedScore: 1.0,
  ),
  'ESPCN': ModelInfo(
    name: 'ESPCN',
    description:
        'Optimized for video and real-time. Provides much sharper edges than CNNU without sacrificing speed.',
    recommendedFor: 'Video, real-time upscaling, vector graphics.',
    strengths: ['Sharp edges', 'Memory efficient'],
    speedScore: 0.9,
  ),
  'SRResNet': ModelInfo(
    name: 'SRResNet',
    description:
        'Focused on mathematical accuracy. Tries to stay as true as possible to the original image, resulting in clean but soft images.',
    recommendedFor: 'Medical images, scanned text, photo compression.',
    strengths: ['High fidelity (PSNR)', 'Low noise'],
    speedScore: 0.5,
  ),
  'SRGAN': ModelInfo(
    name: 'SRGAN',
    description:
        'Generates realistic textures (hair, grass, skin) that did not exist in the original. May slightly alter facial identity.',
    recommendedFor: 'Large prints, Restoring old photos.',
    strengths: ['Professional look', 'High perceptual detail'],
    speedScore: 0.2,
  ),
};
