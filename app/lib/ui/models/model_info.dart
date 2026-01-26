/// Describes a super-resolution model and its technical characteristics
class ModelInfo {
  final String name;
  final String description;
  final List<String> strengths;
  final String complexity; // Low, Medium, High

  ModelInfo({
    required this.name,
    required this.description,
    required this.strengths,
    required this.complexity,
  });
}

/// Registry mapping model identifiers to their descriptive metadata
final Map<String, ModelInfo> modelRegistry = {
  'CNNU': ModelInfo(
    name: 'CNNU (Basic CNN)',
    description:
        'A classic three-layer convolutional neural network and a historical baseline for super-resolution.',
    strengths: ['Very fast', 'Ideal for legacy hardware', 'Stable'],
    complexity: 'Low',
  ),
  'ESPCN': ModelInfo(
    name: 'ESPCN',
    description:
        'Uses sub-pixel convolution to reconstruct the image at the final stage, significantly reducing memory usage.',
    strengths: ['Mobile-friendly', 'Sharp edges', 'Low VRAM usage'],
    complexity: 'Low',
  ),
  'SRResNet': ModelInfo(
    name: 'SRResNet',
    description:
        'Residual network focused on pixel-wise fidelity (PSNR), forming the foundation of SRGAN.',
    strengths: ['High fidelity', 'Artifact-free', 'Smooth textures'],
    complexity: 'Medium',
  ),
  'SRGAN': ModelInfo(
    name: 'SRGAN',
    description:
        'Generative Adversarial Network that produces perceptually realistic textures.',
    strengths: [
      'Photorealism',
      'Texture recovery (hair, skin)',
      'Professional look',
    ],
    complexity: 'High',
  ),
};
