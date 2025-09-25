# WeakTOK: Intelligent Dance Music Generation with Curriculum Learning

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-orange.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

## Overview

**WeakTOK** is a novel approach to Intelligent Dance Music (IDM) generation that combines deep learning with curriculum learning principles. The project implements a sophisticated neural architecture that learns to generate complex electronic music by progressively increasing the musical complexity during training, guided by a multi-dimensional complexity metric called **L-Score**.

### Key Features

- 🎵 **IDM Generation**: Specialized for generating complex, experimental electronic music
- 🧠 **Curriculum Learning**: Progressive training from simple to complex musical patterns
- 📊 **L-Score Metrics**: 4-dimensional complexity measurement (Timbral, Rhythmic Density, Rhythmic Irregularity, Structural)
- 🏗️ **Hybrid Architecture**: U-Net encoder-decoder with Dynamic Chunking and Transformer processing
- 🔄 **Audio Variations**: Generate musical variations from seed audio samples
- ⚡ **GPU Optimized**: CUDA-accelerated training and inference

## Architecture

The WeakTOK-IDM model consists of three main components:

### 1. U-Net Encoder-Decoder
- **Encoder**: Progressively downsamples mel spectrograms through convolutional layers
- **Decoder**: Reconstructs spectrograms with skip connections for detail preservation
- **Lightweight Design**: Reduced channel width and depth for memory efficiency

### 2. Dynamic Chunking Layer
```python
class DynamicChunkingLayer(nn.Module):
    """
    Intelligently segments audio sequences into variable-length chunks
    based on content similarity, enabling context-aware processing.
    """
```

### 3. Transformer Processing
- Multi-head attention mechanism for sequence modeling
- Processes dynamically chunked representations
- Configurable depth and attention heads

## L-Score: Musical Complexity Metric

The L-Score quantifies the "listening challenge" of generated music across four dimensions:

1. **Timbral Complexity** (Spectral Entropy): Measures the noise-like quality and spectral richness
2. **Rhythmic Density** (Onset Density): Number of onsets per second
3. **Rhythmic Irregularity** (IOI Variance): Variability in inter-onset intervals
4. **Structural Complexity** (Low Self-Similarity): Measures repetitiveness vs. variation

### Curriculum Learning Process

The model uses L-Score distribution matching to progressively learn:
1. Start with simple, regular patterns (low L-Scores)
2. Gradually increase complexity requirements during training
3. Target the L-Score distribution of real IDM datasets

## Installation

### Prerequisites
- Python 3.8+
- CUDA-compatible GPU (recommended)
- 8GB+ GPU memory for training

### Setup

1. **Clone the repository:**
```bash
git clone https://github.com/your-username/WeakTOK.git
cd WeakTOK
```

2. **Install PyTorch with CUDA support:**
```bash
# For CUDA 12.1 (RTX 40-series GPUs)
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
```

3. **Install dependencies:**
```bash
pip install -r requirements.txt
```

## Usage

### Configuration

Edit `config.yaml` to customize training parameters:

```yaml
# Audio processing settings
audio:
  sample_rate: 24000
  chunk_duration_s: 8.0
  n_fft: 1024
  hop_length: 256
  n_mels: 128

# Model architecture
model:
  use_dynamic_chunking: true
  d_model: 256
  n_head: 4
  num_main_layers: 4

# Training parameters
training:
  epochs: 100
  batch_size: 4
  learning_rate: 0.0001
  lambda_l_score: 0.1
```

### Training

Train the model on your IDM dataset:

```bash
python train.py --config config.yaml
```

The training script will:
- Initialize the WeakTOK-IDM model
- Set up L-Score calculation and distribution matching loss
- Save the best model based on validation loss
- Log training progress and GPU memory usage

### Audio Generation

#### Generate from Random Noise
```bash
python generate.py \
    --model_path results/models/best_model.pth \
    --output_dir generated_audio/ \
    --num_samples 5 \
    --duration 8.0
```

#### Generate Variations from Seed Audio
```bash
python generate_variations.py \
    --model_path results/models/best_model.pth \
    --seed_audio path/to/seed.wav \
    --output_dir variations/ \
    --num_variations 5 \
    --noise_levels 0.05 0.1 0.2
```

### Example Output

The generation scripts provide detailed L-Score analysis:

```
L-Score Analysis:
Sample | Timbral | Rhythmic | Rhythmic | Structural
       | Complex | Density  | Irregul. | Complex   
--------------------------------------------------
Seed   |  0.724  |  12.450  |  0.156   |  0.823
Var1   |  0.731  |  11.890  |  0.142   |  0.797
Var2   |  0.698  |  13.120  |  0.168   |  0.845
```

## Project Structure

```
WeakTOK/
├── config.yaml              # Training configuration
├── model.py                 # WeakTOK-IDM architecture
├── train.py                 # Training script
├── loss.py                  # L-Score calculation and loss functions
├── generate.py              # Audio generation from noise
├── generate_variations.py   # Variation generation from seed
├── requirements.txt         # Python dependencies
├── generated_audio/         # Generated samples
└── variations/             # Audio variations
    ├── seed_original.wav
    ├── variation_001_noise0.05.wav
    ├── variation_002_noise0.05.wav
    └── variation_003_noise0.10.wav
```

## Model Details

### WeakTOK-IDM Architecture

```python
class WeakTokIDM(nn.Module):
    """
    Lightweight U-Net based model with Dynamic Chunking and Transformer.
    
    Key components:
    - ConvBlock: (Conv2d -> BatchNorm2d -> LeakyReLU) x 2
    - DeconvBlock: Transposed convolution with skip connections
    - DynamicChunkingLayer: Content-aware sequence segmentation
    - TransformerEncoder: Multi-head attention processing
    """
```

### Memory Optimization

The model implements several optimizations for stable training:
- Reduced channel width (32→64→128 instead of 64→128→256→512)
- Reduced depth (3 layers instead of 4)
- Gradient accumulation for larger effective batch sizes
- Dynamic memory management with GPU monitoring

## Loss Function

The training combines two loss components:

1. **L-Score Distribution Loss**: Matches the statistical distribution of complexity metrics
2. **Reconstruction Loss**: Standard L1 loss for audio quality preservation

```python
loss = l_score_distribution_loss + (lambda_recon * reconstruction_loss)
```

## Results

The WeakTOK model generates IDM tracks with:
- **Controllable complexity** through L-Score targeting
- **Musical coherence** via curriculum learning progression
- **Timbral diversity** through spectral entropy optimization
- **Rhythmic sophistication** via onset pattern modeling

### Sample Results

Generated samples demonstrate:
- Progressive increase in musical complexity during training
- Coherent musical structure with IDM characteristics
- Successful variation generation preserving musical identity
- L-Score distributions matching target IDM datasets

## Research Contributions

1. **Novel Curriculum Learning Approach**: First application of complexity-based curriculum learning to IDM generation
2. **Multi-dimensional Complexity Metric**: L-Score provides comprehensive musical complexity measurement
3. **Hybrid Architecture**: Combines U-Net spatial processing with Transformer sequence modeling
4. **Dynamic Chunking**: Content-aware sequence segmentation for better musical understanding

## Citation



## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Built with PyTorch and Essentia for robust audio processing
- Inspired by curriculum learning principles in deep learning
- Developed for the ISMIR LLM4Music Satellite Event

---

**🎧 Experience the future of intelligent electronic music generation with WeakTOK!**
