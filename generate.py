#!/usr/bin/env python3
"""
Audio generation script for WeakTok-IDM model.
Generates audio samples from a trained model and saves them as WAV files.
"""

import torch
import numpy as np
import librosa
import soundfile as sf
import argparse
import os
import sys
from pathlib import Path

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from model import WeakTokIDM
from loss import LScoreCalculator

def load_model(model_path, device):
    """Load trained model from checkpoint."""
    print(f"Loading model from: {model_path}")
    
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device)
    
    # Initialize model with default config (should match training config)
    model = WeakTokIDM(
        in_channels=1,
        out_channels=1,
        d_model=256,
        n_head=4,
        num_main_layers=4,
        dropout_rate=0.1,
        use_dynamic_chunking=True
    ).to(device)
    
    # Load model weights
    model.load_state_dict(checkpoint)
    model.eval()
    
    print("✓ Model loaded successfully")
    return model

def generate_random_input(batch_size, n_mels, frames, device):
    """Generate random input spectrogram for unconditional generation."""
    # Create random noise as input
    random_input = torch.randn(batch_size, 1, n_mels, frames, device=device)
    return random_input

def spectrogram_to_audio(spectrogram, sample_rate=24000, n_fft=1024, hop_length=256):
    """Convert mel spectrogram back to audio waveform."""
    # Convert from tensor to numpy and squeeze channel dimension
    if isinstance(spectrogram, torch.Tensor):
        spec_np = spectrogram.squeeze(0).detach().cpu().numpy()
    else:
        spec_np = spectrogram
    
    # Convert mel spectrogram to audio using Griffin-Lim
    audio = librosa.feature.inverse.mel_to_audio(
        spec_np,
        sr=sample_rate,
        n_fft=n_fft,
        hop_length=hop_length,
        n_iter=32  # More iterations for better quality
    )
    
    return audio

def calculate_l_scores(audio_batch, sample_rate=24000):
    """Calculate L-Scores for generated audio to verify complexity."""
    calculator = LScoreCalculator(sample_rate=sample_rate)
    
    # Convert audio to spectrograms for L-Score calculation
    spectrograms = []
    for audio in audio_batch:
        # Convert to mel spectrogram
        mel_spec = librosa.feature.melspectrogram(
            y=audio,
            sr=sample_rate,
            n_fft=1024,
            hop_length=256,
            n_mels=128
        )
        # Convert to dB
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        spectrograms.append(mel_spec_db)
    
    # Stack into batch tensor
    spec_tensor = torch.tensor(np.stack(spectrograms), dtype=torch.float32).unsqueeze(1)
    
    # Calculate L-Scores
    with torch.no_grad():
        l_scores = calculator(spec_tensor)
    
    return l_scores.cpu().numpy()

def main():
    parser = argparse.ArgumentParser(description='Generate audio samples from trained WeakTok-IDM model')
    parser.add_argument('--model_path', type=str, required=True, help='Path to trained model checkpoint')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to save generated audio')
    parser.add_argument('--num_samples', type=int, default=5, help='Number of audio samples to generate')
    parser.add_argument('--duration', type=float, default=8.0, help='Duration of each sample in seconds')
    parser.add_argument('--sample_rate', type=int, default=24000, help='Audio sample rate')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model
    model = load_model(args.model_path, device)
    
    # Calculate spectrogram dimensions
    frames = int(args.duration * args.sample_rate / 256)  # hop_length = 256
    n_mels = 128
    
    print(f"Generating {args.num_samples} audio samples...")
    print(f"Duration: {args.duration}s, Sample rate: {args.sample_rate}Hz")
    print(f"Spectrogram shape: {n_mels} mels × {frames} frames")
    
    generated_audio = []
    
    with torch.no_grad():
        for i in range(args.num_samples):
            print(f"\nGenerating sample {i+1}/{args.num_samples}...")
            
            # Generate random input
            input_spec = generate_random_input(1, n_mels, frames, device)
            
            # Generate output spectrogram
            try:
                output_spec = model(input_spec)
                print(f"  ✓ Model forward pass successful")
                
                # Convert to audio
                audio = spectrogram_to_audio(
                    output_spec[0], 
                    sample_rate=args.sample_rate
                )
                
                # Normalize audio to prevent clipping
                audio = audio / (np.max(np.abs(audio)) + 1e-8)
                audio = np.clip(audio, -1.0, 1.0)
                
                generated_audio.append(audio)
                
                # Save audio file
                output_path = os.path.join(args.output_dir, f'generated_sample_{i+1:03d}.wav')
                sf.write(output_path, audio, args.sample_rate)
                print(f"  ✓ Saved: {output_path}")
                
            except Exception as e:
                print(f"  ✗ Error generating sample {i+1}: {e}")
                continue
    
    if generated_audio:
        print(f"\n🎵 Successfully generated {len(generated_audio)} audio samples!")
        
        # Calculate L-Scores for analysis
        try:
            print("\n📊 Calculating L-Scores for generated audio...")
            l_scores = calculate_l_scores(generated_audio, args.sample_rate)
            
            print("L-Score Analysis:")
            print("Sample | Timbral | Rhythmic | Rhythmic | Structural")
            print("       | Complex | Density  | Irregul. | Complex   ")
            print("-" * 50)
            
            for i, scores in enumerate(l_scores):
                print(f"  {i+1:2d}   |  {scores[0]:.3f}  |  {scores[1]:.3f}   |  {scores[2]:.3f}   |  {scores[3]:.3f}")
            
            # Calculate mean scores
            mean_scores = np.mean(l_scores, axis=0)
            print("-" * 50)
            print(f" Mean  |  {mean_scores[0]:.3f}  |  {mean_scores[1]:.3f}   |  {mean_scores[2]:.3f}   |  {mean_scores[3]:.3f}")
            
        except Exception as e:
            print(f"Warning: Could not calculate L-Scores: {e}")
        
        print(f"\n✓ All files saved to: {args.output_dir}")
        print("\n🎧 Listen to the generated samples to evaluate the curriculum learning effect!")
        
    else:
        print("\n❌ No audio samples were generated successfully.")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
