#!/usr/bin/env python3
"""
Improved audio generation script for WeakTok-IDM model.
Instead of generating from random noise, this script creates variations
of existing audio samples, which should produce more musical results.
"""

import torch
import numpy as np
import librosa
import soundfile as sf
import argparse
import os
import sys
from pathlib import Path
import random

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from model import WeakTokIDM
from loss import LScoreCalculator

def load_model(model_path, device):
    """Load trained model from checkpoint."""
    print(f"Loading model from: {model_path}")
    
    checkpoint = torch.load(model_path, map_location=device)
    
    model = WeakTokIDM(
        in_channels=1,
        out_channels=1,
        d_model=256,
        n_head=4,
        num_main_layers=4,
        dropout_rate=0.1,
        use_dynamic_chunking=True
    ).to(device)
    
    model.load_state_dict(checkpoint)
    model.eval()
    
    print("✓ Model loaded successfully")
    return model

def load_seed_audio(audio_path, target_duration=8.0, sample_rate=24000):
    """Load and preprocess seed audio for variation generation."""
    print(f"Loading seed audio: {audio_path}")
    
    # Load audio
    audio, sr = librosa.load(audio_path, sr=sample_rate)
    
    # Trim or pad to target duration
    target_samples = int(target_duration * sample_rate)
    if len(audio) > target_samples:
        # Random crop
        start_idx = random.randint(0, len(audio) - target_samples)
        audio = audio[start_idx:start_idx + target_samples]
    else:
        # Pad with zeros
        audio = np.pad(audio, (0, target_samples - len(audio)), mode='constant')
    
    return audio

def audio_to_spectrogram(audio, sample_rate=24000, n_fft=1024, hop_length=256, n_mels=128):
    """Convert audio to mel spectrogram."""
    mel_spec = librosa.feature.melspectrogram(
        y=audio,
        sr=sample_rate,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels
    )
    # Convert to dB
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    return mel_spec_db

def spectrogram_to_audio(spectrogram, sample_rate=24000, n_fft=1024, hop_length=256):
    """Convert mel spectrogram back to audio waveform."""
    if isinstance(spectrogram, torch.Tensor):
        spec_np = spectrogram.squeeze(0).detach().cpu().numpy()
    else:
        spec_np = spectrogram
    
    audio = librosa.feature.inverse.mel_to_audio(
        spec_np,
        sr=sample_rate,
        n_fft=n_fft,
        hop_length=hop_length,
        n_iter=32
    )
    
    return audio

def add_noise_to_spectrogram(spectrogram, noise_level=0.1):
    """Add controlled noise to spectrogram for variation."""
    noise = torch.randn_like(spectrogram) * noise_level
    return spectrogram + noise

def main():
    parser = argparse.ArgumentParser(description='Generate audio variations from seed audio using WeakTok-IDM')
    parser.add_argument('--model_path', type=str, required=True, help='Path to trained model checkpoint')
    parser.add_argument('--seed_audio', type=str, required=True, help='Path to seed audio file')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to save generated variations')
    parser.add_argument('--num_variations', type=int, default=5, help='Number of variations to generate')
    parser.add_argument('--noise_levels', type=float, nargs='+', default=[0.05, 0.1, 0.2], 
                        help='Noise levels for variations')
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
    
    # Load seed audio
    seed_audio = load_seed_audio(args.seed_audio, args.duration, args.sample_rate)
    
    # Convert to spectrogram
    seed_spec = audio_to_spectrogram(seed_audio, args.sample_rate)
    seed_spec_tensor = torch.tensor(seed_spec, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
    
    print(f"Seed audio shape: {seed_audio.shape}")
    print(f"Seed spectrogram shape: {seed_spec_tensor.shape}")
    
    # Save original seed audio for comparison
    seed_output_path = os.path.join(args.output_dir, 'seed_original.wav')
    sf.write(seed_output_path, seed_audio, args.sample_rate)
    print(f"✓ Saved seed audio: {seed_output_path}")
    
    # Generate variations
    print(f"\nGenerating {args.num_variations} variations with noise levels: {args.noise_levels}")
    
    generated_audio = []
    variation_count = 0
    
    with torch.no_grad():
        for noise_level in args.noise_levels:
            for i in range(args.num_variations // len(args.noise_levels) + 1):
                if variation_count >= args.num_variations:
                    break
                    
                variation_count += 1
                print(f"\nGenerating variation {variation_count}/{args.num_variations} (noise: {noise_level})...")
                
                try:
                    # Add noise to seed spectrogram
                    noisy_spec = add_noise_to_spectrogram(seed_spec_tensor, noise_level)
                    
                    # Generate variation through the model
                    output_spec = model(noisy_spec)
                    print(f"  ✓ Model forward pass successful")
                    
                    # Convert to audio
                    audio = spectrogram_to_audio(output_spec[0], args.sample_rate)
                    
                    # Normalize audio
                    audio = audio / (np.max(np.abs(audio)) + 1e-8)
                    audio = np.clip(audio, -1.0, 1.0)
                    
                    generated_audio.append(audio)
                    
                    # Save audio file
                    output_path = os.path.join(args.output_dir, f'variation_{variation_count:03d}_noise{noise_level:.2f}.wav')
                    sf.write(output_path, audio, args.sample_rate)
                    print(f"  ✓ Saved: {output_path}")
                    
                except Exception as e:
                    print(f"  ✗ Error generating variation {variation_count}: {e}")
                    continue
    
    if generated_audio:
        print(f"\n🎵 Successfully generated {len(generated_audio)} audio variations!")
        
        # Calculate L-Scores for analysis
        try:
            print("\n📊 Calculating L-Scores for generated variations...")
            
            # Include seed audio in analysis
            all_audio = [seed_audio] + generated_audio
            all_labels = ['Seed'] + [f'Var{i+1}' for i in range(len(generated_audio))]
            
            # Calculate L-Scores
            calculator = LScoreCalculator(sample_rate=args.sample_rate)
            spectrograms = []
            
            for audio in all_audio:
                mel_spec = audio_to_spectrogram(audio, args.sample_rate)
                spectrograms.append(mel_spec)
            
            spec_tensor = torch.tensor(np.stack(spectrograms), dtype=torch.float32).unsqueeze(1)
            l_scores = calculator(spec_tensor).cpu().numpy()
            
            print("\nL-Score Analysis:")
            print("Sample | Timbral | Rhythmic | Rhythmic | Structural")
            print("       | Complex | Density  | Irregul. | Complex   ")
            print("-" * 50)
            
            for i, (label, scores) in enumerate(zip(all_labels, l_scores)):
                print(f"{label:6s} |  {scores[0]:.3f}  |  {scores[1]:.3f}   |  {scores[2]:.3f}   |  {scores[3]:.3f}")
            
            # Calculate variation statistics
            if len(generated_audio) > 0:
                var_scores = l_scores[1:]  # Exclude seed
                mean_var = np.mean(var_scores, axis=0)
                std_var = np.std(var_scores, axis=0)
                
                print("-" * 50)
                print(f"VarMean|  {mean_var[0]:.3f}  |  {mean_var[1]:.3f}   |  {mean_var[2]:.3f}   |  {mean_var[3]:.3f}")
                print(f"VarStd |  {std_var[0]:.3f}  |  {std_var[1]:.3f}   |  {std_var[2]:.3f}   |  {std_var[3]:.3f}")
            
        except Exception as e:
            print(f"Warning: Could not calculate L-Scores: {e}")
        
        print(f"\n✓ All files saved to: {args.output_dir}")
        print("\n🎧 Listen to the variations and compare with the seed audio!")
        print("💡 Tip: Variations should sound like musical transformations of the seed, not random noise.")
        
    else:
        print("\n❌ No audio variations were generated successfully.")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
