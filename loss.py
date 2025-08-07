# src/loss.py

import torch
import torch.nn as nn
import librosa
import numpy as np
import essentia
import essentia.standard as es
from concurrent.futures import ThreadPoolExecutor
import functools

# Essentia is used for robust onset detection.
# It's a powerful MIR library.

# ----------------------------------------------------------------------------
# L-Score Calculation Module
# ----------------------------------------------------------------------------

class LScoreCalculator(nn.Module):
    """
    Calculates the 4-dimensional L-Score vector for a batch of audio.
    The L-Score quantifies the "listening challenge" and consists of:
    1. Timbral Complexity (Spectral Entropy)
    2. Rhythmic Complexity (Onset Density)
    3. Rhythmic Complexity (Rhythmic Irregularity)
    4. Structural Complexity (Low Self-Similarity)
    """
    def __init__(self, sample_rate=24000, n_fft=1024, hop_length=256, eps=1e-8):
        super().__init__()
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.eps = eps

        # Initialize Essentia's onset detection algorithm.
        # The 'complex' method is robust for complex electronic music.[2, 3, 4]
        self.onset_detector = es.OnsetDetection(method='complex', sampleRate=self.sample_rate)

    def _calculate_spectral_entropy(self, power_spec):
        """
        Calculates normalized spectral entropy. High entropy means more noise-like.
        This is a differentiable operation.
        Args:
            power_spec (Tensor): Power spectrogram, shape (batch, freqs, frames).
        Returns:
            Tensor: Spectral entropy for each item in the batch, shape (batch,).
        """
        # Normalize each frame's power spectrum to be a probability distribution
        p = power_spec / (power_spec.sum(dim=1, keepdim=True) + self.eps)
        
        # Calculate entropy: H = -sum(p * log2(p))
        entropy = -torch.sum(p * torch.log2(p + self.eps), dim=1)
        
        # Normalize by log2 of the number of frequency bins
        # This bounds the entropy between 0 and 1.[5, 6]
        normalized_entropy = entropy / torch.log2(torch.tensor(power_spec.shape[1]))
        
        # Return the mean entropy across all frames for each audio clip
        final_entropy = normalized_entropy.mean(dim=1)
        
        # Final check for NaN/Inf, can happen with extreme inputs
        final_entropy = torch.nan_to_num(final_entropy, nan=0.5, posinf=1.0, neginf=0.0)
        return final_entropy

    def _calculate_rhythmic_complexity(self, audio_waveform):
        """
        Calculates onset density and irregularity using Essentia.
        This operation is NOT differentiable and is treated as a reward signal.
        Args:
            audio_waveform (np.ndarray): A single audio waveform.
        Returns:
            tuple: (onset_density, rhythmic_irregularity)
        """
        # --- Stability Check --- 
        # If the inverse STFT produced non-finite values, we cannot proceed.
        if not np.all(np.isfinite(audio_waveform)):
            return 0.0, 0.0 # Return default, safe values

        # Essentia works with numpy arrays (convert to float32 to avoid warnings)
        audio_waveform = audio_waveform.astype(np.float32)

        # --- Resample for Essentia Compatibility ---
        # Essentia's algorithms are often optimized for or expect specific sample rates.
        # We resample to 44100 Hz, a standard in audio processing, to avoid warnings and ensure correctness.
        target_sr = 44100
        if self.sample_rate != target_sr:
            audio_waveform = librosa.resample(audio_waveform, orig_sr=self.sample_rate, target_sr=target_sr)
        
        # Update the onset detector's sample rate to match the resampled audio
        self.onset_detector = es.OnsetDetection(method='complex', sampleRate=target_sr)
        # Correct Essentia Usage: First, compute the full ODF for the signal.
        pool = essentia.Pool()
        w = es.Windowing(type='hann')
        fft = es.FFT()
        c2p = es.CartesianToPolar()
        od = self.onset_detector

        # Process the audio frame by frame to get the ODF
        for frame in es.FrameGenerator(audio_waveform, frameSize=self.n_fft, hopSize=self.hop_length):
            mag, phase = c2p(fft(w(frame)))
            odf_value = od(mag, phase)
            pool.add('features.odf', odf_value)

        # Retrieve the full ODF array from the pool
        odf_array = pool['features.odf']

        # Use librosa for onset detection instead of problematic Essentia Onsets
        try:
            # Convert ODF to numpy array
            odf_array = np.array(odf_array)
            
            # Use librosa's onset detection on the ODF
            # The hop_length here refers to the ODF's hop length, not the original audio
            onset_frames = librosa.onset.onset_detect(
                onset_envelope=odf_array,
                sr=target_sr / self.hop_length,  # ODF sample rate
                hop_length=1,  # ODF is already downsampled
                units='time'
            )
            
            onset_times = onset_frames
            
        except Exception as e:
            print(f"[Warning] Librosa onset detection failed: {e}")
            onset_times = np.array([])

        if len(onset_times) < 2:
            return 0.0, 0.0 # Not enough onsets to calculate complexity

        # Onset Density: onsets per second
        duration = len(audio_waveform) / target_sr
        if duration < self.eps:
            return 0.0, 0.0 # Avoid division by zero
        onset_density = len(onset_times) / duration

        # Rhythmic Irregularity: standard deviation of inter-onset intervals (IOIs)
        iois = np.diff(onset_times)
        rhythmic_irregularity = np.std(iois)

        return onset_density, rhythmic_irregularity

    def _calculate_structural_complexity(self, audio_waveform):
        """
        Calculates the low self-similarity of the audio using MFCCs and a recurrence matrix.
        This measures how repetitive vs. varied the audio is structurally.
        Args:
            audio_waveform (np.ndarray): A single audio waveform.
        Returns:
            float: The low self-similarity score.
        """
        try:
            # Check for minimum audio length and non-silence
            if len(audio_waveform) < self.sample_rate * 0.5:  # Less than 0.5 seconds
                return 0.5  # Default moderate complexity
            
            # Check for silence or very low energy
            if np.max(np.abs(audio_waveform)) < 1e-6:
                return 0.0  # Silent audio has no structural complexity
            
            # Use MFCCs as the feature representation for similarity
            mfccs = librosa.feature.mfcc(y=audio_waveform, sr=self.sample_rate, n_mfcc=20)
            
            # Check if MFCC extraction succeeded
            if mfccs.size == 0 or np.all(np.isnan(mfccs)):
                return 0.5  # Default moderate complexity for failed extraction
            
            # Compute the self-similarity matrix with error handling
            try:
                ssm = librosa.segment.recurrence_matrix(mfccs, mode='affinity', sparse=False)
            except (librosa.util.exceptions.ParameterError, ValueError) as e:
                # Handle bandwidth estimation errors or other librosa issues
                return 0.5  # Default moderate complexity
            
            # Check if recurrence matrix is valid
            if ssm.size == 0 or np.all(np.isnan(ssm)):
                return 0.5
            
            # We want to measure how dissimilar it is, so we are interested in the
            # average similarity *off* the main diagonal.
            # Exclude a small band around the diagonal to ignore local correlations.
            lag = min(5, ssm.shape[0] // 4)  # Adaptive lag based on matrix size
            ssm_off_diag = np.copy(ssm)
            for i in range(ssm.shape[0]):
                ssm_off_diag[i, max(0, i-lag):min(ssm.shape[1], i+lag+1)] = 0
            
            # Low self-similarity = 1 - mean similarity
            valid_similarities = ssm_off_diag[ssm_off_diag > 0]
            if len(valid_similarities) == 0:
                return 1.0  # No similarities found = maximum complexity
            
            mean_similarity = np.mean(valid_similarities)
            low_self_similarity = 1.0 - (mean_similarity if not np.isnan(mean_similarity) else 0.0)
            
            # Clamp to valid range [0, 1]
            return np.clip(low_self_similarity, 0.0, 1.0)
            
        except Exception as e:
            # Catch any other unexpected errors
            print(f"Warning: Structural complexity calculation failed: {e}")
            return 0.5  # Default moderate complexity

    def forward(self, generated_spectrograms):
        """
        Main forward pass to compute L-Scores for a batch of spectrograms.
        Args:
            generated_spectrograms (Tensor): Batch of Mel spectrograms from the model.
                                             Shape: (batch, num_mels, frames)
        Returns:
            Tensor: A tensor of L-Scores for the batch. Shape: (batch, 4)
        """
        batch_size = generated_spectrograms.shape[0]
        device = generated_spectrograms.device
        
        # --- Input Validation ---
        # Check for NaN/Inf in the input spectrograms before any processing
        if not torch.all(torch.isfinite(generated_spectrograms)):
            # Return default complexity scores for invalid inputs
            return torch.full((batch_size, 4), 0.5, device=device)
        
        # --- 1. Timbral Complexity (Differentiable) ---
        # This part remains in the computation graph to provide a gradient.
        # This part remains in the computation graph to provide a gradient.
        # Squeeze channel dim and convert from dB to power using pure PyTorch
        # to avoid detaching the tensor from the computation graph.
        db_spec = generated_spectrograms.squeeze(1)

        # --- Stability Clamp ---
        # Clamp the dB values to a reasonable range before converting to power
        # to prevent generating `inf` from extreme model outputs.
        db_spec = torch.clamp(db_spec, min=-100, max=80)

        power_spec = 10**(db_spec / 10.0)
        l_timbre = self._calculate_spectral_entropy(power_spec)

        # --- 2, 3, 4. Rhythm & Structure (Non-Differentiable) ---
        # These are calculated without tracking gradients.
        with torch.no_grad():
            # Convert all spectrograms to audio first (can be parallelized)
            audio_batch = []
            for i in range(batch_size):
                # Squeeze the channel dimension (C) from (1, H, W) to (H, W) for librosa
                # Squeeze the channel dimension (C) from (1, H, W) to (H, W) for librosa
                spec_single = generated_spectrograms[i].squeeze(0).detach().cpu().numpy().astype(np.float32)
                audio_single = librosa.feature.inverse.mel_to_audio(
                    spec_single,
                    sr=self.sample_rate,
                    n_fft=self.n_fft,
                    hop_length=self.hop_length
                )
                audio_batch.append(audio_single)
            
            # Parallel processing for complexity calculations
            def process_single_audio(audio):
                density, irreg = self._calculate_rhythmic_complexity(audio)
                structure = self._calculate_structural_complexity(audio)
                return density, irreg, structure
            
            # Use ThreadPoolExecutor for parallel processing
            with ThreadPoolExecutor(max_workers=min(4, batch_size)) as executor:
                results = list(executor.map(process_single_audio, audio_batch))
            
            # Unpack results
            l_rhythm_density_list = [r[0] for r in results]
            l_rhythm_irreg_list = [r[1] for r in results]
            l_structure_list = [r[2] for r in results]

            l_rhythm_density = torch.tensor(l_rhythm_density_list, device=device)
            l_rhythm_irreg = torch.tensor(l_rhythm_irreg_list, device=device)
            l_structure = torch.tensor(l_structure_list, device=device)

        # Stack all scores into a single tensor
        l_scores = torch.stack([l_timbre, l_rhythm_density, l_rhythm_irreg, l_structure], dim=1)
        
        return l_scores


# ----------------------------------------------------------------------------
# Distribution Matching Loss Module
# ----------------------------------------------------------------------------

class LScoreDistributionLoss(nn.Module):
    """
    Calculates the loss by comparing the distribution of L-Scores from the
    generated batch to the target distribution of the real dataset.
    This implementation uses moment matching (comparing mean and std).
    """
    def __init__(self, target_mean, target_std):
        """
        Args:
            target_mean (Tensor): The pre-computed mean L-Score vector of the training data. Shape: (4,)
            target_std (Tensor): The pre-computed std L-Score vector of the training data. Shape: (4,)
        """
        super().__init__()
        # Register target stats as buffers so they are moved to the correct device
        # with the model, but are not considered model parameters.
        self.register_buffer('target_mean', target_mean)
        self.register_buffer('target_std', target_std)
        
        self.l1_loss = nn.L1Loss()

    def forward(self, batch_l_scores):
        """
        Args:
            batch_l_scores (Tensor): The L-Scores computed for the current batch. Shape: (batch, 4)
        Returns:
            Tensor: A single scalar loss value.
        """
        # Calculate mean and std of the L-Scores for the generated batch
        # Add a small epsilon for numerical stability in std calculation
        batch_mean = torch.mean(batch_l_scores, dim=0)
        mean_loss = self.l1_loss(batch_mean, self.target_mean)

        # CRITICAL FIX: Cannot compute std dev for batch size of 1. It results in NaN.
        if batch_l_scores.shape[0] > 1:
            batch_std = torch.std(batch_l_scores, dim=0)
            std_loss = self.l1_loss(batch_std, self.target_std)
        else:
            # For a batch size of 1, we can't learn anything about distribution variance.
            std_loss = torch.tensor(0.0, device=batch_l_scores.device)
        
        # The total distribution loss is the sum of the two
        return mean_loss + std_loss

# ----------------------------------------------------------------------------
# Example Usage and Sanity Check
# ----------------------------------------------------------------------------

if __name__ == '__main__':
    # This block allows you to run `python src/loss.py` to test the modules
    
    # --- Configuration ---
    SAMPLE_RATE = 24000
    N_FFT = 1024
    HOP_LENGTH = 256
    NUM_MELS = 128
    DURATION_S = 4
    SEQ_LENGTH_FRAMES = int(DURATION_S * SAMPLE_RATE / HOP_LENGTH) + 1
    BATCH_SIZE = 2

    # --- Create dummy data ---
    dummy_spectrograms = torch.randn(BATCH_SIZE, NUM_MELS, SEQ_LENGTH_FRAMES)
    print(f"Input spectrograms shape: {dummy_spectrograms.shape}")

    # --- Test LScoreCalculator ---
    print("\n--- Testing LScoreCalculator ---")
    l_score_calculator = LScoreCalculator(sample_rate=SAMPLE_RATE, n_fft=N_FFT, hop_length=HOP_LENGTH)
    l_scores = l_score_calculator(dummy_spectrograms)
    print(f"Calculated L-Scores shape: {l_scores.shape}")
    print(f"Example L-Scores:\n{l_scores}")
    assert l_scores.shape == (BATCH_SIZE, 4)
    print("LScoreCalculator test successful!")

    # --- Test LScoreDistributionLoss ---
    print("\n--- Testing LScoreDistributionLoss ---")
    # These would be pre-computed from your entire IDM dataset
    target_mean_l_score = torch.tensor([0.6, 10.5, 0.1, 0.8])
    target_std_l_score = torch.tensor([0.1, 2.0, 0.05, 0.15])

    loss_fn = LScoreDistributionLoss(target_mean=target_mean_l_score, target_std=target_std_l_score)
    
    # Compute loss
    distribution_loss = loss_fn(l_scores)
    print(f"Calculated Distribution Loss: {distribution_loss.item()}")
    assert distribution_loss.item() > 0
    print("LScoreDistributionLoss test successful!")