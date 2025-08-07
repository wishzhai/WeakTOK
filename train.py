print("DEBUG: Script execution started.")
import os
import sys
import argparse
import yaml
import json
import shutil
from datetime import datetime
import torch
from torch.utils.data import DataLoader
import numpy as np

# Add project root to the Python path to resolve module import errors
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

# Local imports from the project
from src.model import WeakTokIDM
from src.data_loader import IDMDataset
from src.loss import LScoreCalculator, LScoreDistributionLoss
from src.utils import set_seed, log_gpu_memory, clear_gpu_memory


def evaluate(model, data_loader, loss_fn_dist, l_score_calc, device, loss_config):
    """
    Reusable evaluation function for both validation and testing.
    """
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch in data_loader:
            input_spectrogram = batch['spectrogram'].to(device)
            
            output_spectrogram = model(input_spectrogram.unsqueeze(1))
            
            batch_l_scores = l_score_calc(output_spectrogram)
            loss = loss_fn_dist(batch_l_scores)
            total_loss += loss.item()
            
    return total_loss / len(data_loader)

def run_trial(config_path):
    """
    Runs a single, isolated training trial using a specified configuration file.
    This is designed for maximum stability and to bypass complex frameworks for debugging.
    """
    print("--- [1/7] Starting Single Trial Runner ---")
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        print(f"    \u2713 Successfully loaded config from: {config_path}")

        results = {
            'config': config,
            'start_time': datetime.now().isoformat(),
            'results': {},
            'status': 'started'
        }
        
        # --- Override config with command-line arguments if provided ---
        if args.learning_rate is not None:
            config['training']['learning_rate'] = args.learning_rate
            print(f"    -> Overrode learning_rate with: {args.learning_rate}")
        if args.batch_size is not None:
            config['training']['batch_size'] = args.batch_size
            print(f"    -> Overrode batch_size with: {args.batch_size}")
    except Exception as e:
        print(f"    \u2717 FATAL: Could not load or parse config file: {e}")
        return

    print("--- [2/7] Creating Experiment Directory ---")
    run_id = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    results_base_dir = config['paths'].get('results_dir', 'results/trials')
    experiment_dir = os.path.join(results_base_dir, run_id)
    os.makedirs(experiment_dir, exist_ok=True)
    shutil.copy(config_path, os.path.join(experiment_dir, 'config.yaml'))
    print(f"    ✓ Artifacts will be saved to: {experiment_dir}")

    print("--- [3/7] Setting up Environment ---")
    print("    -> Initializing seed...")
    set_seed(config['training']['seed'])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"    \u2713 Seed set to {config['training']['seed']}")
    print(f"    \u2713 Device set to {device}")
    print("    -> Environment setup complete.")

    print("--- [3/7] Initializing Model and Optimizer ---")
    print("    -> Initializing WeakTokIDM model...")
    try:
        model = WeakTokIDM(**config['model']).to(device)
        print("    \u2713 Model initialized.")
        print("    -> Initializing Adam optimizer...")
        optimizer = torch.optim.Adam(
            model.parameters(), 
            lr=config['training']['learning_rate']
        )
        print("    \u2713 Optimizer initialized.")
    except Exception as e:
        print(f"    \u2717 FATAL: Failed to initialize model or optimizer: {e}")
        return

    print("--- [5/7] Initializing Loss Modules ---")
    print("    -> Initializing LScoreCalculator...")
    try:
        l_score_calculator = LScoreCalculator(
            sample_rate=config['audio']['sample_rate'],
            n_fft=config['audio']['n_fft'],
            hop_length=config['audio']['hop_length']
        ).to(device)
        print("    \u2713 LScoreCalculator initialized.")
        target_mean = torch.tensor(config['loss']['target_mean'], device=device)
        target_std = torch.tensor(config['loss']['target_std'], device=device)
        distribution_loss_fn = LScoreDistributionLoss(target_mean, target_std).to(device)
        print("    \u2713 LScoreDistributionLoss initialized.")

        # --- Reconstruction Loss ---
        recon_loss_fn = torch.nn.L1Loss().to(device)
        lambda_recon = config['loss'].get('lambda_recon', 0.0) # Default to 0 to not break old configs
        print(f"    \u2713 Reconstruction L1 Loss initialized with lambda = {lambda_recon}")
    except Exception as e:
        print(f"    \u2717 FATAL: Failed to initialize loss modules: {e}")
        return

    print("--- [5/7] Preparing Datasets and DataLoaders ---")
    print("    -> Initializing training dataset...")
    try:
        train_dataset = IDMDataset(
            audio_dir=config['paths']['train_data'],
            chunk_duration_s=config['audio']['chunk_duration_s'],
            sample_rate=config['audio']['sample_rate']
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=config['training']['batch_size'],
            shuffle=True,
            num_workers=0,  # Force single-process for stability
            pin_memory=True
        )
        val_dataset = IDMDataset(
            audio_dir=config['paths']['val_data'],
            chunk_duration_s=config['audio']['chunk_duration_s'],
            sample_rate=config['audio']['sample_rate']
        )
        val_loader = DataLoader(
            val_dataset, batch_size=config['training']['batch_size'], num_workers=0
        )

        test_dataset = IDMDataset(
            audio_dir=config['paths']['test_data'],
            chunk_duration_s=config['audio']['chunk_duration_s'],
            sample_rate=config['audio']['sample_rate']
        )
        test_loader = DataLoader(
            test_dataset, batch_size=config['training']['batch_size'], num_workers=0
        )

        print("    \u2713 Train, Val, and Test DataLoaders prepared successfully.")
    except Exception as e:
        print(f"    \u2717 FATAL: Failed to prepare dataset/loader: {e}")
        return

    print("--- [6/7] Starting Training Loop ---")
    max_consecutive_errors = 10  # Stop if 10 batches in a row fail
    consecutive_errors = 0
    training_failed = False

    best_val_loss = float('inf')

    for epoch in range(config['training']['epochs']):
        if training_failed:
            break

        model.train()
        for i, batch in enumerate(train_loader):
            try:
                optimizer.zero_grad()
                
                input_spectrogram = batch['spectrogram'].to(device)
                output_spectrogram = model(input_spectrogram.unsqueeze(1))

                # --- Calculate Combined Loss ---
                l_score_loss = distribution_loss_fn(l_score_calculator(output_spectrogram))
                recon_loss = recon_loss_fn(output_spectrogram, input_spectrogram.unsqueeze(1))
                loss = l_score_loss + (lambda_recon * recon_loss)
                
                loss.backward()
                optimizer.step()

                consecutive_errors = 0  # Reset error count on success

                if (i + 1) % config['training']['log_interval'] == 0:
                    print(f"    Epoch {epoch+1}, Batch {i+1}/{len(train_loader)}, Loss: {loss.item():.4f}")
            
            except Exception as e:
                print(f"    \u2717 ERROR in training step: {e}")
                consecutive_errors += 1
                if consecutive_errors >= max_consecutive_errors:
                    print(f"    \u2717 FATAL: Exceeded max consecutive errors ({max_consecutive_errors}). Stopping training.")
                    training_failed = True
                    break # Exit the inner loop
                continue

        # --- Validation Step ---
        # Note: Validation loss for now will still only be L-Score loss for consistency in reporting.
        # The effect of reconstruction loss will be seen in the generated audio quality.
        val_loss = evaluate(model, val_loader, distribution_loss_fn, l_score_calculator, device, config['loss'])
        print(f"    Epoch {epoch+1} Validation Loss: {val_loss:.4f}")

        # --- Checkpoint Saving ---
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            checkpoint_path = os.path.join(experiment_dir, 'best_model.pth')
            try:
                torch.save(model.state_dict(), checkpoint_path)
                print(f"    -> Saved new best model to {checkpoint_path} (Val Loss: {best_val_loss:.4f})")
            except Exception as e:
                print(f"    \u2717 ERROR: Could not save checkpoint: {e}")

    if training_failed:
        results['status'] = 'failed'
        results['end_time'] = datetime.now().isoformat()
        print("--- [7/7] Training Halted Due to Errors ---")
        print("    Skipping final evaluation.")
        # Optionally, exit with a non-zero status code to make the shell script aware
        # import sys
        # sys.exit(1)
    else:
        print("--- [7/7] Training Finished ---")
        print("--- [8/8] Final Evaluation on Test Set ---")
        test_loss = evaluate(model, test_loader, distribution_loss_fn, l_score_calculator, device, config['loss'])
        print(f"    >> Final Test Loss: {test_loss:.4f} <<")

        results['results']['test_loss'] = test_loss
        results['end_time'] = datetime.now().isoformat()
        results['status'] = 'completed'

        # --- [9/9] Saving Results ---
        results_path = os.path.join(experiment_dir, 'results.json')
        try:
            with open(results_path, 'w') as f:
                json.dump(results, f, indent=4)
            print(f"    ✓ Results saved to {results_path}")
        except Exception as e:
            print(f"    \u2717 ERROR: Could not save results to JSON: {e}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Run a single, stable training trial for WeakTok-IDM."
    )
    parser.add_argument(
        "--config", 
        type=str, 
        required=True, 
        help="Path to the configuration YAML file."
    )
    # Add command-line overrides for key hyperparameters
    parser.add_argument('--learning_rate', type=float, help='Override learning rate.')
    parser.add_argument('--batch_size', type=int, help='Override batch size.')

    args = parser.parse_args()
    print(f"DEBUG: Parsed arguments: {args}")
    
    run_trial(args.config)
