"""
Test script for EEG generator visualization.

This script tests the Izhikevich EEG generator and plots the generated signal.
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
from pathlib import Path

# Add parent directory to path to import modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from modules.IzhikevichEEGGenerator import IzhikevichEEGGenerator

def test_eeg_generator(duration_ms=5000, random_seed=42):
    """
    Test the EEG generator and plot the signal.
    
    Args:
        duration_ms: Duration of simulation in milliseconds
        random_seed: Random seed for reproducibility
    """
    print("Initializing EEG Generator...")
    print(f"  Duration: {duration_ms} ms ({duration_ms/1000:.1f} seconds)")
    print(f"  Random seed: {random_seed}")
    
    # Create EEG generator
    eeg_gen = IzhikevichEEGGenerator(random_seed=random_seed)
    
    # Warm up the network to reduce initialization transients
    print("\nWarming up network (1000 ms)...")
    eeg_gen.warmup(duration_ms=1000)
    
    # Collect EEG signal
    eeg_signal = []
    time_points = []
    
    print("Running simulation...")
    for t in range(duration_ms):
        eeg_value = eeg_gen.step(thalamic_modulation=0.0)
        eeg_signal.append(eeg_value)
        time_points.append(t)
        
        if (t + 1) % 1000 == 0:
            print(f"  Progress: {t+1}/{duration_ms} ms")
    
    eeg_signal = np.array(eeg_signal)
    time_points = np.array(time_points) / 1000.0  # Convert to seconds
    
    # Create plot
    fig, axes = plt.subplots(2, 1, figsize=(14, 8))
    
    # Plot 1: Full signal
    axes[0].plot(time_points, eeg_signal, 'b-', linewidth=0.5, alpha=0.7)
    axes[0].set_xlabel('Time (s)')
    axes[0].set_ylabel('EEG Signal')
    axes[0].set_title(f'EEG Signal - Full Duration ({duration_ms/1000:.1f} seconds)')
    axes[0].grid(True, alpha=0.3)
    
    # Plot 2: Zoomed view (last 2 seconds)
    zoom_start = max(0, len(eeg_signal) - 2000)
    axes[1].plot(time_points[zoom_start:], eeg_signal[zoom_start:], 'r-', linewidth=0.8, alpha=0.8)
    axes[1].set_xlabel('Time (s)')
    axes[1].set_ylabel('EEG Signal')
    axes[1].set_title('EEG Signal - Last 2 Seconds (Zoomed)')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    output_path = 'results/eeg_generator_test.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nPlot saved to: {output_path}")
    
    # Print statistics
    print("\nSignal Statistics:")
    print(f"  Mean: {np.mean(eeg_signal):.2f}")
    print(f"  Std:  {np.std(eeg_signal):.2f}")
    print(f"  Min:  {np.min(eeg_signal):.2f}")
    print(f"  Max:  {np.max(eeg_signal):.2f}")
    print(f"  Range: {np.max(eeg_signal) - np.min(eeg_signal):.2f}")
    
    # Don't show plot interactively, just save it
    plt.close()
    
    return eeg_signal, time_points

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Test EEG generator')
    parser.add_argument('--duration', type=int, default=5000, 
                       help='Duration in milliseconds (default: 5000)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed (default: 42)')
    
    args = parser.parse_args()
    
    test_eeg_generator(duration_ms=args.duration, random_seed=args.seed)

