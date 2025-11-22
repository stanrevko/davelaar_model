"""
Script to generate plots showing PAF (Peak Alpha Frequency) and Alpha Band Power.

Usage:
    python plot_paf_alpha.py [--input FILE] [--duration SEC] [--output OUTPUT_DIR]
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import signal as sp_signal
from modules.NeurofeedbackSimulation import NeurofeedbackSimulation
from modules.SpectralAnalyzer import SpectralAnalyzer
from tqdm import tqdm


def compute_alpha_band_power(eeg_signal, sampling_rate=1000.0, alpha_band=(8.0, 12.0), window_size=1024):
    """
    Compute alpha band power over time using sliding windows.
    
    Args:
        eeg_signal: EEG signal array
        sampling_rate: Sampling rate in Hz
        alpha_band: Alpha frequency band (low, high) in Hz
        window_size: Window size in samples
        
    Returns:
        Tuple of (time_points, alpha_power_values)
    """
    n_samples = len(eeg_signal)
    n_windows = n_samples - window_size + 1
    alpha_power = []
    time_points = []
    
    for i in range(0, n_windows, window_size // 4):  # 75% overlap
        window = eeg_signal[i:i+window_size]
        
        # Normalize by number of neurons
        window = window / 800.0
        
        # Remove DC
        window = window - np.mean(window)
        
        # Apply Hamming window
        windowed = window * np.hamming(window_size)
        
        # Compute FFT
        fft_result = np.fft.rfft(windowed)
        psd = np.abs(fft_result)**2 / window_size
        freqs = np.fft.rfftfreq(window_size, d=1/sampling_rate)
        
        # Find alpha band
        alpha_mask = (freqs >= alpha_band[0]) & (freqs <= alpha_band[1])
        alpha_power_window = np.sum(psd[alpha_mask])
        
        alpha_power.append(alpha_power_window)
        time_points.append(i / sampling_rate)
    
    return np.array(time_points), np.array(alpha_power)


def create_paf_alpha_plots(eeg_signal, paf, uaf_band, output_dir, title_prefix="Baseline"):
    """
    Create comprehensive plots showing PAF and alpha band power.
    
    Args:
        eeg_signal: EEG signal array
        paf: Peak alpha frequency in Hz
        uaf_band: Upper alpha frequency band (low, high) in Hz
        output_dir: Directory to save plots
        title_prefix: Prefix for plot titles
    """
    sampling_rate = 1000.0
    window_size = 1024
    alpha_band = (8.0, 12.0)
    
    # Normalize EEG signal
    eeg_normalized = eeg_signal / 800.0
    
    # Remove DC component
    eeg_normalized = eeg_normalized - np.mean(eeg_normalized)
    
    # Design bandpass filter (1-30 Hz, Butterworth, 4th order)
    nyquist = sampling_rate / 2.0
    low_norm = 1.0 / nyquist
    high_norm = 30.0 / nyquist
    b, a = sp_signal.butter(4, [low_norm, high_norm], btype='band')
    
    # Apply bandpass filter
    eeg_filtered = sp_signal.filtfilt(b, a, eeg_normalized)
    
    # Use last window_size samples for spectrum
    window_data = eeg_filtered[-window_size:]
    windowed = window_data * np.hamming(window_size)
    
    # Compute power spectral density
    fft_result = np.fft.rfft(windowed)
    psd = np.abs(fft_result)**2 / window_size
    freqs = np.fft.rfftfreq(window_size, d=1/sampling_rate)
    
    # Compute alpha band power over time
    time_points, alpha_power = compute_alpha_band_power(
        eeg_signal, sampling_rate, alpha_band, window_size
    )
    
    # Create figure with subplots
    fig = plt.figure(figsize=(16, 10))
    
    # === Plot 1: Power Spectral Density with PAF ===
    ax1 = plt.subplot(2, 3, 1)
    ax1.plot(freqs, psd, 'b-', linewidth=2, alpha=0.8)
    ax1.axvline(x=paf, color='r', linestyle='--', linewidth=2, label=f'PAF: {paf:.2f} Hz')
    ax1.axvspan(alpha_band[0], alpha_band[1], alpha=0.2, color='gray', label='Alpha band (8-12 Hz)')
    ax1.axvspan(uaf_band[0], uaf_band[1], alpha=0.3, color='orange', label=f'UAF band ({uaf_band[0]:.2f}-{uaf_band[1]:.2f} Hz)')
    ax1.set_xlabel('Frequency (Hz)')
    ax1.set_ylabel('Power Spectral Density')
    ax1.set_title(f'{title_prefix}: Power Spectrum with PAF')
    ax1.set_xlim(0, 30)
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # === Plot 2: Zoomed Alpha Band ===
    ax2 = plt.subplot(2, 3, 2)
    alpha_mask = (freqs >= 5) & (freqs <= 15)
    ax2.plot(freqs[alpha_mask], psd[alpha_mask], 'b-', linewidth=2, alpha=0.8)
    ax2.axvline(x=paf, color='r', linestyle='--', linewidth=2, label=f'PAF: {paf:.2f} Hz')
    ax2.axvspan(alpha_band[0], alpha_band[1], alpha=0.2, color='gray', label='Alpha band')
    ax2.axvspan(uaf_band[0], uaf_band[1], alpha=0.3, color='orange', label='UAF band')
    ax2.set_xlabel('Frequency (Hz)')
    ax2.set_ylabel('Power Spectral Density')
    ax2.set_title(f'{title_prefix}: Alpha Band Detail')
    ax2.set_xlim(5, 15)
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # === Plot 3: Alpha Band Power Over Time ===
    ax3 = plt.subplot(2, 3, 3)
    ax3.plot(time_points, alpha_power, 'g-', linewidth=1.5, alpha=0.7)
    ax3.axhline(y=np.mean(alpha_power), color='r', linestyle='--', 
                label=f'Mean: {np.mean(alpha_power):.2f}')
    ax3.fill_between(time_points, 
                     np.mean(alpha_power) - np.std(alpha_power),
                     np.mean(alpha_power) + np.std(alpha_power),
                     alpha=0.2, color='gray', label=f'±1 SD: {np.std(alpha_power):.2f}')
    ax3.set_xlabel('Time (s)')
    ax3.set_ylabel('Alpha Band Power (8-12 Hz)')
    ax3.set_title(f'{title_prefix}: Alpha Power Over Time')
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    
    # === Plot 4: EEG Time Series (last 2 seconds) ===
    ax4 = plt.subplot(2, 3, 4)
    last_2s = int(2 * sampling_rate)
    time_axis = np.arange(len(eeg_signal[-last_2s:])) / sampling_rate
    ax4.plot(time_axis, eeg_signal[-last_2s:], 'b-', linewidth=0.5, alpha=0.7)
    ax4.set_xlabel('Time (s)')
    ax4.set_ylabel('EEG Signal')
    ax4.set_title(f'{title_prefix}: EEG Time Series (Last 2s)')
    ax4.grid(True, alpha=0.3)
    
    # === Plot 5: Alpha Power Distribution ===
    ax5 = plt.subplot(2, 3, 5)
    ax5.hist(alpha_power, bins=50, alpha=0.7, color='green', edgecolor='black')
    ax5.axvline(x=np.mean(alpha_power), color='r', linestyle='--', linewidth=2,
                label=f'Mean: {np.mean(alpha_power):.2f}')
    ax5.axvline(x=np.median(alpha_power), color='orange', linestyle='--', linewidth=2,
                label=f'Median: {np.median(alpha_power):.2f}')
    ax5.set_xlabel('Alpha Band Power')
    ax5.set_ylabel('Frequency')
    ax5.set_title(f'{title_prefix}: Alpha Power Distribution')
    ax5.grid(True, alpha=0.3, axis='y')
    ax5.legend()
    
    # === Plot 6: Summary Statistics ===
    ax6 = plt.subplot(2, 3, 6)
    ax6.axis('off')
    
    # Compute band powers
    alpha_mask_psd = (freqs >= alpha_band[0]) & (freqs <= alpha_band[1])
    uaf_mask_psd = (freqs >= uaf_band[0]) & (freqs <= uaf_band[1])
    
    alpha_power_total = np.sum(psd[alpha_mask_psd])
    uaf_power_total = np.sum(psd[uaf_mask_psd])
    
    summary_text = (
        f"{title_prefix.upper()} STATISTICS\n"
        f"{'='*40}\n\n"
        f"Peak Alpha Frequency (PAF):\n"
        f"  {paf:.2f} Hz\n\n"
        f"Alpha Band (8-12 Hz):\n"
        f"  Total Power: {alpha_power_total:.4f}\n"
        f"  Mean Power: {np.mean(alpha_power):.4f}\n"
        f"  Std Power:  {np.std(alpha_power):.4f}\n"
        f"  Min Power:  {np.min(alpha_power):.4f}\n"
        f"  Max Power:  {np.max(alpha_power):.4f}\n\n"
        f"Upper Alpha Band ({uaf_band[0]:.2f}-{uaf_band[1]:.2f} Hz):\n"
        f"  Total Power: {uaf_power_total:.4f}\n\n"
        f"Signal Statistics:\n"
        f"  Duration: {len(eeg_signal)/sampling_rate:.1f} s\n"
        f"  Mean: {np.mean(eeg_signal):.2f}\n"
        f"  Std:  {np.std(eeg_signal):.2f}"
    )
    
    ax6.text(0.1, 0.9, summary_text, transform=ax6.transAxes,
             fontsize=10, verticalalignment='top', family='monospace',
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
    
    plt.tight_layout()
    
    # Save figure
    output_path = output_dir / f'{title_prefix.lower()}_paf_alpha_plots.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nPlots saved to: {output_path}")
    
    plt.close()


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description='Generate plots showing PAF and Alpha Band Power'
    )
    parser.add_argument(
        '--input',
        type=str,
        default=None,
        help='Input .npz file with baseline data (if None, will run baseline)'
    )
    parser.add_argument(
        '--duration',
        type=float,
        default=30,
        help='Baseline duration in seconds if running new baseline (default: 30)'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility (default: 42)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='results',
        help='Output directory for plots (default: results/)'
    )
    parser.add_argument(
        '--warmup-duration',
        type=float,
        default=1.0,
        help='Warmup duration in seconds (default: 1.0)'
    )
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load or generate baseline data
    if args.input and Path(args.input).exists():
        print(f"Loading baseline data from: {args.input}")
        data = np.load(args.input)
        baseline_eeg = data['baseline_eeg']
        paf = float(data['paf'])
        uaf_band = tuple(data['uaf_band'])
        print(f"Loaded: PAF={paf:.2f} Hz, UAF band={uaf_band}")
    else:
        print("Running baseline recording...")
        # Initialize simulation
        simulation = NeurofeedbackSimulation(random_seed=args.seed)
        
        # Convert durations to milliseconds
        baseline_duration_ms = int(args.duration * 1000)
        warmup_duration_ms = int(args.warmup_duration * 1000)
        
        # Warmup
        if warmup_duration_ms > 0:
            print(f"Warming up EEG generator ({args.warmup_duration:.1f} s)...")
            simulation.eeg_generator.warmup(duration_ms=warmup_duration_ms)
            print("✓ Warmup complete")
        
        # Baseline recording
        print(f"Recording baseline ({args.duration:.1f} s)...")
        baseline_eeg = []
        
        for t in tqdm(
            range(baseline_duration_ms),
            desc="Baseline",
            unit="ms"
        ):
            eeg_sample = simulation.eeg_generator.step(thalamic_modulation=0.0)
            baseline_eeg.append(eeg_sample)
        
        baseline_eeg = np.array(baseline_eeg)
        
        # Establish baseline statistics
        paf = simulation.analyzer.find_peak_alpha_frequency(baseline_eeg)
        baseline_mean, baseline_std = simulation.analyzer.set_baseline(baseline_eeg, paf=paf)
        uaf_band = simulation.analyzer.get_uaf_band()
        
        print(f"✓ Baseline established: PAF={paf:.2f} Hz, UAF band={uaf_band}")
    
    # Generate plots
    print("\nGenerating plots...")
    create_paf_alpha_plots(
        baseline_eeg, paf, uaf_band, output_dir, title_prefix="Baseline"
    )
    
    print("\nPlot generation complete!")


if __name__ == '__main__':
    main()

