"""
Core script for running neurofeedback simulations with different NFT protocols.

This module exposes a programmatic entry point (`run_simulation`) that can be
reused by protocol-specific launchers (e.g., protocol_zoefel_2011.py) and a CLI
for ad-hoc runs. Results are written to timestamped folders under
`results/{protocol}/`.
"""

import argparse
from datetime import datetime
from pathlib import Path
from typing import Dict, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

from modules.NeurofeedbackSimulation import NeurofeedbackSimulation
from modules.SpectralAnalyzer import SpectralAnalyzer


def create_visualizations(results: dict, output_dir: Path = Path('.')) -> None:
    """
    Create comprehensive visualizations of simulation results.
    
    Generates figures matching Davelaar (2018) Figures 4 and 5:
    - Figure 4A: EEG time series (before/after training)
    - Figure 4B: Power spectral density (before/after)
    - Figure 5: UAF distributions (histograms before/after)
    - Learning curves: Target probability and feedback rate
    
    Args:
        results: Dictionary containing simulation results
        output_dir: Directory to save plots
    """
    # Create figure with subplots
    fig = plt.figure(figsize=(16, 12))
    
    # === Figure 4A: EEG Time Series ===
    ax1 = plt.subplot(3, 3, 1)
    # Before training (last 2 seconds of baseline)
    baseline_eeg = results['baseline_eeg']
    time_before = np.arange(len(baseline_eeg[-2000:])) / 1000.0  # Convert ms to seconds
    ax1.plot(time_before, baseline_eeg[-2000:], 'b-', alpha=0.7, linewidth=0.5, label='Before')
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('EEG Signal')
    ax1.set_title('EEG Time Series (Before Training)')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    ax2 = plt.subplot(3, 3, 2)
    # After training (last 2 seconds of post)
    post_eeg = results['post_eeg']
    time_after = np.arange(len(post_eeg[-2000:])) / 1000.0
    ax2.plot(time_after, post_eeg[-2000:], 'r-', alpha=0.7, linewidth=0.5, label='After')
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('EEG Signal')
    ax2.set_title('EEG Time Series (After Training)')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # === Figure 4B: Power Spectral Density ===
    ax3 = plt.subplot(3, 3, 3)
    analyzer = SpectralAnalyzer()
    
    # Import scipy signal for filtering
    from scipy import signal as sp_signal
    
    # Design bandpass filter (1-30 Hz, Butterworth, 4th order)
    nyquist = 1000.0 / 2.0
    low_norm = 1.0 / nyquist
    high_norm = 30.0 / nyquist
    b, a = sp_signal.butter(4, [low_norm, high_norm], btype='band')
    
    # Compute spectra
    # Before: last 1024 samples of baseline
    window_before = baseline_eeg[-1024:]
    # Normalize EEG signal: scale by number of excitatory neurons (800)
    window_before = window_before / 800.0
    # Remove DC component (mean) to avoid 0 Hz peak
    window_before = window_before - np.mean(window_before)
    # Apply bandpass filter (1-30 Hz) to reduce harmonics
    window_before = sp_signal.filtfilt(b, a, window_before)
    windowed_before = window_before * np.hamming(1024)
    fft_before = np.fft.rfft(windowed_before)
    psd_before = np.abs(fft_before)**2 / 1024
    freqs = np.fft.rfftfreq(1024, d=1/1000)
    
    # After: last 1024 samples of post
    window_after = post_eeg[-1024:]
    # Normalize EEG signal: scale by number of excitatory neurons (800)
    window_after = window_after / 800.0
    # Remove DC component (mean) to avoid 0 Hz peak
    window_after = window_after - np.mean(window_after)
    windowed_after = window_after * np.hamming(1024)
    fft_after = np.fft.rfft(windowed_after)
    psd_after = np.abs(fft_after)**2 / 1024
    
    # Plot spectra
    ax3.plot(freqs, psd_before, 'b-', linewidth=2, label='Before', alpha=0.7)
    ax3.plot(freqs, psd_after, 'r-', linewidth=2, label='After', alpha=0.7)
    ax3.set_xlabel('Frequency (Hz)')
    ax3.set_ylabel('Power Spectral Density')
    ax3.set_title('Power Spectrum (Before vs After)')
    ax3.set_xlim(0, 30)
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    
    # Highlight alpha band
    ax3.axvspan(8, 12, alpha=0.1, color='gray', label='Alpha band')
    
    # === Figure 5: UAF Distributions (Density Plot) ===
    ax4 = plt.subplot(3, 3, 4)
    baseline_uaf = results['baseline_uaf']
    post_uaf = results['post_uaf']
    
    # Create density plot (KDE) matching Davelaar paper Figure 5
    # Define range for density estimation
    uaf_min = min(np.min(baseline_uaf), np.min(post_uaf))
    uaf_max = max(np.max(baseline_uaf), np.max(post_uaf))
    uaf_range = np.linspace(uaf_min, uaf_max, 500)
    
    # Compute KDE for both distributions
    kde_baseline = stats.gaussian_kde(baseline_uaf)
    kde_post = stats.gaussian_kde(post_uaf)
    
    # Evaluate KDE at range points
    density_baseline = kde_baseline(uaf_range)
    density_post = kde_post(uaf_range)
    
    # Normalize to relative frequency (density already normalized to integrate to 1)
    # For relative frequency, we need to scale by bin width or use density=True
    # The paper shows "relative frequency" which is essentially the PDF scaled appropriately
    ax4.plot(uaf_range, density_baseline, 'b-', linewidth=2, label='Before', alpha=0.8)
    ax4.plot(uaf_range, density_post, 'r-', linewidth=2, label='After', alpha=0.8)
    
    # Add vertical line at baseline mean (threshold)
    baseline_mean = np.mean(baseline_uaf)
    ax4.axvline(x=baseline_mean, color='black', linewidth=2, linestyle='-', alpha=0.7)
    
    ax4.set_xlabel('UAF values')
    ax4.set_ylabel('Relative frequency')
    ax4.set_title('A. Simulation result')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    ax4.set_xlim(uaf_min, uaf_max)
    
    # Add statistics text
    stats_text = (
        f"Before: μ={np.mean(baseline_uaf):.1f}, σ={np.std(baseline_uaf):.1f}\n"
        f"After:  μ={np.mean(post_uaf):.1f}, σ={np.std(post_uaf):.1f}"
    )
    ax4.text(0.05, 0.95, stats_text, transform=ax4.transAxes,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # === Learning Curve: Target Probability ===
    ax5 = plt.subplot(3, 3, 5)
    target_probs = results['target_prob_history']
    update_indices = np.arange(len(target_probs))
    ax5.plot(update_indices, target_probs, 'g-', linewidth=2)
    ax5.set_xlabel('Feedback Cycle')
    ax5.set_ylabel('P(Target MSN)')
    ax5.set_title('Learning Curve: Target Probability')
    ax5.grid(True, alpha=0.3)
    ax5.set_yscale('log')
    
    # Add initial and final values
    ax5.axhline(y=results['initial_target_prob'], color='b', linestyle='--', alpha=0.5, label='Initial')
    ax5.axhline(y=results['final_target_prob'], color='r', linestyle='--', alpha=0.5, label='Final')
    ax5.legend()
    
    # === Feedback Rate Over Time ===
    ax6 = plt.subplot(3, 3, 6)
    feedback_history = results['feedback_history']
    if len(feedback_history) > 0:
        # Compute moving average
        window_size = max(10, len(feedback_history) // 20)
        moving_avg = np.convolve(feedback_history.astype(float), 
                                 np.ones(window_size)/window_size, mode='valid')
        ax6.plot(moving_avg, 'purple', linewidth=2, label='Moving average')
        ax6.axhline(y=np.mean(feedback_history), color='r', linestyle='--', 
                   label=f'Mean: {np.mean(feedback_history):.1%}')
        ax6.set_xlabel('Feedback Cycle')
        ax6.set_ylabel('Feedback Rate')
        ax6.set_title('Feedback Rate Over Time')
        ax6.set_ylim(0, 1)
        ax6.grid(True, alpha=0.3)
        ax6.legend()
    
    # === UAF Power During Training ===
    ax7 = plt.subplot(3, 3, 7)
    if len(results['uaf_power_history']) > 0:
        uaf_history = results['uaf_power_history']
        ax7.plot(uaf_history, 'orange', linewidth=1.5, alpha=0.7)
        ax7.axhline(y=results['baseline_mean'], color='b', linestyle='--', 
                   label=f'Baseline: {results["baseline_mean"]:.1f}')
        ax7.set_xlabel('Feedback Cycle')
        ax7.set_ylabel('UAF Power')
        ax7.set_title('UAF Power During Training')
        ax7.grid(True, alpha=0.3)
        ax7.legend()
    
    # === Summary Statistics ===
    ax8 = plt.subplot(3, 3, 8)
    ax8.axis('off')
    
    summary_text = (
        "SIMULATION SUMMARY\n"
        "==================\n\n"
        f"Peak Alpha Frequency: {results['paf']:.2f} Hz\n"
        f"UAF Band: [{results['uaf_band'][0]:.2f}, {results['uaf_band'][1]:.2f}] Hz\n\n"
        f"Baseline UAF:\n"
        f"  Mean: {results['baseline_uaf_mean']:.2f}\n"
        f"  Std:  {results['baseline_uaf_std']:.2f}\n\n"
        f"Post-Training UAF:\n"
        f"  Mean: {results['post_uaf_mean']:.2f}\n"
        f"  Std:  {results['post_uaf_std']:.2f}\n\n"
        f"Change: {results['uaf_change']:+.2f} ({results['uaf_change_pct']:+.1f}%)\n\n"
        f"Target Probability:\n"
        f"  Initial: {results['initial_target_prob']:.4f}\n"
        f"  Final:   {results['final_target_prob']:.4f}\n"
        f"  Increase: {results['final_target_prob'] / results['initial_target_prob']:.1f}×\n\n"
        f"Feedback Rate: {results['feedback_rate']:.1%}\n\n"
        f"Learning Success: {'YES' if results['learning_success'] else 'NO'}\n"
        f"Target MSN: {results['target_index']}"
    )
    
    ax8.text(0.1, 0.9, summary_text, transform=ax8.transAxes,
             fontsize=9, verticalalignment='top', family='monospace',
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
    
    # === UAF Distribution Comparison (Box Plot) ===
    ax9 = plt.subplot(3, 3, 9)
    box_data = [baseline_uaf, post_uaf]
    bp = ax9.boxplot(box_data, tick_labels=['Before', 'After'], patch_artist=True)
    bp['boxes'][0].set_facecolor('blue')
    bp['boxes'][1].set_facecolor('red')
    for box in bp['boxes']:
        box.set_alpha(0.6)
    ax9.set_ylabel('UAF Power')
    ax9.set_title('UAF Distribution Comparison')
    ax9.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    # Save figure
    output_path = output_dir / 'simulation_results.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nPlots saved to: {output_path}")
    
    plt.close()


def save_eeg_signals(results: Dict, output_dir: Path) -> Path:
    """Persist raw EEG signals first so critical data is saved even if later steps fail."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    eeg_path = output_dir / 'eeg_signals.npz'
    np.savez_compressed(
        eeg_path,
        baseline=results['baseline_eeg'],
        training=results['training_eeg'],
        post=results['post_eeg']
    )
    print(f"EEG signals saved to: {eeg_path}")
    return eeg_path


def save_results(results: Dict, output_dir: Path = Path('.')) -> Tuple[Path, Path]:
    """
    Save simulation results to files. EEG signals should already be saved separately.
    
    Args:
        results: Dictionary containing simulation results
        output_dir: Directory to save files
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save full NumPy archive
    npz_path = output_dir / 'results.npz'
    np.savez_compressed(npz_path, **results)
    print(f"Results saved to: {npz_path}")
    
    # Save learning curve CSV
    csv_path = output_dir / 'learning_curve.csv'
    target_probs = results['target_prob_history']
    np.savetxt(csv_path, target_probs, delimiter=',', header='target_probability', comments='')
    print(f"Learning curve saved to: {csv_path}")
    
    return npz_path, csv_path


def build_output_dir(protocol_name: str, output_root: Path = Path('results')) -> Path:
    """Create a timestamped output directory for a protocol."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(output_root) / protocol_name / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def run_simulation(
    protocol_name: str,
    seed: int = 42,
    output_root: Union[Path, str] = Path('results'),
    baseline_duration: float = 5 * 60,
    training_duration: float = 5 * 60,
    n_training_phases: int = 5,
    post_duration: float = 5 * 60,
    update_interval: float = 0.1,
    warmup_duration: float = 1.0,
    feedback_threshold_offset: float = 0.5,
    verbose: bool = True
) -> Tuple[Dict, Path]:
    """
    Run the neurofeedback simulation for a given protocol and persist results.
    
    Returns:
        (results_dict, output_dir) where output_dir is the timestamped results folder.
    """
    output_dir = build_output_dir(protocol_name, Path(output_root))
    print(f"Results will be saved to: {output_dir.absolute()}")
    
    simulation = NeurofeedbackSimulation(random_seed=seed)
    results = simulation.run_training_protocol(
        baseline_duration=baseline_duration,
        training_duration=training_duration,
        n_training_phases=n_training_phases,
        post_duration=post_duration,
        update_interval=update_interval,
        warmup_duration=warmup_duration,
        feedback_threshold_offset=feedback_threshold_offset,
        verbose=verbose
    )
    
    save_eeg_signals(results, output_dir)
    save_results(results, output_dir=output_dir)
    create_visualizations(results, output_dir=output_dir)
    
    return results, output_dir


def parse_args() -> argparse.Namespace:
    """CLI argument parser for the core simulation runner."""
    parser = argparse.ArgumentParser(
        description='Run neurofeedback simulation with a specified protocol'
    )
    parser.add_argument(
        '--protocol',
        required=True,
        help='Protocol identifier used for naming the results folder (e.g., zoefel_2011)'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility (default: 42)'
    )
    parser.add_argument(
        '--output-root', '--output',
        dest='output_root',
        type=str,
        default='results',
        help='Root directory for results (default: results/)'
    )
    parser.add_argument(
        '--baseline-duration',
        type=float,
        default=5 * 60,
        help='Baseline duration in seconds (default: 300 = 5 minutes)'
    )
    parser.add_argument(
        '--training-duration',
        type=float,
        default=5 * 60,
        help='Training duration per phase in seconds (default: 300 = 5 minutes)'
    )
    parser.add_argument(
        '--n-training-phases',
        type=int,
        default=5,
        help='Number of training phases (default: 5)'
    )
    parser.add_argument(
        '--post-duration',
        type=float,
        default=5 * 60,
        help='Post-training duration in seconds (default: 300 = 5 minutes)'
    )
    parser.add_argument(
        '--update-interval',
        type=float,
        default=0.1,
        help='Feedback update interval in seconds (default: 0.1 = 100 ms)'
    )
    parser.add_argument(
        '--warmup-duration',
        type=float,
        default=1.0,
        help='Warmup duration in seconds (default: 1.0)'
    )
    parser.add_argument(
        '--feedback-threshold-offset',
        type=float,
        default=0.5,
        help='Feedback threshold offset in std units (baseline_mean + offset*std, default: 0.5)'
    )
    return parser.parse_args()


def main():
    """Main function to run simulation from CLI."""
    args = parse_args()
    run_simulation(
        protocol_name=args.protocol,
        seed=args.seed,
        output_root=args.output_root,
        baseline_duration=args.baseline_duration,
        training_duration=args.training_duration,
        n_training_phases=args.n_training_phases,
        post_duration=args.post_duration,
        update_interval=args.update_interval,
        warmup_duration=args.warmup_duration,
        feedback_threshold_offset=args.feedback_threshold_offset,
        verbose=True
    )
    print("\nSimulation complete!")


if __name__ == '__main__':
    main()
