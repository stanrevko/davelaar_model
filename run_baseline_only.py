"""
Script to run only the baseline phase of the neurofeedback simulation.

Usage:
    python run_baseline_only.py [--seed SEED] [--duration SEC] [--output OUTPUT_DIR]
"""

import argparse
import numpy as np
from pathlib import Path
from modules.NeurofeedbackSimulation import NeurofeedbackSimulation
from tqdm import tqdm


def main():
    """Main function to run baseline only."""
    parser = argparse.ArgumentParser(
        description='Run baseline phase only'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility (default: 42)'
    )
    parser.add_argument(
        '--duration',
        type=float,
        default=30,
        help='Baseline duration in seconds (default: 30)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='results',
        help='Output directory for results (default: results/)'
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
    
    print(f"Results will be saved to: {output_dir.absolute()}")
    
    # Initialize simulation
    simulation = NeurofeedbackSimulation(random_seed=args.seed)
    
    # Convert durations to milliseconds
    baseline_duration_ms = int(args.duration * 1000)
    warmup_duration_ms = int(args.warmup_duration * 1000)
    
    print("=" * 70)
    print("BASELINE RECORDING ONLY")
    print("=" * 70)
    print(f"\nConfiguration:")
    print(f"  MSN units: {simulation.striatum.n_units}")
    print(f"  Cortical neurons: {simulation.eeg_generator.n_exc}E + {simulation.eeg_generator.n_inh}I")
    print(f"  Random seed: {args.seed}")
    print(f"  Baseline duration: {args.duration:.1f} s ({args.duration/60:.1f} min)")
    print()
    
    # Warmup
    if warmup_duration_ms > 0:
        print(f"Warming up EEG generator ({args.warmup_duration:.1f} s)...")
        simulation.eeg_generator.warmup(duration_ms=warmup_duration_ms)
        print("✓ Warmup complete\n")
    
    # Baseline recording
    print("=" * 70)
    print("BASELINE RECORDING")
    print("=" * 70)
    
    baseline_eeg = []
    
    for t in tqdm(
        range(baseline_duration_ms),
        desc="Baseline",
        unit="ms"
    ):
        # Generate EEG without modulation (no feedback, no learning)
        eeg_sample = simulation.eeg_generator.step(thalamic_modulation=0.0)
        baseline_eeg.append(eeg_sample)
    
    baseline_eeg = np.array(baseline_eeg)
    
    # Establish baseline statistics
    paf = simulation.analyzer.find_peak_alpha_frequency(baseline_eeg)
    baseline_mean, baseline_std = simulation.analyzer.set_baseline(baseline_eeg, paf=paf)
    uaf_band = simulation.analyzer.get_uaf_band()
    
    # Compute UAF distribution
    baseline_uaf = simulation.analyzer.compute_uaf_distribution(baseline_eeg)
    
    print(f"\n✓ Baseline established:")
    print(f"    PAF: {paf:.2f} Hz")
    print(f"    UAF band: [{uaf_band[0]:.2f}, {uaf_band[1]:.2f}] Hz")
    print(f"    Baseline UAF: {baseline_mean:.4f} ± {baseline_std:.4f}")
    print(f"    UAF min: {np.min(baseline_uaf):.4f}")
    print(f"    UAF max: {np.max(baseline_uaf):.4f}")
    print()
    
    # Save results
    npz_path = output_dir / 'baseline_results.npz'
    np.savez_compressed(
        npz_path,
        baseline_eeg=baseline_eeg,
        baseline_uaf=baseline_uaf,
        paf=paf,
        uaf_band=uaf_band,
        baseline_mean=baseline_mean,
        baseline_std=baseline_std,
        duration=args.duration,
        seed=args.seed
    )
    print(f"Results saved to: {npz_path}")
    
    print("\nBaseline recording complete!")


if __name__ == '__main__':
    main()

