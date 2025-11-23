"""
Zoefel et al. (2011) protocol launcher.

This script runs the neurofeedback simulation with parameters aligned to the
upper-alpha training protocol described by Zoefel et al. (2011). Results are
stored under `results/zoefel_2011/{timestamp}/`.
"""

import argparse
from pathlib import Path
from typing import Tuple

from run_simulation import run_simulation

PROTOCOL_NAME = "zoefel_2011"


def run_protocol(seed: int, output_root: Path) -> Tuple[dict, Path]:
    """
    Execute the Zoefel 2011 protocol and return (results, output_dir).
    
    The durations reflect a multi-session upper-alpha training schedule
    (baseline/post at 3 minutes, 6 training blocks of 4 minutes each).
    """
    return run_simulation(
        protocol_name=PROTOCOL_NAME,
        seed=seed,
        output_root=output_root,
        baseline_duration=5 * 60,
        training_duration=5 * 60,
        n_training_phases=6,
        post_duration=5 * 60,
        update_interval=0.1,
        warmup_duration=1.0,
        feedback_threshold_offset=0.5,
        verbose=True
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the Zoefel et al. (2011) upper-alpha neurofeedback protocol"
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
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    _, output_dir = run_protocol(seed=args.seed, output_root=Path(args.output_root))
    print(f"\nZoefel 2011 protocol complete. Results saved to: {output_dir}")


if __name__ == "__main__":
    main()
