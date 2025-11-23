"""
Davelaar (2018) protocol launcher.

Runs Simulation Study 1 configuration: 5 min baseline, 5 training sessions
of 5 min each, and 5 min post, with threshold set to baseline mean.
Results are stored under `results/davelaar_2018/{timestamp}/`.
"""

import argparse
from pathlib import Path
from typing import Tuple

from run_simulation import run_simulation

PROTOCOL_NAME = "davelaar_2018"


def run_protocol(seed: int, output_root: Path) -> Tuple[dict, Path]:
    """
    Execute the Davelaar 2018 Simulation Study 1 protocol.
    
    Uses baseline/post at 5 minutes, 5 training blocks of 5 minutes,
    100 ms feedback interval, and threshold at baseline mean (offset=0.0).
    """
    return run_simulation(
        protocol_name=PROTOCOL_NAME,
        seed=seed,
        output_root=output_root,
        baseline_duration=5 * 60,
        training_duration=5 * 60,
        n_training_phases=5,
        post_duration=5 * 60,
        update_interval=0.1,
        warmup_duration=1.0,
        feedback_threshold_offset=0.5,
        verbose=True
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the Davelaar (2018) Simulation Study 1 protocol"
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
    print(f"\nDavelaar 2018 protocol complete. Results saved to: {output_dir}")


if __name__ == "__main__":
    main()
