"""
Neurofeedback Simulation Model.

This module integrates all components (EEG generator, striatal learning, spectral analysis)
to implement the complete neurofeedback training protocol from Davelaar (2018) Simulation Study 1.

Reference:
    Davelaar, E.J. (2018). Mechanisms of Neurofeedback: A Computation-theoretic Approach.
    Neuroscience, 378, 175-188.

Author: Stanislav Revko (stanislav.revko@gmail.com)
Date: November 22, 2025
"""

import numpy as np
from typing import Dict, Optional, Tuple, List
from tqdm import tqdm
import warnings

# Import components - adjust paths based on your project structure
try:
    from .IzhikevichEEGGenerator import IzhikevichEEGGenerator
    from .ActivationBuffer import ActivationBuffer
    from .StriatalLearning import StriatalLearning
    from .SpectralAnalyzer import SpectralAnalyzer
except ImportError:
    # Try direct imports if running as standalone
    try:
        from IzhikevichEEGGenerator import IzhikevichEEGGenerator
        from ActivationBuffer import ActivationBuffer
        from StriatalLearning import StriatalLearning
        from SpectralAnalyzer import SpectralAnalyzer
    except ImportError:
        raise ImportError(
            "Could not import required modules. Please ensure all modules "
            "(IzhikevichEEGGenerator, ActivationBuffer, StriatalLearning, SpectralAnalyzer) "
            "are in the same directory or properly installed."
        )


class NeurofeedbackSimulation:
    """
    Complete neurofeedback simulation integrating all components.
    
    Implements the three-phase training protocol:
    1. Baseline: 5 minutes recording to establish PAF and baseline statistics
    2. Training: 1-60 minutes neurofeedback learning with reward-modulated plasticity
    3. Post-training: 5 minutes recording to measure training effects
    
    The simulation follows Davelaar (2018) and Zoefel et al. (2011) protocols,
    implementing importance sampling at the striatal level to learn EEG state
    control through reward-modulated plasticity.
    
    Attributes:
        eeg_generator (IzhikevichEEGGenerator): EEG signal generator
        striatum (StriatalLearning): Striatal learning system
        analyzer (SpectralAnalyzer): Spectral analysis for feedback
        activation_buffer (ActivationBuffer): Buffer for credit assignment
        random_seed (Optional[int]): Random seed for reproducibility
        results (Dict): Dictionary storing simulation results
        
    Examples:
        >>> # Basic usage
        >>> sim = NeurofeedbackSimulation(random_seed=42)
        >>> results = sim.run_training_protocol(
        ...     baseline_duration=5*60,    # 5 minutes
        ...     training_duration=1*60,    # 1 minute
        ...     post_duration=5*60,        # 5 minutes
        ...     verbose=True
        ... )
        >>> 
        >>> # Check results
        >>> print(f"PAF: {results['paf']:.2f} Hz")
        >>> print(f"Learning success: {results['learning_success']}")
        >>> print(f"Target prob: {results['initial_target_prob']:.4f} → "
        ...       f"{results['final_target_prob']:.4f}")
    """
    
    def __init__(
        self,
        n_exc: int = 800,
        n_inh: int = 200,
        n_msn: int = 1000,
        n_active_expected: float = 10.0,
        learning_rate: float = 0.1,
        random_seed: Optional[int] = None,
        measurement_noise_std: float = 0.0,
        apply_bandpass: bool = False
    ):
        """
        Initialize neurofeedback simulation.
        
        Args:
            n_exc: Number of excitatory neurons (default: 800)
            n_inh: Number of inhibitory neurons (default: 200)
            n_msn: Number of MSN units (default: 1000)
            n_active_expected: Expected number of active MSN units (default: 10)
            learning_rate: Learning rate for striatal weight updates (default: 0.1)
            random_seed: Random seed for reproducibility (None for random)
            measurement_noise_std: Std of additive Gaussian noise on EEG (default: 0.0)
            apply_bandpass: Whether to apply bandpass filtering in spectral analysis
            
        Raises:
            ValueError: If parameters are invalid
        """
        if n_exc <= 0 or n_inh <= 0 or n_msn <= 0:
            raise ValueError("All neuron counts must be positive")
        if learning_rate <= 0:
            raise ValueError(f"learning_rate must be positive, got {learning_rate}")
        if measurement_noise_std < 0:
            raise ValueError(
                f"measurement_noise_std must be non-negative, got {measurement_noise_std}"
            )
        
        self.random_seed = random_seed
        
        # Initialize EEG generator
        self.eeg_generator = IzhikevichEEGGenerator(
            n_exc=n_exc,
            n_inh=n_inh,
            random_seed=random_seed,
            measurement_noise_std=measurement_noise_std
        )
        
        # Initialize striatal learning
        self.striatum = StriatalLearning(
            n_units=n_msn,
            n_active_expected=n_active_expected,
            learning_rate=learning_rate,
            window_size=1024,  # Match feedback window
            random_seed=random_seed
        )
        
        # Initialize spectral analyzer
        self.analyzer = SpectralAnalyzer(
            sampling_rate=1000.0,
            window_size=1024,
            update_interval=100,
            alpha_band=(8.0, 12.0),
            apply_bandpass=apply_bandpass,
            normalize_by_n_neurons=True,
            n_neurons=n_exc
        )
        
        # Activation buffer will be created during training
        self.activation_buffer = None
        
        # Results storage
        self.results = {}
        
        # Training state
        self._is_trained = False
    
    def _validate_protocol_params(
        self,
        baseline_duration: float,
        training_duration: float,
        post_duration: float,
        update_interval: float,
        warmup_duration: float
    ) -> None:
        """
        Validate protocol parameters.
        
        Args:
            baseline_duration: Baseline duration in seconds
            training_duration: Training duration in seconds
            post_duration: Post-training duration in seconds
            update_interval: Feedback interval in seconds
            warmup_duration: Warmup duration in seconds
            
        Raises:
            ValueError: If any parameter is invalid
        """
        if baseline_duration < 1.024:
            raise ValueError(
                f"baseline_duration must be >= 1.024 seconds (1 window), "
                f"got {baseline_duration}"
            )
        
        if training_duration < 1.024:
            raise ValueError(
                f"training_duration must be >= 1.024 seconds, got {training_duration}"
            )
        
        if post_duration < 1.024:
            raise ValueError(
                f"post_duration must be >= 1.024 seconds, got {post_duration}"
            )
        
        if update_interval <= 0 or update_interval > 1.0:
            raise ValueError(
                f"update_interval must be in (0, 1.0] seconds, got {update_interval}"
            )
        
        if warmup_duration < 0:
            raise ValueError(
                f"warmup_duration must be non-negative, got {warmup_duration}"
            )
        
        # Check that update_interval is divisible by 1 ms
        if abs(update_interval * 1000 - round(update_interval * 1000)) > 1e-6:
            warnings.warn(
                f"update_interval ({update_interval} s) not evenly divisible by 1 ms. "
                f"Rounding to {round(update_interval * 1000)} ms.",
                RuntimeWarning
            )
    
    def run_training_protocol(
        self,
        baseline_duration: float = 5 * 60,
        training_duration: float = 1 * 60,
        n_training_phases: int = 1,
        post_duration: float = 5 * 60,
        update_interval: float = 0.1,
        warmup_duration: float = 1.0,
        feedback_threshold_offset: float = 0.0,
        verbose: bool = True
    ) -> Dict:
        """
        Run complete training protocol: baseline + training + post-measurement.
        
        Args:
            baseline_duration: Baseline recording duration in seconds (default: 5 min)
            training_duration: Training duration per phase in seconds (default: 1 min)
            n_training_phases: Number of training phases/sessions (default: 1)
            post_duration: Post-training duration in seconds (default: 5 min)
            update_interval: Feedback update interval in seconds (default: 0.1 = 100 ms)
            warmup_duration: Warmup duration in seconds (default: 1.0 s)
            feedback_threshold_offset: Threshold offset in std units (default: 0.0)
                                      threshold = baseline_mean + offset * baseline_std
            verbose: Whether to print progress information (default: True)
            
        Returns:
            Dictionary containing all simulation results with keys:
                - 'baseline_eeg': Baseline EEG signal
                - 'training_eeg': Training EEG signal
                - 'post_eeg': Post-training EEG signal
                - 'baseline_uaf': Baseline UAF distribution
                - 'post_uaf': Post-training UAF distribution
                - 'target_prob_history': Target probability over time
                - 'feedback_history': Feedback signals
                - 'uaf_power_history': UAF power over time
                - 'learning_success': Boolean indicating success
                - 'paf': Detected peak alpha frequency
                - ... (see full list in code)
                
        Raises:
            ValueError: If parameters are invalid
            
        Examples:
            >>> sim = NeurofeedbackSimulation(random_seed=42)
            >>> 
            >>> # Quick test (30 seconds total)
            >>> results = sim.run_training_protocol(
            ...     baseline_duration=10,
            ...     training_duration=10,
            ...     post_duration=10,
            ...     verbose=True
            ... )
            >>> 
            >>> # Full protocol (Davelaar 2018: 11 minutes total)
            >>> results = sim.run_training_protocol(
            ...     baseline_duration=5*60,
            ...     training_duration=1*60,
            ...     post_duration=5*60,
            ...     verbose=True
            ... )
        """
        # Validate parameters
        self._validate_protocol_params(
            baseline_duration,
            training_duration,
            post_duration,
            update_interval,
            warmup_duration
        )
        
        # Convert durations to milliseconds (internal time step is 1 ms)
        baseline_duration_ms = int(baseline_duration * 1000)
        training_duration_ms = int(training_duration * 1000)
        post_duration_ms = int(post_duration * 1000)
        update_interval_ms = int(round(update_interval * 1000))
        warmup_duration_ms = int(warmup_duration * 1000)
        
        if verbose:
            print("=" * 70)
            print("NEUROFEEDBACK SIMULATION (Davelaar 2018)")
            print("=" * 70)
            print(f"\nSimulation Configuration:")
            print(f"  MSN units: {self.striatum.n_units}")
            print(f"  Expected active MSN: {self.striatum.n_active_expected}")
            print(f"  Cortical neurons: {self.eeg_generator.n_exc}E + {self.eeg_generator.n_inh}I")
            print(f"  Target MSN: {self.striatum.get_target_index()} (hidden from model)")
            print(f"  Learning rate: {self.striatum.learning_rate}")
            print(f"  Random seed: {self.random_seed}")
            print()
            print(f"Protocol:")
            print(f"  Baseline: {baseline_duration:.1f} s ({baseline_duration/60:.1f} min)")
            print(f"  Training: {n_training_phases} sessions × {training_duration:.1f} s ({training_duration/60:.1f} min each)")
            print(f"  Post: {post_duration:.1f} s ({post_duration/60:.1f} min)")
            print(f"  Total: {(baseline_duration + n_training_phases * training_duration + post_duration)/60:.1f} min")
            print(f"  Feedback interval: {update_interval*1000:.0f} ms")
            print()
        
        # ======================================================================
        # WARMUP: Reduce initialization transients
        # ======================================================================
        if warmup_duration > 0:
            if verbose:
                print(f"Warming up EEG generator ({warmup_duration:.1f} s)...")
            self.eeg_generator.warmup(duration_ms=warmup_duration_ms)
            if verbose:
                print("✓ Warmup complete\n")
        
        # ======================================================================
        # PHASE 1: BASELINE RECORDING
        # ======================================================================
        if verbose:
            print("=" * 70)
            print("PHASE 1: BASELINE RECORDING")
            print("=" * 70)

        # Pre-allocate array for better performance
        baseline_eeg = np.empty(baseline_duration_ms, dtype=np.float64)

        for t in tqdm(
            range(baseline_duration_ms),
            desc="Baseline",
            disable=not verbose,
            unit="ms",
            mininterval=0.1
        ):
            # Generate EEG without modulation (no feedback, no learning)
            baseline_eeg[t] = self.eeg_generator.step(thalamic_modulation=0.0)
        
        # Establish baseline statistics
        paf = self.analyzer.find_peak_alpha_frequency(baseline_eeg)
        baseline_mean, baseline_std = self.analyzer.set_baseline(baseline_eeg, paf=paf)
        uaf_band = self.analyzer.get_uaf_band()
        
        # Compute feedback threshold
        feedback_threshold = baseline_mean + feedback_threshold_offset * baseline_std
        
        if verbose:
            print(f"\n✓ Baseline established:")
            print(f"    PAF: {paf:.2f} Hz")
            print(f"    UAF band: [{uaf_band[0]:.2f}, {uaf_band[1]:.2f}] Hz")
            print(f"    Baseline UAF: {baseline_mean:.4f} ± {baseline_std:.4f}")
            print(f"    Feedback threshold: {feedback_threshold:.4f} "
                  f"(mean + {feedback_threshold_offset:.1f}σ)")
            print()
        
        # ======================================================================
        # PHASE 2: TRAINING WITH NEUROFEEDBACK
        # ======================================================================
        if verbose:
            print("=" * 70)
            print(f"PHASE 2: TRAINING ({n_training_phases} sessions × {training_duration:.1f} seconds)")
            print("=" * 70)
        
        # Initialize activation buffer
        self.activation_buffer = ActivationBuffer(
            window_size=1024,
            n_units=self.striatum.n_units
        )

        # Pre-allocate arrays for better performance
        total_training_duration_ms = n_training_phases * training_duration_ms
        training_eeg = np.empty(total_training_duration_ms, dtype=np.float64)

        # Estimate number of feedback updates
        max_updates = (total_training_duration_ms // update_interval_ms) + 1
        feedback_history = []  # Keep as list, append is fine for small counts
        target_prob_history = []
        uaf_power_history = []

        # Track initial target probability
        initial_target_prob = self.striatum.get_target_probability()
        target_prob_history.append(initial_target_prob)
        
        total_n_updates = 0
        total_n_positive_feedback = 0
        
        # Run multiple training phases
        training_idx = 0  # Index for pre-allocated training_eeg array

        for phase in range(n_training_phases):
            if verbose and n_training_phases > 1:
                print(f"\nPhase {phase + 1}/{n_training_phases}: Training Phase {phase + 1}")
                print(f"  Duration: {training_duration:.1f} seconds ({training_duration/60:.1f} minutes)")

            phase_initial_prob = self.striatum.get_target_probability()
            phase_n_updates = 0
            phase_n_positive_feedback = 0

            for t in tqdm(
                range(training_duration_ms),
                desc=f"Training Phase {phase + 1}" if n_training_phases > 1 else "Training",
                disable=not verbose,
                unit="ms",
                mininterval=0.1
            ):
                # Sample striatum
                active_units = self.striatum.sample()
                target_active = self.striatum.is_target_active(active_units)
                self.activation_buffer.add(active_units)

                # Generate EEG with thalamic modulation
                modulation = 1.0 if target_active else 0.0
                eeg_sample = self.eeg_generator.step(thalamic_modulation=modulation)
                training_eeg[training_idx] = eeg_sample
                training_idx += 1

                # Add to analyzer buffer
                self.analyzer.add_sample(eeg_sample)
                
                # Feedback and learning every update_interval ms
                if ((t + 1) % update_interval_ms == 0 and
                    self.activation_buffer.is_full() and
                    self.analyzer.is_ready()):
                    
                    # Compute UAF power
                    uaf_power = self.analyzer.compute_uaf_power()
                    uaf_power_history.append(uaf_power)
                    
                    # Generate feedback signal
                    feedback = self.analyzer.get_feedback(threshold=feedback_threshold)
                    feedback_history.append(feedback)
                    
                    if feedback:
                        phase_n_positive_feedback += 1
                        total_n_positive_feedback += 1
                    
                    # Update striatal weights (reward-modulated plasticity)
                    activation_counts = self.activation_buffer.get_counts()
                    self.striatum.update_weights(feedback, activation_counts)
                    
                    # CRITICAL: Do NOT reset activation_buffer!
                    # It maintains sliding window via circular buffering
                    
                    # Track learning progress
                    target_prob = self.striatum.get_target_probability()
                    target_prob_history.append(target_prob)
                    
                    phase_n_updates += 1
                    total_n_updates += 1

            phase_final_prob = self.striatum.get_target_probability()
            phase_feedback_rate = phase_n_positive_feedback / max(phase_n_updates, 1)

            if verbose and n_training_phases > 1:
                print(f"\nTraining Phase {phase + 1} results:")
                print(f"  Initial P(target): {phase_initial_prob:.4f}")
                print(f"  Final P(target): {phase_final_prob:.4f}")
                print(f"  Increase: {phase_final_prob / max(phase_initial_prob, 1e-10):.1f}×")
                print(f"  Feedback rate: {phase_feedback_rate:.1%}")
                print(f"  Weight updates: {phase_n_updates}")

        # training_eeg already pre-allocated as numpy array
        final_target_prob = self.striatum.get_target_probability()
        overall_feedback_rate = total_n_positive_feedback / max(total_n_updates, 1)
        
        if verbose:
            if n_training_phases > 1:
                print(f"\nOverall Training Summary:")
                print(f"  Initial P(target): {initial_target_prob:.4f}")
                print(f"  Final P(target): {final_target_prob:.4f}")
                print(f"  Total increase: {final_target_prob / max(initial_target_prob, 1e-10):.1f}×")
                print(f"  Overall feedback rate: {overall_feedback_rate:.1%}")
                print(f"  Total weight updates: {total_n_updates}")
            else:
                print(f"\n✓ Training complete:")
                print(f"    Initial P(target): {initial_target_prob:.6f}")
                print(f"    Final P(target): {final_target_prob:.6f}")
                print(f"    Increase: {final_target_prob / max(initial_target_prob, 1e-10):.1f}×")
                print(f"    Feedback rate: {overall_feedback_rate:.2%} "
                      f"({total_n_positive_feedback}/{total_n_updates} positive)")
                print(f"    Weight updates: {total_n_updates}")
            
            # Striatal statistics
            stats = self.striatum.get_statistics()
            print(f"    Target weight: {stats['target_weight']:.4f}")
            print(f"    Mean weight: {stats['mean_weight']:.4f}")
            print(f"    Weight entropy: {stats['entropy']:.2f} bits")
            print()
        
        # Mark as trained
        self._is_trained = True
        
        # ======================================================================
        # PHASE 3: POST-TRAINING MEASUREMENT
        # ======================================================================
        if verbose:
            print("=" * 70)
            print("PHASE 3: POST-TRAINING MEASUREMENT")
            print("=" * 70)

        # Pre-allocate array for better performance
        post_eeg = np.empty(post_duration_ms, dtype=np.float64)

        for t in tqdm(
            range(post_duration_ms),
            desc="Post-training",
            disable=not verbose,
            unit="ms",
            mininterval=0.1
        ):
            # Continue sampling striatum (with learned weights)
            active_units = self.striatum.sample()
            target_active = self.striatum.is_target_active(active_units)

            # Apply modulation to see effects of learning
            # but do NOT update weights (no learning in post phase)
            modulation = 1.0 if target_active else 0.0
            post_eeg[t] = self.eeg_generator.step(thalamic_modulation=modulation)
        
        if verbose:
            print("✓ Post-training recording complete\n")
        
        # ======================================================================
        # ANALYSIS: Compare baseline vs post-training
        # ======================================================================
        if verbose:
            print("=" * 70)
            print("ANALYSIS")
            print("=" * 70)
        
        # Compute UAF distributions
        baseline_uaf = self.analyzer.compute_uaf_distribution(baseline_eeg)
        post_uaf = self.analyzer.compute_uaf_distribution(post_eeg)
        
        # Statistics
        baseline_uaf_mean = np.mean(baseline_uaf)
        baseline_uaf_std = np.std(baseline_uaf)
        baseline_uaf_min = np.min(baseline_uaf)
        baseline_uaf_max = np.max(baseline_uaf)
        
        post_uaf_mean = np.mean(post_uaf)
        post_uaf_std = np.std(post_uaf)
        post_uaf_min = np.min(post_uaf)
        post_uaf_max = np.max(post_uaf)
        
        uaf_change = post_uaf_mean - baseline_uaf_mean
        uaf_change_pct = (
            (uaf_change / baseline_uaf_mean) * 100
            if baseline_uaf_mean > 0 else 0.0
        )
        
        # Check learning success criteria (from Figure 5, Davelaar 2018)
        learning_success = (
            final_target_prob > 0.1 and                    # P(target) increased 100×
            post_uaf_mean > baseline_uaf_mean and          # Mean increased
            post_uaf_std > baseline_uaf_std and            # Variance increased
            post_uaf_min == baseline_uaf_min               # Min unchanged (key criterion!)
        )
        
        if verbose:
            print("UAF Distribution Analysis:")
            print(f"  Baseline:")
            print(f"    Mean: {baseline_uaf_mean:.4f}")
            print(f"    Std:  {baseline_uaf_std:.4f}")
            print(f"    Min:  {baseline_uaf_min:.4f}")
            print(f"    Max:  {baseline_uaf_max:.4f}")
            print()
            print(f"  Post-training:")
            print(f"    Mean: {post_uaf_mean:.4f}")
            print(f"    Std:  {post_uaf_std:.4f}")
            print(f"    Min:  {post_uaf_min:.4f}")
            print(f"    Max:  {post_uaf_max:.4f}")
            print()
            print(f"  Changes:")
            print(f"    ΔMean: {uaf_change:+.4f} ({uaf_change_pct:+.1f}%)")
            print(f"    ΔStd:  {post_uaf_std - baseline_uaf_std:+.4f}")
            print(f"    ΔMin:  {post_uaf_min - baseline_uaf_min:+.4f} "
                  f"{'✓ (unchanged)' if post_uaf_min == baseline_uaf_min else '✗'}")
            print(f"    ΔMax:  {post_uaf_max - baseline_uaf_max:+.4f}")
            print()
            print("=" * 70)
            print(f"LEARNING SUCCESS: {'✓ YES' if learning_success else '✗ NO'}")
            print("=" * 70)
            
            if learning_success:
                print(f"✓ Target probability increased "
                      f"{final_target_prob / max(initial_target_prob, 1e-10):.0f}×")
                print(f"✓ UAF distribution stretched to higher values")
                print(f"✓ Minimum UAF unchanged (characteristic of model)")
            else:
                print("Criteria not met:")
                if final_target_prob <= 0.1:
                    print(f"  ✗ P(target) = {final_target_prob:.4f} ≤ 0.1")
                if post_uaf_mean <= baseline_uaf_mean:
                    print(f"  ✗ Post mean ≤ baseline mean")
                if post_uaf_std <= baseline_uaf_std:
                    print(f"  ✗ Post std ≤ baseline std")
                if post_uaf_min != baseline_uaf_min:
                    print(f"  ✗ Minimum changed (unexpected)")
            
            print("=" * 70)
        
        # ======================================================================
        # STORE RESULTS
        # ======================================================================
        self.results = {
            # EEG signals
            'baseline_eeg': baseline_eeg,
            'training_eeg': training_eeg,
            'post_eeg': post_eeg,
            
            # UAF distributions
            'baseline_uaf': baseline_uaf,
            'post_uaf': post_uaf,
            
            # UAF statistics
            'baseline_uaf_mean': baseline_uaf_mean,
            'baseline_uaf_std': baseline_uaf_std,
            'baseline_uaf_min': baseline_uaf_min,
            'baseline_uaf_max': baseline_uaf_max,
            'post_uaf_mean': post_uaf_mean,
            'post_uaf_std': post_uaf_std,
            'post_uaf_min': post_uaf_min,
            'post_uaf_max': post_uaf_max,
            'uaf_change': uaf_change,
            'uaf_change_pct': uaf_change_pct,
            
            # Learning trajectories
            'target_prob_history': np.array(target_prob_history),
            'feedback_history': np.array(feedback_history),
            'uaf_power_history': np.array(uaf_power_history),
            
            # Learning statistics
            'initial_target_prob': initial_target_prob,
            'final_target_prob': final_target_prob,
            'target_prob_increase': final_target_prob / max(initial_target_prob, 1e-10),
            'n_updates': total_n_updates,
            'n_positive_feedback': total_n_positive_feedback,
            'feedback_rate': overall_feedback_rate,
            
            # Spectral analysis results
            'paf': paf,
            'uaf_band': uaf_band,
            'baseline_mean': baseline_mean,
            'baseline_std': baseline_std,
            'feedback_threshold': feedback_threshold,
            
            # Success criterion
            'learning_success': learning_success,
            
            # Model configuration
            'target_index': self.striatum.get_target_index(),
            'random_seed': self.random_seed,
            'striatal_stats': self.striatum.get_statistics(),
            'analyzer_stats': self.analyzer.get_statistics(),
            
            # Protocol parameters
            'protocol_params': {
                'baseline_duration': baseline_duration,
                'training_duration': training_duration,
                'post_duration': post_duration,
                'update_interval': update_interval,
                'warmup_duration': warmup_duration,
                'feedback_threshold_offset': feedback_threshold_offset
            }
        }
        
        return self.results
    
    def get_results(self) -> Dict:
        """
        Get simulation results.
        
        Returns:
            Dictionary containing all simulation results
            
        Raises:
            RuntimeError: If simulation hasn't been run yet
            
        Examples:
            >>> sim = NeurofeedbackSimulation()
            >>> results = sim.run_training_protocol()
            >>> results_copy = sim.get_results()
        """
        if not self._is_trained:
            raise RuntimeError(
                "Simulation has not been run yet. "
                "Call run_training_protocol() first."
            )
        
        return self.results.copy()
    
    def reset(self, preserve_components: bool = False) -> None:
        """
        Reset simulation to initial state.
        
        Args:
            preserve_components: If True, keep same component instances
                               (just reset their states). If False, create
                               new instances with same parameters.
                               
        Examples:
            >>> sim = NeurofeedbackSimulation(random_seed=42)
            >>> results1 = sim.run_training_protocol()
            >>> 
            >>> # Reset and run again
            >>> sim.reset(preserve_components=True)
            >>> results2 = sim.run_training_protocol()
        """
        if preserve_components:
            # Reset component states
            self.eeg_generator.reset()
            self.striatum.reset(preserve_target=False)
            self.analyzer.reset(clear_baseline=True)
            if self.activation_buffer is not None:
                self.activation_buffer.reset(reason="simulation reset")
        else:
            # Create new instances
            self.eeg_generator = IzhikevichEEGGenerator(
                n_exc=self.eeg_generator.n_exc,
                n_inh=self.eeg_generator.n_inh,
                random_seed=self.random_seed,
                measurement_noise_std=self.eeg_generator.measurement_noise_std
            )
            
            self.striatum = StriatalLearning(
                n_units=self.striatum.n_units,
                n_active_expected=self.striatum.n_active_expected,
                learning_rate=self.striatum.learning_rate,
                window_size=self.striatum.window_size,
                random_seed=self.random_seed
            )
            
            self.analyzer = SpectralAnalyzer(
                sampling_rate=self.analyzer.sampling_rate,
                window_size=self.analyzer.window_size,
                update_interval=self.analyzer.update_interval,
                alpha_band=self.analyzer.alpha_band
            )
            
            self.activation_buffer = None
        
        # Clear results
        self.results = {}
        self._is_trained = False
    
    def __repr__(self) -> str:
        """String representation of the simulation."""
        status = "trained" if self._is_trained else "untrained"
        return (
            f"NeurofeedbackSimulation("
            f"status={status}, "
            f"n_exc={self.eeg_generator.n_exc}, "
            f"n_inh={self.eeg_generator.n_inh}, "
            f"n_msn={self.striatum.n_units}, "
            f"target={self.striatum.get_target_index()}, "
            f"seed={self.random_seed})"
        )


# ==============================================================================
# CONVENIENCE FUNCTION
# ==============================================================================

def run_quick_test(random_seed: Optional[int] = 42, verbose: bool = True) -> Dict:
    """
    Run a quick test simulation (30 seconds total).
    
    Args:
        random_seed: Random seed for reproducibility
        verbose: Whether to print progress
        
    Returns:
        Simulation results dictionary
        
    Examples:
        >>> results = run_quick_test(random_seed=42)
        >>> print(f"Learning success: {results['learning_success']}")
    """
    sim = NeurofeedbackSimulation(random_seed=random_seed)
    
    results = sim.run_training_protocol(
        baseline_duration=10,   # 10 seconds
        training_duration=10,   # 10 seconds
        post_duration=10,       # 10 seconds
        verbose=verbose
    )
    
    return results


def run_full_protocol(random_seed: Optional[int] = None, verbose: bool = True) -> Dict:
    """
    Run full Davelaar (2018) protocol (11 minutes total).
    
    Args:
        random_seed: Random seed for reproducibility
        verbose: Whether to print progress
        
    Returns:
        Simulation results dictionary
        
    Examples:
        >>> results = run_full_protocol(random_seed=42)
        >>> print(f"PAF: {results['paf']:.2f} Hz")
        >>> print(f"Learning success: {results['learning_success']}")
    """
    sim = NeurofeedbackSimulation(random_seed=random_seed)
    
    results = sim.run_training_protocol(
        baseline_duration=5*60,   # 5 minutes
        training_duration=1*60,   # 1 minute
        post_duration=5*60,       # 5 minutes
        verbose=verbose
    )
    
    return results


# ==============================================================================
# MAIN - DEMO
# ==============================================================================

if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("NEUROFEEDBACK SIMULATION DEMO")
    print("=" * 70 + "\n")
    
    print("Running quick test (30 seconds)...\n")
    
    try:
        results = run_quick_test(random_seed=42, verbose=True)
        
        print("\n" + "=" * 70)
        print("DEMO COMPLETE")
        print("=" * 70)
        print("\nKey Results:")
        print(f"  PAF: {results['paf']:.2f} Hz")
        print(f"  Learning success: {results['learning_success']}")
        print(f"  P(target): {results['initial_target_prob']:.6f} → "
              f"{results['final_target_prob']:.6f}")
        print(f"  UAF change: {results['uaf_change']:+.4f} "
              f"({results['uaf_change_pct']:+.1f}%)")
        print("=" * 70)
        
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()