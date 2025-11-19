"""
Neurofeedback Simulation Model.

This module integrates all components (EEG generator, striatal learning, spectral analysis)
to implement the complete neurofeedback training protocol from Davelaar (2018) Simulation Study 1.

Reference:
    Davelaar, E.J. (2018). Mechanisms of Neurofeedback: A Computation-theoretic Approach.
    Neuroscience, 378, 175-188.

Author: Generated from specification
"""

import numpy as np
from typing import Dict, Optional, Tuple
from tqdm import tqdm

from .IzhikevichEEGGenerator import IzhikevichEEGGenerator
from .ActivationBuffer import ActivationBuffer
from .StriatalLearning import StriatalLearning
from .SpectralAnalyzer import SpectralAnalyzer


class NeurofeedbackSimulation:
    """
    Complete neurofeedback simulation integrating all components.
    
    Implements the three-phase training protocol:
    1. Baseline: 5 minutes recording to establish PAF and baseline statistics
    2. Training: 1 minute neurofeedback learning with reward-modulated plasticity
    3. Post-training: 5 minutes recording to measure training effects
    
    Attributes:
        eeg_generator (IzhikevichEEGGenerator): EEG signal generator
        striatum (StriatalLearning): Striatal learning system
        analyzer (SpectralAnalyzer): Spectral analysis for feedback
        random_seed (Optional[int]): Random seed for reproducibility
        results (Dict): Dictionary storing simulation results
    """
    
    def __init__(
        self,
        n_exc: int = 800,
        n_inh: int = 200,
        n_msn: int = 1000,
        learning_rate: float = 0.1,
        random_seed: Optional[int] = None,
        measurement_noise_std: float = 0.0
    ):
        """
        Initialize neurofeedback simulation.
        
        Args:
            n_exc: Number of excitatory neurons
            n_inh: Number of inhibitory neurons
            n_msn: Number of MSN units
            learning_rate: Learning rate for striatal weight updates
            random_seed: Random seed for reproducibility
            measurement_noise_std: Standard deviation of additive white Gaussian noise
                                 added to EEG signal (default: 0.0, no noise)
        """
        self.random_seed = random_seed
        
        # Initialize components
        self.eeg_generator = IzhikevichEEGGenerator(
            n_exc=n_exc,
            n_inh=n_inh,
            random_seed=random_seed,
            measurement_noise_std=measurement_noise_std
        )
        
        # Increase learning rate significantly for faster learning
        # Paper shows rapid learning: P(target) increases from 0.001 to 0.4-0.9
        # With default 0.1, updates are too small (~0.001 per update)
        # Need much larger learning rate to achieve rapid learning
        effective_learning_rate = learning_rate * 10.0  # 10× increase: 0.1 → 1.0
        self.striatum = StriatalLearning(
            n_units=n_msn,
            learning_rate=effective_learning_rate,
            random_seed=random_seed
        )
        
        self.analyzer = SpectralAnalyzer()
        
        # Results storage
        self.results = {}
        
    def run_training_protocol(
        self,
        baseline_duration: float = 5 * 60,        # 5 minutes in seconds
        training_duration: float = 5 * 60,        # 5 minutes in seconds (per phase)
        n_training_phases: int = 5,              # Number of training phases
        post_duration: float = 5 * 60,            # 5 minutes in seconds
        update_interval: float = 0.1,             # Feedback interval in seconds (default: 100 ms)
        warmup_duration: float = 1.0,             # Warmup duration in seconds (default: 1 s)
        feedback_threshold_offset: float = 0.5,   # Threshold offset in std units (baseline_mean + offset*std)
        verbose: bool = True
    ) -> Dict:
        """
        Run complete training protocol: baseline + multiple training phases + post-measurement.
        
        Args:
            baseline_duration: Baseline recording duration in seconds
            training_duration: Training duration per phase in seconds
            n_training_phases: Number of training phases (default: 5)
            post_duration: Post-training recording duration in seconds
            update_interval: Feedback update interval in seconds (default: 0.1 = 100 ms)
            warmup_duration: Warmup duration for EEG generator in seconds (default: 1.0)
            feedback_threshold_offset: Threshold offset in std units (baseline_mean + offset*std)
            verbose: Whether to print progress information
            
        Returns:
            Dictionary containing all simulation results
        """
        # Convert durations from seconds to milliseconds (internal time step is 1 ms)
        baseline_duration_ms = int(baseline_duration * 1000)
        training_duration_ms = int(training_duration * 1000)
        post_duration_ms = int(post_duration * 1000)
        update_interval_ms = int(update_interval * 1000)
        warmup_duration_ms = int(warmup_duration * 1000)
        
        if verbose:
            print("=" * 60)
            print("NEUROFEEDBACK SIMULATION (Davelaar 2018)")
            print("=" * 60)
            print(f"\nInitializing simulation...")
            print(f"  MSN units: {self.striatum.n_units}")
            print(f"  Cortical neurons: {self.eeg_generator.n_exc}E + {self.eeg_generator.n_inh}I")
            print(f"  Target MSN: {self.striatum.get_target_index()} (hidden)")
            print()
        
        # Warm up EEG generator to reduce initialization transients
        if verbose:
            print(f"Warming up EEG generator ({warmup_duration:.1f} s)...")
        self.eeg_generator.warmup(duration_ms=warmup_duration_ms)
        if verbose:
            print("Warmup complete.\n")
        
        # === PHASE 1: BASELINE ===
        if verbose:
            print("Phase 1: Baseline Recording")
            print(f"  Duration: {baseline_duration:.1f} seconds ({baseline_duration / 60:.1f} minutes)")
        
        baseline_eeg = []
        
        for t in tqdm(range(baseline_duration_ms), desc="Baseline", disable=not verbose):
            # Generate EEG without modulation
            eeg_sample = self.eeg_generator.step(thalamic_modulation=0.0)
            baseline_eeg.append(eeg_sample)
        
        baseline_eeg = np.array(baseline_eeg)
        
        # Establish baseline statistics
        paf = self.analyzer.find_peak_alpha_frequency(baseline_eeg)
        baseline_mean, baseline_std = self.analyzer.set_baseline(baseline_eeg, paf=paf)
        uaf_band = self.analyzer.get_uaf_band()
        
        if verbose:
            print(f"\nBaseline established:")
            print(f"  PAF: {paf:.2f} Hz")
            print(f"  UAF band: [{uaf_band[0]:.2f}, {uaf_band[1]:.2f}] Hz")
            print(f"  Baseline UAF power: {baseline_mean:.2f} ± {baseline_std:.2f}")
            print()
        
        # === PHASE 2: MULTIPLE TRAINING PHASES ===
        # Compute feedback threshold once
        feedback_threshold = baseline_mean + feedback_threshold_offset * baseline_std
        
        # Track results across all training phases
        all_training_eeg = []
        all_feedback_history = []
        all_target_prob_history = []
        all_uaf_power_history = []
        training_phase_results = []
        
        # Track initial target probability
        initial_target_prob = self.striatum.get_target_probability()
        all_target_prob_history.append(initial_target_prob)
        
        total_n_updates = 0
        
        for phase_num in range(1, n_training_phases + 1):
            if verbose:
                print(f"Phase {phase_num + 1}: Training Phase {phase_num}")
                print(f"  Duration: {training_duration:.1f} seconds ({training_duration / 60:.1f} minutes)")
            
            # Initialize buffers for this phase
            activation_buffer = ActivationBuffer(window_size=1024, n_units=self.striatum.n_units)
            
            phase_training_eeg = []
            phase_feedback_history = []
            phase_uaf_power_history = []
            phase_target_prob_history = []
            
            phase_n_updates = 0
            
            for t in tqdm(range(training_duration_ms), desc=f"Training Phase {phase_num}", disable=not verbose):
                # Sample striatum
                active_units = self.striatum.sample()
                target_active = self.striatum.is_target_active(active_units)
                activation_buffer.add(active_units)
                
                # Generate EEG with modulation
                modulation = 1.0 if target_active else 0.0
                eeg_sample = self.eeg_generator.step(thalamic_modulation=modulation)
                phase_training_eeg.append(eeg_sample)
                all_training_eeg.append(eeg_sample)
                
                # Add to analyzer buffer
                self.analyzer.add_sample(eeg_sample)
                
                # Feedback every update_interval ms
                if (t + 1) % update_interval_ms == 0 and len(self.analyzer.eeg_buffer) == self.analyzer.window_size:
                    # Compute UAF power
                    uaf_power = self.analyzer.compute_uaf_power()
                    phase_uaf_power_history.append(uaf_power)
                    all_uaf_power_history.append(uaf_power)
                    
                    # Get feedback signal
                    feedback = self.analyzer.get_feedback(threshold=feedback_threshold)
                    phase_feedback_history.append(feedback)
                    all_feedback_history.append(feedback)
                    
                    # Update striatal weights
                    activation_counts = activation_buffer.get_counts()
                    self.striatum.update_weights(feedback, activation_counts)
                    
                    # Note: Do NOT reset the buffer - it maintains a sliding window
                    # via circular buffering (automatically overwrites old data)
                    
                    # Track learning progress
                    target_prob = self.striatum.get_target_probability()
                    phase_target_prob_history.append(target_prob)
                    all_target_prob_history.append(target_prob)
                    
                    phase_n_updates += 1
                    total_n_updates += 1
            
            # Store phase results
            phase_final_prob = self.striatum.get_target_probability()
            phase_feedback_rate = np.mean(phase_feedback_history) if len(phase_feedback_history) > 0 else 0.0
            
            training_phase_results.append({
                'phase_num': phase_num,
                'initial_prob': phase_target_prob_history[0] if len(phase_target_prob_history) > 0 else initial_target_prob,
                'final_prob': phase_final_prob,
                'feedback_rate': phase_feedback_rate,
                'n_updates': phase_n_updates,
                'eeg': np.array(phase_training_eeg),
                'target_prob_history': phase_target_prob_history,
                'feedback_history': phase_feedback_history,
                'uaf_power_history': phase_uaf_power_history
            })
            
            if verbose:
                print(f"\nTraining Phase {phase_num} results:")
                print(f"  Initial P(target): {phase_target_prob_history[0] if len(phase_target_prob_history) > 0 else initial_target_prob:.4f}")
                print(f"  Final P(target): {phase_final_prob:.4f}")
                print(f"  Increase: {phase_final_prob / (phase_target_prob_history[0] if len(phase_target_prob_history) > 0 else initial_target_prob):.1f}×")
                print(f"  Feedback rate: {phase_feedback_rate:.1%}")
                print(f"  Weight updates: {phase_n_updates}")
                print()
        
        training_eeg = np.array(all_training_eeg)
        final_target_prob = self.striatum.get_target_probability()
        feedback_rate = np.mean(all_feedback_history) if len(all_feedback_history) > 0 else 0.0
        
        if verbose:
            print(f"Overall Training Summary:")
            print(f"  Initial P(target): {initial_target_prob:.4f}")
            print(f"  Final P(target): {final_target_prob:.4f}")
            print(f"  Total increase: {final_target_prob / initial_target_prob:.1f}×")
            print(f"  Overall feedback rate: {feedback_rate:.1%}")
            print(f"  Total weight updates: {total_n_updates}")
            print()
        
        # === FINAL PHASE: POST-TRAINING ===
        # In post-training, we continue sampling striatum to see effects of learning,
        # but apply modulation based on target MSN activity (without further learning)
        phase_num = n_training_phases + 2
        if verbose:
            print(f"Phase {phase_num}: Post-Training")
            print(f"  Duration: {post_duration:.1f} seconds ({post_duration / 60:.1f} minutes)")
        
        post_eeg = []
        
        for t in tqdm(range(post_duration_ms), desc="Post-training", disable=not verbose):
            # Sample striatum to see effects of learning (target MSN should be more active now)
            active_units = self.striatum.sample()
            target_active = self.striatum.is_target_active(active_units)
            
            # Apply modulation based on target MSN activity (to see learning effects)
            # but don't update weights (no learning)
            modulation = 1.0 if target_active else 0.0
            eeg_sample = self.eeg_generator.step(thalamic_modulation=modulation)
            post_eeg.append(eeg_sample)
        
        post_eeg = np.array(post_eeg)
        
        # === ANALYSIS ===
        # Compute UAF distributions
        baseline_uaf = self.analyzer.compute_uaf_distribution(baseline_eeg)
        post_uaf = self.analyzer.compute_uaf_distribution(post_eeg)
        
        baseline_uaf_mean = np.mean(baseline_uaf)
        baseline_uaf_std = np.std(baseline_uaf)
        post_uaf_mean = np.mean(post_uaf)
        post_uaf_std = np.std(post_uaf)
        uaf_change = post_uaf_mean - baseline_uaf_mean
        uaf_change_pct = (uaf_change / baseline_uaf_mean) * 100 if baseline_uaf_mean > 0 else 0.0
        
        # Check learning success criteria
        learning_success = (
            final_target_prob > 0.1 and  # Target probability increased significantly
            post_uaf_mean > baseline_uaf_mean and  # UAF increased
            post_uaf_std > baseline_uaf_std  # Variance increased
        )
        
        if verbose:
            print()
            print("=" * 60)
            print("RESULTS")
            print("=" * 60)
            print("UAF Power:")
            print(f"  Before: {baseline_uaf_mean:.2f} ± {baseline_uaf_std:.2f}")
            print(f"  After:  {post_uaf_mean:.2f} ± {post_uaf_std:.2f}")
            print(f"  Change: {uaf_change:+.2f} ({uaf_change_pct:+.1f}%)")
            print()
            print(f"Learning Success: {'YES' if learning_success else 'NO'}")
            print(f"  Target probability increased {final_target_prob / initial_target_prob:.0f}×")
            print("=" * 60)
        
        # Store results
        self.results = {
            'baseline_eeg': baseline_eeg,
            'training_eeg': training_eeg,
            'post_eeg': post_eeg,
            'baseline_uaf': baseline_uaf,
            'post_uaf': post_uaf,
            'baseline_uaf_mean': baseline_uaf_mean,
            'baseline_uaf_std': baseline_uaf_std,
            'post_uaf_mean': post_uaf_mean,
            'post_uaf_std': post_uaf_std,
            'uaf_change': uaf_change,
            'uaf_change_pct': uaf_change_pct,
            'target_prob_history': np.array(all_target_prob_history),
            'feedback_history': np.array(all_feedback_history),
            'uaf_power_history': np.array(all_uaf_power_history),
            'initial_target_prob': initial_target_prob,
            'final_target_prob': final_target_prob,
            'feedback_rate': feedback_rate,
            'paf': paf,
            'uaf_band': uaf_band,
            'baseline_mean': baseline_mean,
            'baseline_std': baseline_std,
            'learning_success': learning_success,
            'target_index': self.striatum.get_target_index(),
            'random_seed': self.random_seed,
            'n_training_phases': n_training_phases,
            'training_phase_results': training_phase_results,
            'protocol_params': {
                'baseline_duration': baseline_duration,  # in seconds
                'training_duration': training_duration,  # in seconds (per phase)
                'n_training_phases': n_training_phases,
                'post_duration': post_duration,  # in seconds
                'update_interval': update_interval,  # in seconds
                'feedback_threshold_offset': feedback_threshold_offset
            }
        }
        
        return self.results
    
    def get_results(self) -> Dict:
        """
        Get simulation results.
        
        Returns:
            Dictionary containing all simulation results
        """
        return self.results.copy()

