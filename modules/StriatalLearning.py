"""
Striatal Learning System with Reward-Modulated Plasticity.

This module implements the striatal learning component from Davelaar (2018) Simulation Study 1.
The system consists of 1000 binary MSN (Medium Spiny Neuron) units that learn through
reward-modulated plasticity based on neurofeedback signals.

Reference:
    Davelaar, E.J. (2018). Mechanisms of Neurofeedback: A Computation-theoretic Approach.
    Neuroscience, 378, 175-188.

Author: Stanislav Revko (stanislav.revko@gmail.com)
Date: November 22, 2025
"""

import numpy as np
from typing import Optional, Tuple, Dict
import warnings


class StriatalLearning:
    """
    Striatal learning system with 1000 binary MSN units.
    
    Implements reward-modulated plasticity where weights are updated based on
    feedback signals and activation history. The system learns to increase the
    probability of a target MSN unit being active through importance sampling.
    
    The learning mechanism solves the credit assignment problem by statistically
    accumulating evidence: units that are frequently active when reward occurs
    receive weight increases, while units active during no-reward periods are
    weakened.
    
    Attributes:
        n_units (int): Number of MSN units (default: 1000)
        n_active_expected (float): Expected number of active units per sample (default: 10)
        weights (np.ndarray): Fronto-striatal weights for each unit
        target_index (int): Index of target MSN unit (hidden from model)
        learning_rate (float): Learning rate for weight updates (default: 0.1)
        window_size (int): Feedback window size in ms (default: 1024)
        
    Examples:
        >>> striatum = StriatalLearning(n_units=1000, learning_rate=0.1)
        >>> 
        >>> # Sample active units
        >>> active = striatum.sample()
        >>> print(len(active))  # ~10 units
        >>> 
        >>> # Check if target is active
        >>> target_active = striatum.is_target_active(active)
        >>> 
        >>> # Update weights (after accumulating activation counts)
        >>> counts = np.zeros(1000)
        >>> counts[active] += 1
        >>> striatum.update_weights(feedback=True, activation_counts=counts)
        >>> 
        >>> # Check learning progress
        >>> print(striatum.get_target_probability())  # Increases over time
    """
    
    def __init__(
        self,
        n_units: int = 1000,
        n_active_expected: float = 10.0,
        learning_rate: float = 0.1,
        window_size: int = 1024,
        random_seed: Optional[int] = None,
        target_index: Optional[int] = None
    ):
        """
        Initialize striatal learning system.
        
        Args:
            n_units: Number of MSN units (must be positive)
            n_active_expected: Expected number of active units per sample (default: 10)
            learning_rate: Learning rate for weight updates (default: 0.1)
            window_size: Feedback window size in milliseconds (default: 1024)
            random_seed: Random seed for reproducibility (None for random)
            target_index: Specific target index (for testing), or None for random
            
        Raises:
            ValueError: If parameters are invalid
        """
        # Validate parameters
        if n_units <= 0:
            raise ValueError(f"n_units must be positive, got {n_units}")
        if n_active_expected <= 0:
            raise ValueError(f"n_active_expected must be positive, got {n_active_expected}")
        if n_active_expected > n_units:
            raise ValueError(
                f"n_active_expected ({n_active_expected}) cannot exceed n_units ({n_units})"
            )
        if learning_rate <= 0:
            raise ValueError(f"learning_rate must be positive, got {learning_rate}")
        if window_size <= 0:
            raise ValueError(f"window_size must be positive, got {window_size}")
        
        self.n_units = n_units
        self.n_active_expected = n_active_expected
        self.learning_rate = learning_rate
        self.window_size = window_size
        
        # Set random seed
        if random_seed is not None:
            np.random.seed(random_seed)
        
        # Initialize weights (all start equal)
        self.weights = np.ones(n_units, dtype=np.float64)
        
        # Select target MSN unit
        if target_index is not None:
            # Explicit target (for testing)
            if target_index < 0 or target_index >= n_units:
                raise ValueError(
                    f"target_index must be in [0, {n_units}), got {target_index}"
                )
            self.target_index = target_index
        else:
            # Random target (normal operation)
            self.target_index = np.random.randint(0, n_units)
        
        # Store initial state for reset
        self.weights_init = self.weights.copy()
        self.target_index_init = self.target_index
        
        # Statistics tracking
        self._total_updates = 0
        self._positive_feedbacks = 0
        self._negative_feedbacks = 0
        self._weight_history = []  # For debugging (optional)
        
    def sample(self) -> np.ndarray:
        """
        Sample active MSN units based on current weights.
        
        Uses stochastic binary activation where each unit has a probability
        proportional to its normalized weight. Expected number of active units
        is approximately n_active_expected (default: 10).
        
        The sampling process implements importance sampling: weights encode
        the learned probability distribution over MSN activation patterns.
        
        Returns:
            Array of indices of active units (shape: (n_active,) where n_active varies)
            
        Examples:
            >>> striatum = StriatalLearning()
            >>> active = striatum.sample()
            >>> print(len(active))  # Varies, but ~10 on average
            >>> print(active)  # e.g., [42, 157, 389, 501, ...]
        """
        # Normalize weights to get probability distribution
        # Sum of normalized weights = 1.0
        weight_sum = np.sum(self.weights)
        
        # Handle edge case: all weights are zero (shouldn't happen, but be safe)
        if weight_sum < 1e-10:
            warnings.warn(
                "All weights are near zero. Resetting to uniform distribution.",
                RuntimeWarning
            )
            self.weights = np.ones(self.n_units, dtype=np.float64)
            weight_sum = self.n_units
        
        normalized_weights = self.weights / weight_sum
        
        # Scale by n_active_expected to get activation probabilities
        # If n_active_expected = 10: sum of probabilities ≈ 10
        # → Expected number of active units ≈ 10
        probabilities = normalized_weights * self.n_active_expected
        
        # Clip probabilities to valid range [0, 1]
        probabilities = np.clip(probabilities, 0.0, 1.0)
        
        # Stochastic binary activation: each unit activates with its probability
        random_values = np.random.rand(self.n_units)
        active_mask = random_values < probabilities
        active_units = np.where(active_mask)[0]
        
        return active_units
    
    def is_target_active(self, active_units: Optional[np.ndarray] = None) -> bool:
        """
        Check if target MSN unit is active.
        
        Args:
            active_units: Array of indices of active units, or None to sample
            
        Returns:
            True if target unit is in active_units, False otherwise
            
        Examples:
            >>> striatum = StriatalLearning(target_index=42)
            >>> active = np.array([10, 42, 100])
            >>> print(striatum.is_target_active(active))  # True
            >>> active = np.array([10, 20, 100])
            >>> print(striatum.is_target_active(active))  # False
        """
        if active_units is None:
            active_units = self.sample()
        
        # Convert to array if needed
        active_units = np.asarray(active_units)
        
        return self.target_index in active_units
    
    def update_weights(
        self,
        feedback: bool,
        activation_counts: np.ndarray
    ) -> None:
        """
        Update weights based on reward-modulated plasticity rule.
        
        Implements the learning rule from Davelaar (2018):
        - Positive feedback: Δw_i = η × (count_i / window_size)
        - Negative feedback: Δw_i = -η × (mean_count_active / window_size) for active units
        
        After update, weights are normalized to maintain probability distribution.
        
        Args:
            feedback: Boolean indicating positive reward (True) or no reward (False)
            activation_counts: Array of activation counts for each unit over feedback window
                              (shape: (n_units,), values: 0 to window_size)
        
        Raises:
            ValueError: If activation_counts has wrong shape or invalid values
            
        Examples:
            >>> striatum = StriatalLearning()
            >>> # Simulate 1024 ms with unit 42 active 500 times
            >>> counts = np.zeros(1000)
            >>> counts[42] = 500
            >>> counts[100] = 300
            >>> 
            >>> # Positive feedback: strengthen active units
            >>> striatum.update_weights(feedback=True, activation_counts=counts)
            >>> # Weight[42] and Weight[100] increased
            >>> 
            >>> # Negative feedback: weaken active units
            >>> striatum.update_weights(feedback=False, activation_counts=counts)
            >>> # Weight[42] and Weight[100] decreased
        """
        # Validate input
        activation_counts = np.asarray(activation_counts, dtype=np.float64)
        
        if activation_counts.shape != (self.n_units,):
            raise ValueError(
                f"activation_counts must have shape ({self.n_units},), "
                f"got {activation_counts.shape}"
            )
        
        if np.any(activation_counts < 0):
            raise ValueError("activation_counts cannot contain negative values")
        
        if np.any(activation_counts > self.window_size):
            warnings.warn(
                f"activation_counts contains values > window_size ({self.window_size}). "
                f"This may indicate a bug in the activation buffer.",
                RuntimeWarning
            )
        
        # Identify which units were active
        active_mask = activation_counts > 0
        n_active = np.sum(active_mask)
        
        if n_active == 0:
            # No units were active during the window → no learning
            return
        
        if feedback:
            # ============================================================
            # POSITIVE FEEDBACK: Strengthen active units
            # ============================================================
            # Weight update proportional to activation frequency
            # Δw_i = η × (count_i / window_size)
            
            # Compute update for all units (will be 0 for inactive units)
            update = self.learning_rate * (activation_counts / self.window_size)
            self.weights += update
            
            self._positive_feedbacks += 1
            
        else:
            # ============================================================
            # NEGATIVE FEEDBACK: Weaken active units
            # ============================================================
            # All active units are weakened by the same amount
            # Δw_i = -η × (mean_activation / window_size)
            
            # Compute mean activation count among active units
            mean_activation = np.mean(activation_counts[active_mask])
            
            # Compute update magnitude
            update_magnitude = self.learning_rate * (mean_activation / self.window_size)
            
            # Apply update only to active units
            self.weights[active_mask] -= update_magnitude
            
            self._negative_feedbacks += 1
        
        # ============================================================
        # POST-UPDATE: Ensure non-negativity
        # ============================================================
        # Weights must remain non-negative for probability interpretation
        self.weights = np.maximum(self.weights, 1e-10)
        
        # Track updates
        self._total_updates += 1
    
    def get_target_probability(self) -> float:
        """
        Get current probability of target MSN unit being active.
        
        Computes the probability that the target unit will be sampled
        in the next call to sample().
        
        Returns:
            Probability of target unit activation (0.0 to 1.0)
            
        Examples:
            >>> striatum = StriatalLearning(target_index=42)
            >>> print(striatum.get_target_probability())  # Initially ~0.01 (1/100)
            >>> # After training...
            >>> print(striatum.get_target_probability())  # Increases, e.g., 0.15
        """
        # Normalize weights to probability distribution
        weight_sum = np.sum(self.weights)
        
        if weight_sum < 1e-10:
            # Edge case: all weights near zero
            return 0.0
        
        normalized_weights = self.weights / weight_sum
        
        # Scale by n_active_expected
        probabilities = normalized_weights * self.n_active_expected
        
        # Clip to [0, 1]
        probabilities = np.clip(probabilities, 0.0, 1.0)
        
        return float(probabilities[self.target_index])
    
    def get_target_index(self) -> int:
        """
        Get the target MSN unit index (for monitoring/debugging).
        
        In a real brain, this index is "hidden" - the brain doesn't explicitly
        know which striatal representation is correct. Learning discovers it
        through trial and error.
        
        Returns:
            Index of target MSN unit
            
        Examples:
            >>> striatum = StriatalLearning(target_index=42)
            >>> print(striatum.get_target_index())  # 42
        """
        return self.target_index
    
    def get_weights(self) -> np.ndarray:
        """
        Get current weights (for monitoring/debugging).
        
        Returns:
            Copy of weight array for all units (shape: (n_units,))
            
        Examples:
            >>> striatum = StriatalLearning()
            >>> weights = striatum.get_weights()
            >>> print(weights.shape)  # (1000,)
            >>> print(weights.min(), weights.max())  # Initially all ~1.0
        """
        return self.weights.copy()
    
    def get_probabilities(self) -> np.ndarray:
        """
        Get current activation probabilities for all units.
        
        Returns:
            Array of activation probabilities (shape: (n_units,), values: 0.0 to 1.0)
            
        Examples:
            >>> striatum = StriatalLearning()
            >>> probs = striatum.get_probabilities()
            >>> print(probs.sum())  # ~10.0 (expected number active)
            >>> print(probs.max())  # Initially ~0.01, increases for target
        """
        weight_sum = np.sum(self.weights)
        
        if weight_sum < 1e-10:
            return np.zeros(self.n_units)
        
        normalized_weights = self.weights / weight_sum
        probabilities = normalized_weights * self.n_active_expected
        probabilities = np.clip(probabilities, 0.0, 1.0)
        
        return probabilities
    
    def get_statistics(self) -> Dict[str, float]:
        """
        Get learning statistics.
        
        Returns:
            Dictionary containing:
                - 'total_updates': Total number of weight updates
                - 'positive_feedbacks': Number of positive feedback updates
                - 'negative_feedbacks': Number of negative feedback updates
                - 'feedback_rate': Fraction of positive feedbacks
                - 'target_probability': Current target activation probability
                - 'target_weight': Current target weight
                - 'mean_weight': Mean weight across all units
                - 'max_weight': Maximum weight
                - 'weight_std': Standard deviation of weights
                - 'entropy': Shannon entropy of probability distribution (bits)
                
        Examples:
            >>> striatum = StriatalLearning()
            >>> stats = striatum.get_statistics()
            >>> print(f"Target probability: {stats['target_probability']:.4f}")
            >>> print(f"Feedback rate: {stats['feedback_rate']:.2%}")
        """
        probabilities = self.get_probabilities()
        
        # Compute Shannon entropy (measure of uncertainty)
        # H = -Σ p_i × log2(p_i)
        # High entropy → uniform distribution (uncertain)
        # Low entropy → peaked distribution (confident)
        probs_nonzero = probabilities[probabilities > 1e-10]
        if len(probs_nonzero) > 0:
            entropy = -np.sum(probs_nonzero * np.log2(probs_nonzero))
        else:
            entropy = 0.0
        
        return {
            'total_updates': self._total_updates,
            'positive_feedbacks': self._positive_feedbacks,
            'negative_feedbacks': self._negative_feedbacks,
            'feedback_rate': (
                self._positive_feedbacks / max(self._total_updates, 1)
            ),
            'target_probability': self.get_target_probability(),
            'target_weight': float(self.weights[self.target_index]),
            'mean_weight': float(np.mean(self.weights)),
            'max_weight': float(np.max(self.weights)),
            'weight_std': float(np.std(self.weights)),
            'entropy': float(entropy),
            'weight_sum': float(np.sum(self.weights))
        }
    
    def reset(self, preserve_target: bool = False) -> None:
        """
        Reset the system to initial state.
        
        Args:
            preserve_target: If True, keep the same target unit.
                           If False, select a new random target.
                           
        Examples:
            >>> striatum = StriatalLearning()
            >>> old_target = striatum.get_target_index()
            >>> 
            >>> # Reset with new target
            >>> striatum.reset(preserve_target=False)
            >>> new_target = striatum.get_target_index()
            >>> # old_target != new_target (probably)
            >>> 
            >>> # Reset keeping same target
            >>> striatum.reset(preserve_target=True)
            >>> # target unchanged
        """
        # Reset weights to uniform
        self.weights = self.weights_init.copy()
        
        # Reset or change target
        if preserve_target:
            self.target_index = self.target_index_init
        else:
            self.target_index = np.random.randint(0, self.n_units)
            self.target_index_init = self.target_index
        
        # Reset statistics
        self._total_updates = 0
        self._positive_feedbacks = 0
        self._negative_feedbacks = 0
        self._weight_history = []
    
    def __repr__(self) -> str:
        """String representation of the striatal learning system."""
        stats = self.get_statistics()
        return (
            f"StriatalLearning("
            f"n_units={self.n_units}, "
            f"target_idx={self.target_index}, "
            f"P(target)={stats['target_probability']:.4f}, "
            f"updates={stats['total_updates']}, "
            f"feedback_rate={stats['feedback_rate']:.2%})"
        )


# ==============================================================================
# UNIT TESTS
# ==============================================================================

def _test_initialization():
    """Test initialization and parameter validation."""
    print("Testing initialization...")
    
    # Valid initialization
    striatum = StriatalLearning(n_units=100, learning_rate=0.1)
    assert striatum.n_units == 100
    assert striatum.learning_rate == 0.1
    assert 0 <= striatum.target_index < 100
    
    # Test with explicit target
    striatum = StriatalLearning(n_units=100, target_index=42)
    assert striatum.target_index == 42
    
    # Test invalid parameters
    try:
        StriatalLearning(n_units=0)
        assert False, "Should raise ValueError"
    except ValueError:
        pass
    
    try:
        StriatalLearning(n_units=100, learning_rate=-0.1)
        assert False, "Should raise ValueError"
    except ValueError:
        pass
    
    print("✓ Initialization tests passed")


def _test_sampling():
    """Test MSN unit sampling."""
    print("Testing sampling...")
    
    striatum = StriatalLearning(n_units=1000, n_active_expected=10)
    
    # Sample multiple times and check statistics
    n_samples = 1000
    active_counts = []
    
    for _ in range(n_samples):
        active = striatum.sample()
        active_counts.append(len(active))
        
        # Check all indices are valid
        assert np.all(active >= 0)
        assert np.all(active < 1000)
        
        # Check no duplicates
        assert len(active) == len(np.unique(active))
    
    # Check expected number of active units
    mean_active = np.mean(active_counts)
    print(f"  Mean active units: {mean_active:.2f} (expected: 10)")
    assert 8 < mean_active < 12, f"Mean active {mean_active} outside expected range"
    
    print("✓ Sampling tests passed")


def _test_learning():
    """Test reward-modulated learning."""
    print("Testing learning...")
    
    striatum = StriatalLearning(n_units=100, target_index=42, learning_rate=0.1)
    
    # Initial target probability should be low
    initial_prob = striatum.get_target_probability()
    print(f"  Initial P(target): {initial_prob:.4f}")
    assert initial_prob < 0.2
    
    # Simulate training: always activate target and give reward
    for _ in range(100):
        counts = np.zeros(100)
        counts[42] = 100  # Target always active
        striatum.update_weights(feedback=True, activation_counts=counts)
    
    # Target probability should increase
    final_prob = striatum.get_target_probability()
    print(f"  Final P(target): {final_prob:.4f}")
    assert final_prob > initial_prob, "Learning did not occur"
    
    print("✓ Learning tests passed")


def _test_credit_assignment():
    """Test that learning solves credit assignment problem."""
    print("Testing credit assignment...")
    
    striatum = StriatalLearning(n_units=100, target_index=42, learning_rate=0.1)
    
    # Simulate realistic scenario: 10 units active, reward when target is one of them
    for trial in range(500):
        # Sample active units
        active = striatum.sample()
        target_active = striatum.is_target_active(active)
        
        # Simulate activation counts
        counts = np.zeros(100)
        counts[active] = 50  # All active units equally active
        
        # Reward if target was active
        feedback = target_active
        striatum.update_weights(feedback=feedback, activation_counts=counts)
    
    # Target should have learned to have higher probability
    final_prob = striatum.get_target_probability()
    print(f"  Final P(target): {final_prob:.4f}")
    
    # Target should be more likely than initial uniform (0.1)
    assert final_prob > 0.15, "Credit assignment failed"
    
    stats = striatum.get_statistics()
    print(f"  Feedback rate: {stats['feedback_rate']:.2%}")
    print(f"  Total updates: {stats['total_updates']}")
    
    print("✓ Credit assignment tests passed")


def _test_input_validation():
    """Test input validation in update_weights."""
    print("Testing input validation...")
    
    striatum = StriatalLearning(n_units=100)
    
    # Valid update
    counts = np.zeros(100)
    counts[10] = 50
    striatum.update_weights(feedback=True, activation_counts=counts)
    
    # Test wrong shape
    try:
        striatum.update_weights(feedback=True, activation_counts=np.zeros(50))
        assert False, "Should raise ValueError"
    except ValueError:
        pass
    
    # Test negative values
    try:
        counts = np.zeros(100)
        counts[10] = -5
        striatum.update_weights(feedback=True, activation_counts=counts)
        assert False, "Should raise ValueError"
    except ValueError:
        pass
    
    print("✓ Input validation tests passed")


def _test_statistics():
    """Test statistics tracking."""
    print("Testing statistics...")
    
    striatum = StriatalLearning(n_units=100)
    
    # Initial statistics
    stats = striatum.get_statistics()
    assert stats['total_updates'] == 0
    assert stats['positive_feedbacks'] == 0
    assert stats['negative_feedbacks'] == 0
    
    # Perform updates
    counts = np.zeros(100)
    counts[10] = 50
    
    striatum.update_weights(feedback=True, activation_counts=counts)
    stats = striatum.get_statistics()
    assert stats['total_updates'] == 1
    assert stats['positive_feedbacks'] == 1
    
    striatum.update_weights(feedback=False, activation_counts=counts)
    stats = striatum.get_statistics()
    assert stats['total_updates'] == 2
    assert stats['negative_feedbacks'] == 1
    assert abs(stats['feedback_rate'] - 0.5) < 0.01
    
    print("✓ Statistics tests passed")


def _test_reset():
    """Test reset functionality."""
    print("Testing reset...")
    
    striatum = StriatalLearning(n_units=100, target_index=42)
    
    # Modify weights
    counts = np.zeros(100)
    counts[42] = 100
    for _ in range(50):
        striatum.update_weights(feedback=True, activation_counts=counts)
    
    # Weights should be modified
    weights_before = striatum.get_weights()
    assert not np.allclose(weights_before, np.ones(100))
    
    # Reset preserving target
    striatum.reset(preserve_target=True)
    assert striatum.target_index == 42
    weights_after = striatum.get_weights()
    assert np.allclose(weights_after, np.ones(100))
    
    # Statistics should be reset
    stats = striatum.get_statistics()
    assert stats['total_updates'] == 0
    
    print("✓ Reset tests passed")


if __name__ == "__main__":
    print("=" * 70)
    print("RUNNING STRIATAL LEARNING TESTS")
    print("=" * 70)
    
    _test_initialization()
    _test_sampling()
    _test_learning()
    _test_credit_assignment()
    _test_input_validation()
    _test_statistics()
    _test_reset()
    
    print("\n" + "=" * 70)
    print("ALL TESTS PASSED ✓")
    print("=" * 70)