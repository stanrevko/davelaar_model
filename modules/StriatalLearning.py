"""
Striatal Learning System with Reward-Modulated Plasticity.

This module implements the striatal learning component from Davelaar (2018) Simulation Study 1.
The system consists of 1000 binary MSN (Medium Spiny Neuron) units that learn through
reward-modulated plasticity based on neurofeedback signals.

Reference:
    Davelaar, E.J. (2018). Mechanisms of Neurofeedback: A Computation-theoretic Approach.
    Neuroscience, 378, 175-188.

Author: Generated from specification
"""

import numpy as np
from typing import Optional


class StriatalLearning:
    """
    Striatal learning system with 1000 binary MSN units.
    
    Implements reward-modulated plasticity where weights are updated based on
    feedback signals and activation history. The system learns to increase the
    probability of a target MSN unit being active.
    
    Attributes:
        n_units (int): Number of MSN units (default: 1000)
        weights (np.ndarray): Fronto-striatal weights for each unit
        target_index (int): Index of target MSN unit (hidden from model)
        learning_rate (float): Learning rate for weight updates (default: 0.1)
    """
    
    def __init__(
        self,
        n_units: int = 1000,
        learning_rate: float = 0.1,
        random_seed: Optional[int] = None
    ):
        """
        Initialize striatal learning system.
        
        Args:
            n_units: Number of MSN units
            learning_rate: Learning rate for weight updates
            random_seed: Random seed for reproducibility (None for random)
        """
        self.n_units = n_units
        self.learning_rate = learning_rate
        
        # Set random seed
        if random_seed is not None:
            np.random.seed(random_seed)
        
        # Initialize weights (all start equal)
        self.weights = np.ones(n_units, dtype=float)
        
        # Randomly select target MSN unit (unknown to model)
        self.target_index = np.random.randint(0, n_units)
        
    def sample(self) -> np.ndarray:
        """
        Sample active MSN units based on current weights.
        
        Uses stochastic binary activation where each unit has a probability
        proportional to its normalized weight. Expected number of active units
        is approximately 10.
        
        Returns:
            Array of indices of active units
        """
        # Normalize weights to get probabilities
        # Expected ~10 active units: probabilities sum to 10
        normalized_weights = self.weights / np.sum(self.weights)
        probabilities = normalized_weights * 10
        
        # Clip probabilities to [0, 1]
        probabilities = np.clip(probabilities, 0.0, 1.0)
        
        # Sample active units
        active_mask = np.random.rand(self.n_units) < probabilities
        active_units = np.where(active_mask)[0]
        
        return active_units
    
    def is_target_active(self, active_units: np.ndarray) -> bool:
        """
        Check if target MSN unit is active.
        
        Args:
            active_units: Array of indices of active units
            
        Returns:
            True if target unit is active, False otherwise
        """
        return self.target_index in active_units
    
    def update_weights(
        self,
        feedback: bool,
        activation_counts: np.ndarray
    ) -> None:
        """
        Update weights based on reward-modulated plasticity rule.
        
        Every 100 ms (feedback interval), weights are updated:
        - Positive feedback: Strengthen active units proportional to activation frequency
        - No feedback: Weaken active units
        
        Args:
            feedback: Boolean indicating positive reward (True) or no reward (False)
            activation_counts: Array of activation counts for each unit over feedback window
        """
        if feedback:
            # Positive reward: strengthen active units
            # Weight update proportional to activation frequency
            active_mask = activation_counts > 0
            if np.any(active_mask):
                # Update: weights += learning_rate * (count / window_size)
                update = self.learning_rate * (activation_counts / 1024.0)
                self.weights += update
        else:
            # No reward: weaken active units
            active_mask = activation_counts > 0
            if np.any(active_mask):
                # Compute mean activation for normalization
                mean_activation = np.mean(activation_counts[active_mask])
                # Update: weights -= learning_rate * (mean_activation / window_size)
                update = self.learning_rate * (mean_activation / 1024.0)
                # Only weaken units that were active
                self.weights[active_mask] -= update
        
        # Ensure weights remain non-negative
        self.weights = np.maximum(self.weights, 1e-10)
    
    def get_target_probability(self) -> float:
        """
        Get current probability of target MSN unit being active.
        
        Returns:
            Probability of target unit activation (0.0 to 1.0)
        """
        # Normalize weights
        normalized_weights = self.weights / np.sum(self.weights)
        probabilities = normalized_weights * 10
        probabilities = np.clip(probabilities, 0.0, 1.0)
        
        return probabilities[self.target_index]
    
    def get_target_index(self) -> int:
        """
        Get the target MSN unit index (for monitoring/debugging).
        
        Returns:
            Index of target MSN unit
        """
        return self.target_index
    
    def get_weights(self) -> np.ndarray:
        """
        Get current weights (for monitoring/debugging).
        
        Returns:
            Array of weights for all units
        """
        return self.weights.copy()
    
    def reset(self) -> None:
        """Reset the system to initial state."""
        self.weights = np.ones(self.n_units, dtype=float)
        self.target_index = np.random.randint(0, self.n_units)

