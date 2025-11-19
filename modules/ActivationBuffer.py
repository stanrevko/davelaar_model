"""
Activation Buffer for Striatal Learning.

This module implements a circular buffer for tracking MSN unit activations
over a feedback window, enabling credit assignment for reward-modulated learning.

Reference:
    Davelaar, E.J. (2018). Mechanisms of Neurofeedback: A Computation-theoretic Approach.
    Neuroscience, 378, 175-188.

Author: Generated from specification
"""

import numpy as np
from typing import Optional


class ActivationBuffer:
    """
    Buffer for tracking MSN unit activations over a feedback window.
    
    Maintains a circular (sliding) buffer of activation states over the past 1024 ms
    to enable credit assignment for reward-modulated learning. The buffer automatically
    overwrites old data via circular indexing - it should NEVER be reset during training,
    as this would break the sliding window mechanism.
    
    Important: The buffer maintains a continuous sliding window. When get_counts() is called,
    it returns counts over the entire 1024ms window, which naturally includes recent
    activations. Do NOT reset the buffer between feedback updates - let it maintain
    the sliding window via circular indexing.
    
    Attributes:
        buffer (np.ndarray): Binary activation matrix (1024 ms × 1000 units)
        index (int): Current position in circular buffer
        window_size (int): Size of the sliding window in milliseconds
        n_units (int): Number of MSN units
    """
    
    def __init__(self, window_size: int = 1024, n_units: int = 1000):
        """
        Initialize activation buffer.
        
        Args:
            window_size: Size of feedback window in milliseconds (default: 1024)
            n_units: Number of MSN units (default: 1000)
        """
        self.window_size = window_size
        self.n_units = n_units
        self.buffer = np.zeros((window_size, n_units), dtype=int)
        self.index = 0
        
    def add(self, active_units: np.ndarray) -> None:
        """
        Add current activation state to buffer (circular/sliding window).
        
        Automatically overwrites the oldest data via circular indexing.
        This maintains a continuous sliding window of the last 1024ms.
        
        Args:
            active_units: Array of indices of currently active units
        """
        # Clear current row
        self.buffer[self.index, :] = 0
        # Set active units to 1
        if len(active_units) > 0:
            self.buffer[self.index, active_units] = 1
        # Advance circular buffer index
        self.index = (self.index + 1) % self.window_size
        
    def get_counts(self) -> np.ndarray:
        """
        Get activation count for each unit over the buffer window.
        
        Returns:
            Array of shape (n_units,) with activation counts
        """
        return np.sum(self.buffer, axis=0)
    
    def reset(self) -> None:
        """
        Reset the buffer to all zeros.
        
        WARNING: This should ONLY be called at the start of a new training phase,
        NOT between feedback updates during training! Resetting during training
        breaks the sliding window mechanism and severely weakens credit assignment.
        """
        self.buffer.fill(0)
        self.index = 0

