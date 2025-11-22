"""
Activation Buffer for Striatal Learning.

This module implements a circular buffer for tracking MSN unit activations
over a feedback window, enabling credit assignment for reward-modulated learning.

Reference:
    Davelaar, E.J. (2018). Mechanisms of Neurofeedback: A Computation-theoretic Approach.
    Neuroscience, 378, 175-188.

Author: Stanislav Revko (stanislav.revko@gmail.com)
"""

import numpy as np
from typing import Optional, Union


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
        
    Examples:
        >>> buffer = ActivationBuffer(window_size=1024, n_units=1000)
        >>> 
        >>> # Add activations for 1024 time steps
        >>> for t in range(1024):
        ...     active = np.random.choice(1000, size=10, replace=False)
        ...     buffer.add(active)
        >>> 
        >>> # Get activation counts
        >>> counts = buffer.get_counts()
        >>> print(counts.shape)  # (1000,)
        >>> print(counts.sum())  # ~10240 (10 active × 1024 steps)
        >>> 
        >>> # Check if buffer is full
        >>> print(buffer.is_full())  # True
    """
    
    def __init__(
        self, 
        window_size: int = 1024, 
        n_units: int = 1000,
        dtype: np.dtype = np.uint8
    ):
        """
        Initialize activation buffer.
        
        Args:
            window_size: Size of feedback window in milliseconds (default: 1024)
            n_units: Number of MSN units (default: 1000)
            dtype: Data type for buffer (default: np.uint8 for memory efficiency)
                   Use np.uint8 for binary data (1 byte per element, 8x savings vs int64)
        
        Raises:
            ValueError: If window_size or n_units are not positive integers
        """
        if window_size <= 0:
            raise ValueError(f"window_size must be positive, got {window_size}")
        if n_units <= 0:
            raise ValueError(f"n_units must be positive, got {n_units}")
        
        self.window_size = window_size
        self.n_units = n_units
        
        # Use uint8 for binary data (8x memory savings vs int64)
        # Buffer size: 1024 × 1000 × 1 byte = 1 MB (vs 8 MB with int64)
        self.buffer = np.zeros((window_size, n_units), dtype=dtype)
        self.index = 0
        
        # Track whether buffer has been filled at least once
        self._times_filled = 0
        
        # Statistics tracking (optional, for debugging)
        self._total_additions = 0
        self._last_n_active = 0
        
    def add(self, active_units: Union[np.ndarray, list, tuple, None]) -> None:
        """
        Add current activation state to buffer (circular/sliding window).
        
        Automatically overwrites the oldest data via circular indexing.
        This maintains a continuous sliding window of the last window_size ms.
        
        Args:
            active_units: Array/list/tuple of indices of currently active units,
                         or None for no active units
        
        Raises:
            ValueError: If active_units contains invalid indices or duplicates
            
        Examples:
            >>> buffer = ActivationBuffer()
            >>> buffer.add([10, 20, 30])  # Add 3 active units
            >>> buffer.add(np.array([5, 15, 25]))  # NumPy array
            >>> buffer.add([])  # No active units
            >>> buffer.add(None)  # Also valid for no active units
        """
        # Clear current row (overwrite old data)
        self.buffer[self.index, :] = 0
        
        # Handle different input types
        if active_units is None:
            # No active units this time step
            self._last_n_active = 0
        elif len(active_units) == 0:
            # Empty array/list
            self._last_n_active = 0
        else:
            # Convert to numpy array and ensure 1D integer array
            active_units = np.asarray(active_units, dtype=np.int32).flatten()
            
            # Validate bounds
            if np.any((active_units < 0) | (active_units >= self.n_units)):
                min_idx = active_units.min() if len(active_units) > 0 else 0
                max_idx = active_units.max() if len(active_units) > 0 else 0
                raise ValueError(
                    f"active_units contains invalid indices. "
                    f"Valid range: [0, {self.n_units}), "
                    f"got min={min_idx}, max={max_idx}"
                )
            
            # Check for duplicates (optional but good for catching bugs)
            if len(active_units) != len(np.unique(active_units)):
                unique_count = len(np.unique(active_units))
                raise ValueError(
                    f"active_units contains {len(active_units) - unique_count} "
                    f"duplicate indices. Each unit should appear at most once."
                )
            
            # Set active units to 1
            self.buffer[self.index, active_units] = 1
            self._last_n_active = len(active_units)
        
        # Advance circular buffer index
        old_index = self.index
        self.index = (self.index + 1) % self.window_size
        
        # Track buffer cycles (for is_full() method)
        if self.index < old_index:  # Wrapped around to 0
            self._times_filled += 1
        
        # Update statistics
        self._total_additions += 1
    
    def get_counts(self) -> np.ndarray:
        """
        Get activation count for each unit over the buffer window.
        
        Returns the number of times each unit was active during the
        sliding window (last window_size milliseconds).
        
        Returns:
            Array of shape (n_units,) with activation counts.
            Values range from 0 (never active) to window_size (always active).
            
        Examples:
            >>> buffer = ActivationBuffer(window_size=1024, n_units=1000)
            >>> # Add same units repeatedly
            >>> for t in range(1024):
            ...     buffer.add([0, 1, 2])
            >>> counts = buffer.get_counts()
            >>> print(counts[:3])  # [1024, 1024, 1024]
            >>> print(counts[3:].sum())  # 0 (units 3+ never active)
        """
        return np.sum(self.buffer, axis=0)
    
    def get_activation_rate(self, unit_index: Optional[int] = None) -> Union[float, np.ndarray]:
        """
        Get activation rate (fraction of time active) for unit(s).
        
        Args:
            unit_index: Index of specific unit to query, or None for all units
            
        Returns:
            If unit_index is None: array of rates for all units (shape: n_units)
            If unit_index is int: single rate value for that unit
            
        Raises:
            ValueError: If unit_index is out of bounds
            
        Examples:
            >>> buffer = ActivationBuffer()
            >>> for t in range(1024):
            ...     buffer.add([42])  # Unit 42 always active
            >>> print(buffer.get_activation_rate(42))  # 1.0
            >>> print(buffer.get_activation_rate(100))  # 0.0
        """
        counts = self.get_counts()
        
        if unit_index is None:
            # Return rates for all units
            return counts / self.window_size
        else:
            # Return rate for specific unit
            if unit_index < 0 or unit_index >= self.n_units:
                raise ValueError(
                    f"unit_index must be in [0, {self.n_units}), got {unit_index}"
                )
            return counts[unit_index] / self.window_size
    
    def is_full(self) -> bool:
        """
        Check if buffer has been completely filled at least once.
        
        Returns True after the buffer has cycled through all window_size
        positions at least once. Before this, the buffer contains some
        uninitialized (zero) rows.
        
        Returns:
            True if buffer has been filled at least once, False otherwise
            
        Examples:
            >>> buffer = ActivationBuffer(window_size=1024)
            >>> print(buffer.is_full())  # False
            >>> for t in range(1024):
            ...     buffer.add([t % 1000])
            >>> print(buffer.is_full())  # True
        """
        return self._times_filled > 0
    
    def get_buffer_age(self) -> int:
        """
        Get the number of time steps since buffer was created/reset.
        
        Returns:
            Number of additions made to buffer
            
        Examples:
            >>> buffer = ActivationBuffer()
            >>> print(buffer.get_buffer_age())  # 0
            >>> for t in range(500):
            ...     buffer.add([])
            >>> print(buffer.get_buffer_age())  # 500
        """
        return self._total_additions
    
    def get_fill_fraction(self) -> float:
        """
        Get fraction of buffer that has been filled with real data.
        
        Returns value between 0.0 (empty) and 1.0 (fully filled at least once).
        
        Returns:
            Fraction of buffer filled, clamped to [0.0, 1.0]
            
        Examples:
            >>> buffer = ActivationBuffer(window_size=1024)
            >>> print(buffer.get_fill_fraction())  # 0.0
            >>> for t in range(512):
            ...     buffer.add([])
            >>> print(buffer.get_fill_fraction())  # 0.5
            >>> for t in range(512):
            ...     buffer.add([])
            >>> print(buffer.get_fill_fraction())  # 1.0
        """
        if self._times_filled > 0:
            return 1.0
        else:
            return min(self._total_additions / self.window_size, 1.0)
    
    def reset(self, reason: str = "unknown") -> None:
        """
        Reset the buffer to all zeros.
        
        WARNING: This breaks the sliding window mechanism and should only be called:
        - At the start of a new training session
        - After a warm-up period to discard transients
        - When deliberately restarting credit assignment
        - For testing/debugging purposes
        
        DO NOT call between feedback updates during normal training!
        
        Args:
            reason: Optional reason for reset (for logging/debugging)
            
        Examples:
            >>> buffer = ActivationBuffer()
            >>> for t in range(100):
            ...     buffer.add([0, 1, 2])
            >>> buffer.reset(reason="starting new session")
            >>> print(buffer.get_counts().sum())  # 0
        """
        self.buffer.fill(0)
        self.index = 0
        self._times_filled = 0
        self._total_additions = 0
        self._last_n_active = 0
        
        # Optional: log reset for debugging
        # import warnings
        # warnings.warn(f"ActivationBuffer reset: {reason}", UserWarning)
    
    def get_current_state(self) -> dict:
        """
        Get complete state of the buffer for debugging/inspection.
        
        Returns:
            Dictionary containing:
                - 'index': current circular buffer position
                - 'is_full': whether buffer has been filled once
                - 'age': number of additions
                - 'fill_fraction': fraction of buffer filled
                - 'last_n_active': number of active units in last addition
                - 'total_activations': total activations across all units
                - 'mean_activations_per_step': average units active per step
                
        Examples:
            >>> buffer = ActivationBuffer()
            >>> state = buffer.get_current_state()
            >>> print(state['is_full'])  # False
            >>> print(state['age'])  # 0
        """
        counts = self.get_counts()
        
        return {
            'index': self.index,
            'is_full': self.is_full(),
            'age': self._total_additions,
            'fill_fraction': self.get_fill_fraction(),
            'last_n_active': self._last_n_active,
            'total_activations': counts.sum(),
            'mean_activations_per_step': (
                counts.sum() / max(self._total_additions, 1)
            ),
            'window_size': self.window_size,
            'n_units': self.n_units,
            'buffer_shape': self.buffer.shape,
            'buffer_dtype': self.buffer.dtype,
            'buffer_memory_mb': self.buffer.nbytes / (1024 * 1024)
        }
    
    def __repr__(self) -> str:
        """String representation of the buffer."""
        state = self.get_current_state()
        return (
            f"ActivationBuffer("
            f"window_size={self.window_size}, "
            f"n_units={self.n_units}, "
            f"age={state['age']}, "
            f"is_full={state['is_full']}, "
            f"fill={state['fill_fraction']:.2%})"
        )
    
    def __len__(self) -> int:
        """Return the window size (maximum buffer capacity)."""
        return self.window_size