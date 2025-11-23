"""
EEG Generator using Izhikevich Spiking Neural Network.

This module implements the cortical EEG generator from Davelaar (2018) Simulation Study 1.
The network consists of 800 excitatory and 200 inhibitory Izhikevich neurons that generate
an alpha-band EEG proxy signal through low-pass filtering of excitatory membrane potentials.

Reference:
    Davelaar, E.J. (2018). Mechanisms of Neurofeedback: A Computation-theoretic Approach.
    Neuroscience, 378, 175-188.

Author: Stanislav Revko (stanislav.revko@gmail.com)
"""

import numpy as np
from typing import Optional, Tuple

# Try to import numba for JIT compilation (optional dependency)
try:
    from numba import jit
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    # Create dummy decorator if numba not available
    def jit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator

# Time step in milliseconds (two half-steps per millisecond for stability)
DT_MS = 0.5


# JIT-compiled helper functions for performance
@jit(nopython=True, cache=True)
def _compute_synaptic_input_exc(w_ee, w_ie, spikes_exc, spikes_inh):
    """Compute synaptic input to excitatory neurons."""
    I_syn_ee = np.zeros(w_ee.shape[0])
    I_syn_ie = np.zeros(w_ie.shape[1])

    if np.any(spikes_exc):
        for i in range(w_ee.shape[0]):
            for j in range(w_ee.shape[1]):
                if spikes_exc[j]:
                    I_syn_ee[i] += w_ee[i, j]

    if np.any(spikes_inh):
        for i in range(w_ie.shape[1]):
            for j in range(w_ie.shape[0]):
                if spikes_inh[j]:
                    I_syn_ie[i] += w_ie[j, i]

    return I_syn_ee, I_syn_ie


@jit(nopython=True, cache=True)
def _compute_synaptic_input_inh(w_ei, w_ii, spikes_exc, spikes_inh):
    """Compute synaptic input to inhibitory neurons."""
    I_syn_ei = np.zeros(w_ei.shape[1])
    I_syn_ii = np.zeros(w_ii.shape[0])

    if np.any(spikes_exc):
        for i in range(w_ei.shape[1]):
            for j in range(w_ei.shape[0]):
                if spikes_exc[j]:
                    I_syn_ei[i] += w_ei[j, i]

    if np.any(spikes_inh):
        for i in range(w_ii.shape[0]):
            for j in range(w_ii.shape[1]):
                if spikes_inh[j]:
                    I_syn_ii[i] += w_ii[i, j]

    return I_syn_ei, I_syn_ii


@jit(nopython=True, cache=True)
def _update_neurons_exc(v, u, I_total, a, b, c, d, dt):
    """Update excitatory neurons (one half-step)."""
    # Compute derivatives
    dv = (0.04 * v**2 + 5 * v + 140 - u + I_total) * dt
    du = a * (b * v - u) * dt

    # Update state
    v_new = v + dv
    u_new = u + du

    # Handle spikes
    spikes = v_new >= 30
    if np.any(spikes):
        for i in range(len(v_new)):
            if spikes[i]:
                v_new[i] = c[i]
                u_new[i] += d[i]

    return v_new, u_new, spikes


@jit(nopython=True, cache=True)
def _update_neurons_inh(v, u, I_total, a, b, c, d, dt):
    """Update inhibitory neurons (one half-step)."""
    # Compute derivatives
    dv = (0.04 * v**2 + 5 * v + 140 - u + I_total) * dt
    du = a * (b * v - u) * dt

    # Update state
    v_new = v + dv
    u_new = u + du

    # Handle spikes
    spikes = v_new >= 30
    if np.any(spikes):
        for i in range(len(v_new)):
            if spikes[i]:
                v_new[i] = c[i]
                u_new[i] += d[i]

    return v_new, u_new, spikes


class IzhikevichEEGGenerator:
    """
    Izhikevich spiking neural network for EEG generation.
    
    Implements a network of 800 excitatory (regular spiking) and 200 inhibitory
    (fast spiking) neurons with all-to-all connectivity. The EEG signal is computed
    as a low-pass filtered sum of excitatory membrane potentials.
    
    Attributes:
        n_exc (int): Number of excitatory neurons (default: 800)
        n_inh (int): Number of inhibitory neurons (default: 200)
        eeg_signal (float): Current EEG signal value
        v_exc (np.ndarray): Membrane potentials of excitatory neurons
        u_exc (np.ndarray): Recovery variables of excitatory neurons
        v_inh (np.ndarray): Membrane potentials of inhibitory neurons
        u_inh (np.ndarray): Recovery variables of inhibitory neurons
        a_exc (np.ndarray): Parameter a for excitatory neurons
        b_exc (np.ndarray): Parameter b for excitatory neurons
        c_exc (np.ndarray): Parameter c for excitatory neurons
        d_exc (np.ndarray): Parameter d for excitatory neurons
        a_inh (np.ndarray): Parameter a for inhibitory neurons
        b_inh (np.ndarray): Parameter b for inhibitory neurons
        c_inh (np.ndarray): Parameter c for inhibitory neurons
        d_inh (np.ndarray): Parameter d for inhibitory neurons
        w_ee (np.ndarray): Synaptic weights E->E
        w_ei (np.ndarray): Synaptic weights E->I
        w_ie (np.ndarray): Synaptic weights I->E (negative)
        w_ii (np.ndarray): Synaptic weights I->I (negative)
        spike_history_exc (list): History of excitatory spikes
        spike_history_inh (list): History of inhibitory spikes
    """
    
    def __init__(
        self,
        n_exc: int = 800,
        n_inh: int = 200,
        random_seed: Optional[int] = None,
        measurement_noise_std: float = 0.0,
        track_spikes: bool = False
    ):
        """
        Initialize the Izhikevich EEG generator network.

        Args:
            n_exc: Number of excitatory neurons
            n_inh: Number of inhibitory neurons
            random_seed: Random seed for reproducibility (None for random)
            measurement_noise_std: Standard deviation of additive white Gaussian noise
                                 added to EEG signal (default: 0.0, no noise)
            track_spikes: Whether to track spike times (default: False for performance)
        """
        self.n_exc = n_exc
        self.n_inh = n_inh
        self.n_total = n_exc + n_inh
        self.measurement_noise_std = measurement_noise_std
        self.track_spikes = track_spikes
        
        # Set random seed
        if random_seed is not None:
            np.random.seed(random_seed)
        
        # Initialize EEG signal
        self.eeg_signal = 0.0
        
        # Initialize excitatory neuron parameters (Regular Spiking)
        r_e = np.random.rand(n_exc)
        self.a_exc = np.full(n_exc, 0.02)
        self.b_exc = np.full(n_exc, 0.2)
        self.c_exc = -65 + 15 * r_e**2  # Range: [-65, -50]
        self.d_exc = 8 - 6 * r_e**2      # Range: [2, 8]
        
        # Initialize inhibitory neuron parameters (Fast Spiking / Low Threshold Spiking)
        r_i = np.random.rand(n_inh)
        self.a_inh = 0.02 + 0.08 * r_i   # Range: [0.02, 0.10]
        self.b_inh = 0.25 - 0.05 * r_i   # Range: [0.20, 0.25]
        self.c_inh = np.full(n_inh, -65)
        self.d_inh = np.full(n_inh, 2)
        
        # Initialize membrane potentials and recovery variables
        # Start closer to rest potential to reduce transients
        self.v_exc = np.random.uniform(-65, -60, n_exc)  # Closer to rest
        self.u_exc = self.b_exc * self.v_exc
        
        self.v_inh = np.random.uniform(-65, -60, n_inh)  # Closer to rest
        self.u_inh = self.b_inh * self.v_inh
        
        # Initialize EEG signal to approximate steady-state value
        # Based on typical sum of excitatory potentials at rest
        sum_v_exc_init = np.sum(self.v_exc)
        self.eeg_signal = sum_v_exc_init  # Start at current sum instead of 0
        
        # Store initial values for reset
        self.v_exc_init = self.v_exc.copy()
        self.u_exc_init = self.u_exc.copy()
        self.v_inh_init = self.v_inh.copy()
        self.u_inh_init = self.u_inh.copy()
        
        # Create synapses (all-to-all connectivity)
        # E -> E
        self.w_ee = np.random.uniform(0, 0.5, size=(n_exc, n_exc))
        
        # E -> I
        self.w_ei = np.random.uniform(0, 0.5, size=(n_exc, n_inh))
        
        # I -> E (inhibitory, negative weights)
        self.w_ie = -np.random.uniform(0, 1.0, size=(n_inh, n_exc))
        
        # I -> I (inhibitory, negative weights)
        self.w_ii = -np.random.uniform(0, 1.0, size=(n_inh, n_inh))
        
        # Spike history for tracking (optional, for debugging)
        if self.track_spikes:
            self.spike_history_exc = []
            self.spike_history_inh = []
        else:
            self.spike_history_exc = None
            self.spike_history_inh = None
        self.current_time = 0.0
        
    def step(
        self, 
        thalamic_modulation: float = 0.0,
        I_exc: Optional[np.ndarray] = None
    ) -> float:
        """
        Advance the network by one time step (1 ms) and update EEG signal.
        
        Args:
            thalamic_modulation: Additional input to excitatory neurons (0.0 = baseline, 1.0 = target active).
                                 Only used if I_exc is None.
            I_exc: Optional array of input currents for all excitatory neurons (shape: (n_exc,)).
                   If provided, this replaces the default thalamic input. If None, uses thalamic_modulation.
            
        Returns:
            Current EEG signal value
        """
        # Set thalamic input (Gaussian noise) once per millisecond and reuse for the two half-steps
        if I_exc is not None:
            # Use provided input currents for excitatory neurons
            if len(I_exc) != self.n_exc:
                raise ValueError(
                    f"I_exc must have length {self.n_exc}, got {len(I_exc)}"
                )
            I_thalamic_exc = I_exc.copy()
        else:
            # Use default thalamic input with modulation
            I_thalamic_exc = np.random.normal(5.0 + thalamic_modulation, 1.0, self.n_exc)
        
        I_thalamic_inh = np.random.normal(2.0, 1.0, self.n_inh)
        
        # Two half-step (0.5 ms) updates for better numerical stability (Izhikevich recommendation)
        for _ in range(2):
            # Detect spikes (before updating)
            spikes_exc = self.v_exc >= 30
            spikes_inh = self.v_inh >= 30

            # Compute synaptic input from spikes (JIT-compiled for speed)
            I_syn_ee, I_syn_ie = _compute_synaptic_input_exc(
                self.w_ee, self.w_ie, spikes_exc, spikes_inh
            )
            I_syn_ei, I_syn_ii = _compute_synaptic_input_inh(
                self.w_ei, self.w_ii, spikes_exc, spikes_inh
            )

            # Total input current
            I_exc_total = I_thalamic_exc + I_syn_ee + I_syn_ie
            I_inh_total = I_thalamic_inh + I_syn_ei + I_syn_ii

            # Update neurons (JIT-compiled for speed)
            self.v_exc, self.u_exc, spike_mask_exc = _update_neurons_exc(
                self.v_exc, self.u_exc, I_exc_total,
                self.a_exc, self.b_exc, self.c_exc, self.d_exc, DT_MS
            )
            self.v_inh, self.u_inh, spike_mask_inh = _update_neurons_inh(
                self.v_inh, self.u_inh, I_inh_total,
                self.a_inh, self.b_inh, self.c_inh, self.d_inh, DT_MS
            )

            # Track spikes if enabled
            if self.track_spikes:
                if np.any(spike_mask_exc):
                    spike_indices = np.where(spike_mask_exc)[0]
                    for idx in spike_indices:
                        self.spike_history_exc.append((self.current_time, idx))
                if np.any(spike_mask_inh):
                    spike_indices = np.where(spike_mask_inh)[0]
                    for idx in spike_indices:
                        self.spike_history_inh.append((self.current_time, idx))

            # Advance simulation time by half-step
            self.current_time += DT_MS
        
        # Compute EEG signal once per millisecond: low-pass filtered sum of excitatory potentials
        sum_v_exc = np.sum(self.v_exc)
        self.eeg_signal = 0.9 * self.eeg_signal + 0.1 * sum_v_exc
        
        # Add measurement noise (white Gaussian noise) if specified
        if self.measurement_noise_std > 0:
            measurement_noise = np.random.normal(0.0, self.measurement_noise_std)
            self.eeg_signal += measurement_noise
        
        return self.eeg_signal
    
    def warmup(self, duration_ms: int = 1000) -> None:
        """
        Warm up the network to reduce initialization transients.
        
        Runs the network for a period without recording, allowing it to
        settle into a stable oscillatory state.
        
        Args:
            duration_ms: Duration of warm-up period in milliseconds (default: 1000)
        """
        # Run network without recording spikes
        for _ in range(duration_ms):
            self.step()
        
        # Clear spike history from warm-up
        if self.track_spikes:
            self.spike_history_exc = []
            self.spike_history_inh = []
        self.current_time = 0.0
    
    def get_eeg_signal(self) -> float:
        """
        Get the current EEG signal value.
        
        Returns:
            Current EEG signal
        """
        return self.eeg_signal
    
    def reset(self) -> None:
        """Reset the network to initial state."""
        self.v_exc = self.v_exc_init.copy()
        self.u_exc = self.u_exc_init.copy()
        self.v_inh = self.v_inh_init.copy()
        self.u_inh = self.u_inh_init.copy()
        self.eeg_signal = 0.0
        self.current_time = 0.0
        if self.track_spikes:
            self.spike_history_exc = []
            self.spike_history_inh = []
        
    def get_spike_times(self) -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
        """
        Get spike times for excitatory and inhibitory neurons.

        Returns:
            Tuple of ((exc_times, exc_indices), (inh_times, inh_indices))
        """
        if not self.track_spikes:
            raise RuntimeError("Spike tracking is disabled. Enable with track_spikes=True")

        if len(self.spike_history_exc) == 0:
            exc_times = np.array([])
            exc_indices = np.array([])
        else:
            exc_times, exc_indices = zip(*self.spike_history_exc)
            exc_times = np.array(exc_times)
            exc_indices = np.array(exc_indices)
        
        if len(self.spike_history_inh) == 0:
            inh_times = np.array([])
            inh_indices = np.array([])
        else:
            inh_times, inh_indices = zip(*self.spike_history_inh)
            inh_times = np.array(inh_times)
            inh_indices = np.array(inh_indices)
        
        return (exc_times, exc_indices), (inh_times, inh_indices)
