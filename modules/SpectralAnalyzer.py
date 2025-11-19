"""
Spectral Analysis for Neurofeedback.

This module implements real-time spectral analysis for detecting peak alpha frequency (PAF)
and computing upper alpha frequency (UAF) band power for neurofeedback signals.

Reference:
    Davelaar, E.J. (2018). Mechanisms of Neurofeedback: A Computation-theoretic Approach.
    Neuroscience, 378, 175-188.

Author: Generated from specification
"""

import numpy as np
from typing import Optional, Tuple, List
from scipy import signal as sp_signal
from collections import deque


class SpectralAnalyzer:
    """
    Real-time spectral analyzer for EEG neurofeedback.
    
    Implements FFT-based spectral analysis with:
    - Peak Alpha Frequency (PAF) detection in 8-12 Hz band
    - Upper Alpha Frequency (UAF) band definition [PAF, PAF+2 Hz]
    - Band power computation for feedback thresholding
    
    Attributes:
        sampling_rate (float): Sampling rate in Hz (default: 1000)
        window_size (int): FFT window size in samples (default: 1024)
        update_interval (int): Feedback update interval in ms (default: 100)
        alpha_band (Tuple[float, float]): Alpha frequency band (default: (8, 12))
        paf (Optional[float]): Detected peak alpha frequency in Hz
        uaf_band (Optional[Tuple[float, float]]): Upper alpha frequency band
        baseline_mean (Optional[float]): Baseline UAF power mean
        baseline_std (Optional[float]): Baseline UAF power standard deviation
        eeg_buffer (deque): Sliding window buffer for EEG samples
    """
    
    def __init__(
        self,
        sampling_rate: float = 1000.0,
        window_size: int = 1024,
        update_interval: int = 100,
        alpha_band: Tuple[float, float] = (8, 12),
        bandpass_low: float = 1.0,
        bandpass_high: float = 30.0
    ):
        """
        Initialize spectral analyzer.
        
        Args:
            sampling_rate: Sampling rate in Hz (samples per second)
            window_size: FFT window size in samples (1024 ms at 1 kHz)
            update_interval: Feedback update interval in milliseconds
            alpha_band: Alpha frequency band tuple (low, high) in Hz
            bandpass_low: Low cutoff frequency for bandpass filter (default: 1.0 Hz)
            bandpass_high: High cutoff frequency for bandpass filter (default: 30.0 Hz)
        """
        self.sampling_rate = sampling_rate
        self.window_size = window_size
        self.update_interval = update_interval
        self.alpha_band = alpha_band
        self.bandpass_low = bandpass_low
        self.bandpass_high = bandpass_high
        
        # Initialize state variables
        self.paf = None
        self.uaf_band = None
        self.baseline_mean = None
        self.baseline_std = None
        
        # Initialize sliding window buffer
        self.eeg_buffer = deque(maxlen=window_size)
        
        # Pre-compute frequency array for FFT
        self.freqs = np.fft.rfftfreq(window_size, d=1.0 / sampling_rate)
        
        # Pre-compute Hamming window
        self.hamming_window = sp_signal.windows.hamming(window_size)
        
        # Design bandpass filter (Butterworth, 4th order)
        # Normalize frequencies to Nyquist frequency (sampling_rate / 2)
        nyquist = sampling_rate / 2.0
        low_norm = bandpass_low / nyquist
        high_norm = bandpass_high / nyquist
        self.b, self.a = sp_signal.butter(4, [low_norm, high_norm], btype='band')
        
    def add_sample(self, eeg_value: float) -> None:
        """
        Add EEG sample to sliding window buffer.
        
        Args:
            eeg_value: EEG signal value at current time step
        """
        self.eeg_buffer.append(eeg_value)
    
    def _apply_bandpass_filter(self, signal: np.ndarray) -> np.ndarray:
        """
        Apply bandpass filter to signal.
        
        Args:
            signal: Input signal array
            
        Returns:
            Filtered signal array
        """
        # Use filtfilt for zero-phase filtering (forward and backward)
        filtered = sp_signal.filtfilt(self.b, self.a, signal)
        return filtered
    
    def compute_power_spectrum(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute power spectral density from current buffer.
        
        Returns:
            Tuple of (frequencies, power_spectral_density) arrays
        """
        if len(self.eeg_buffer) < self.window_size:
            raise ValueError(
                f"Buffer size {len(self.eeg_buffer)} < {self.window_size}. "
                "Need more samples before computing spectrum."
            )
        
        # Convert buffer to numpy array
        window_data = np.array(self.eeg_buffer)
        
        # Normalize EEG signal: scale by number of excitatory neurons (800)
        # This brings values to a reasonable range similar to paper
        window_data = window_data / 800.0
        
        # Remove DC component (mean) to avoid 0 Hz peak
        window_data = window_data - np.mean(window_data)
        
        # Apply bandpass filter (1-30 Hz) to reduce harmonics
        window_data = self._apply_bandpass_filter(window_data)
        
        # Apply Hamming window
        windowed = window_data * self.hamming_window
        
        # Compute FFT
        fft_result = np.fft.rfft(windowed)
        
        # Compute power spectral density
        psd = np.abs(fft_result)**2 / self.window_size
        
        return self.freqs, psd
    
    def find_peak_alpha_frequency(self, eeg_signal: np.ndarray) -> float:
        """
        Find peak alpha frequency (PAF) from EEG signal.
        
        Uses the last window_size samples to compute spectrum and find peak
        in the alpha band (8-12 Hz).
        
        Args:
            eeg_signal: Array of EEG signal values
            
        Returns:
            Peak alpha frequency in Hz
        """
        if len(eeg_signal) < self.window_size:
            raise ValueError(
                f"Signal length {len(eeg_signal)} < {self.window_size}. "
                "Need at least window_size samples."
            )
        
        # Use last window_size samples
        window_data = eeg_signal[-self.window_size:]
        
        # Normalize EEG signal: scale by number of excitatory neurons (800)
        # This brings values to a reasonable range similar to paper
        window_data = window_data / 800.0
        
        # Remove DC component (mean) to avoid 0 Hz peak
        window_data = window_data - np.mean(window_data)
        
        # Apply bandpass filter (1-30 Hz) to reduce harmonics
        window_data = self._apply_bandpass_filter(window_data)
        
        # Apply Hamming window
        windowed = window_data * self.hamming_window
        
        # Compute FFT
        fft_result = np.fft.rfft(windowed)
        psd = np.abs(fft_result)**2 / self.window_size
        
        # Find peak in alpha band
        alpha_mask = (self.freqs >= self.alpha_band[0]) & (self.freqs <= self.alpha_band[1])
        
        if not np.any(alpha_mask):
            raise ValueError(
                f"No frequencies found in alpha band {self.alpha_band} Hz. "
                f"Available frequencies: {self.freqs[0]:.2f} - {self.freqs[-1]:.2f} Hz"
            )
        
        alpha_freqs = self.freqs[alpha_mask]
        alpha_psd = psd[alpha_mask]
        
        # Find frequency with maximum power
        peak_idx = np.argmax(alpha_psd)
        paf = alpha_freqs[peak_idx]
        
        return paf
    
    def set_baseline(
        self,
        eeg_signal: np.ndarray,
        paf: Optional[float] = None
    ) -> Tuple[float, float]:
        """
        Establish baseline statistics from EEG signal.
        
        Computes PAF (if not provided), defines UAF band, and computes baseline
        UAF power statistics.
        
        Args:
            eeg_signal: Array of baseline EEG signal values
            paf: Optional pre-computed peak alpha frequency (if None, will compute)
            
        Returns:
            Tuple of (baseline_mean, baseline_std) for UAF power
        """
        # Find PAF if not provided
        if paf is None:
            self.paf = self.find_peak_alpha_frequency(eeg_signal)
        else:
            self.paf = paf
        
        # Define UAF band: [PAF, PAF+2] Hz
        self.uaf_band = (self.paf, self.paf + 2.0)
        
        # Compute UAF power for multiple windows in baseline
        baseline_uaf_values = []
        
        # Slide window through baseline signal
        step_size = self.update_interval  # ms
        for i in range(0, len(eeg_signal) - self.window_size + 1, step_size):
            window = eeg_signal[i:i + self.window_size]
            
            # Normalize EEG signal: scale by number of excitatory neurons (800)
            # This brings values to a reasonable range similar to paper
            window = window / 800.0
            
            # Remove DC component (mean) to avoid 0 Hz peak
            window = window - np.mean(window)
            
            # Apply bandpass filter (1-30 Hz) to reduce harmonics
            window = self._apply_bandpass_filter(window)
            
            # Apply Hamming window
            windowed = window * self.hamming_window
            
            # Compute FFT
            fft_result = np.fft.rfft(windowed)
            psd = np.abs(fft_result)**2 / self.window_size
            
            # Compute band power in UAF
            band_mask = (self.freqs >= self.uaf_band[0]) & (self.freqs <= self.uaf_band[1])
            
            if np.any(band_mask):
                uaf_power = np.mean(psd[band_mask])
                baseline_uaf_values.append(uaf_power)
        
        if len(baseline_uaf_values) == 0:
            raise ValueError(
                "No valid UAF power values computed from baseline. "
                "Check signal length and UAF band definition."
            )
        
        # Compute statistics
        self.baseline_mean = np.mean(baseline_uaf_values)
        self.baseline_std = np.std(baseline_uaf_values)
        
        return self.baseline_mean, self.baseline_std
    
    def compute_uaf_power(self) -> float:
        """
        Compute current UAF band power from buffer.
        
        Returns:
            Mean power spectral density in UAF band
        """
        if self.uaf_band is None:
            raise ValueError(
                "UAF band not defined. Call set_baseline() first."
            )
        
        if len(self.eeg_buffer) < self.window_size:
            raise ValueError(
                f"Buffer size {len(self.eeg_buffer)} < {self.window_size}"
            )
        
        # Get power spectrum
        _, psd = self.compute_power_spectrum()
        
        # Compute band power in UAF
        band_mask = (self.freqs >= self.uaf_band[0]) & (self.freqs <= self.uaf_band[1])
        
        if not np.any(band_mask):
            return 0.0
        
        uaf_power = np.mean(psd[band_mask])
        return uaf_power
    
    def get_feedback(self, threshold: Optional[float] = None) -> bool:
        """
        Compute feedback signal based on UAF power threshold.
        
        Args:
            threshold: Optional threshold value (default: baseline_mean)
            
        Returns:
            True if UAF power > threshold (positive feedback), False otherwise
        """
        if self.baseline_mean is None:
            raise ValueError(
                "Baseline not established. Call set_baseline() first."
            )
        
        if threshold is None:
            threshold = self.baseline_mean
        
        uaf_power = self.compute_uaf_power()
        return uaf_power > threshold
    
    def compute_uaf_distribution(
        self,
        eeg_signal: np.ndarray
    ) -> np.ndarray:
        """
        Compute UAF power distribution from EEG signal.
        
        Slides window through signal and computes UAF power at each position.
        
        Args:
            eeg_signal: Array of EEG signal values
            
        Returns:
            Array of UAF power values
        """
        if self.uaf_band is None:
            raise ValueError(
                "UAF band not defined. Call set_baseline() first."
            )
        
        uaf_values = []
        step_size = self.update_interval
        
        for i in range(0, len(eeg_signal) - self.window_size + 1, step_size):
            window = eeg_signal[i:i + self.window_size]
            
            # Normalize EEG signal: scale by number of excitatory neurons (800)
            # This brings values to a reasonable range similar to paper
            window = window / 800.0
            
            # Remove DC component (mean) to avoid 0 Hz peak
            window = window - np.mean(window)
            
            # Apply bandpass filter (1-30 Hz) to reduce harmonics
            window = self._apply_bandpass_filter(window)
            
            # Apply Hamming window
            windowed = window * self.hamming_window
            
            # Compute FFT
            fft_result = np.fft.rfft(windowed)
            psd = np.abs(fft_result)**2 / self.window_size
            
            # Compute band power in UAF
            band_mask = (self.freqs >= self.uaf_band[0]) & (self.freqs <= self.uaf_band[1])
            
            if np.any(band_mask):
                uaf_power = np.mean(psd[band_mask])
                uaf_values.append(uaf_power)
        
        return np.array(uaf_values)
    
    def get_paf(self) -> Optional[float]:
        """Get detected peak alpha frequency."""
        return self.paf
    
    def get_uaf_band(self) -> Optional[Tuple[float, float]]:
        """Get upper alpha frequency band."""
        return self.uaf_band
    
    def get_baseline_stats(self) -> Tuple[Optional[float], Optional[float]]:
        """Get baseline statistics."""
        return self.baseline_mean, self.baseline_std
    
    def reset(self) -> None:
        """Reset analyzer state (keep parameters, clear buffer)."""
        self.eeg_buffer.clear()
        # Note: baseline stats are kept for feedback computation

