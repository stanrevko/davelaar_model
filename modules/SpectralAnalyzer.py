"""
Spectral Analysis for Neurofeedback.

This module implements real-time spectral analysis for detecting peak alpha frequency (PAF)
and computing upper alpha frequency (UAF) band power for neurofeedback signals.

Reference:
    Davelaar, E.J. (2018). Mechanisms of Neurofeedback: A Computation-theoretic Approach.
    Neuroscience, 378, 175-188.
    
    Zoefel, B., et al. (2011). Neurofeedback training of the upper alpha frequency band
    in EEG improves cognitive performance. Neuroimage, 54(2), 1427-1431.

Author: Stanislav Revko (stanislav.revko@gmail.com)
Date: November 22, 2025
"""

import numpy as np
from typing import Optional, Tuple, List, Dict
from scipy import signal as sp_signal
from collections import deque
import warnings


class SpectralAnalyzer:
    """
    Real-time spectral analyzer for EEG neurofeedback.
    
    Implements FFT-based spectral analysis following Zoefel et al. (2011) protocol:
    - Peak Alpha Frequency (PAF) detection in 8-12 Hz band
    - Upper Alpha Frequency (UAF) band definition [PAF, PAF+2 Hz]
    - Band power computation for feedback thresholding
    
    The analyzer maintains a sliding window buffer and performs spectral analysis
    every update_interval milliseconds. Preprocessing includes DC removal, optional
    bandpass filtering, and Hamming windowing before FFT.
    
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
        
    Examples:
        >>> analyzer = SpectralAnalyzer()
        >>> 
        >>> # Baseline recording (5 minutes)
        >>> baseline_eeg = []
        >>> for t in range(300000):
        ...     eeg_value = eeg_generator.step()
        ...     baseline_eeg.append(eeg_value)
        >>> 
        >>> # Establish baseline
        >>> baseline_mean, baseline_std = analyzer.set_baseline(np.array(baseline_eeg))
        >>> print(f"PAF: {analyzer.get_paf():.2f} Hz")
        >>> print(f"UAF band: {analyzer.get_uaf_band()}")
        >>> 
        >>> # Real-time feedback
        >>> for t in range(60000):
        ...     eeg_value = eeg_generator.step()
        ...     analyzer.add_sample(eeg_value)
        ...     
        ...     if t % 100 == 0 and len(analyzer.eeg_buffer) == 1024:
        ...         feedback = analyzer.get_feedback()
        ...         # Use feedback for learning...
    """
    
    def __init__(
        self,
        sampling_rate: float = 1000.0,
        window_size: int = 1024,
        update_interval: int = 100,
        alpha_band: Tuple[float, float] = (8.0, 12.0),
        apply_bandpass: bool = False,
        bandpass_low: float = 1.0,
        bandpass_high: float = 30.0,
        normalize_by_n_neurons: bool = True,
        n_neurons: int = 800
    ):
        """
        Initialize spectral analyzer.
        
        Args:
            sampling_rate: Sampling rate in Hz (samples per second)
            window_size: FFT window size in samples (1024 ms at 1 kHz)
            update_interval: Feedback update interval in milliseconds
            alpha_band: Alpha frequency band tuple (low, high) in Hz
            apply_bandpass: Whether to apply bandpass filter (default: False)
            bandpass_low: Low cutoff frequency for bandpass filter (default: 1.0 Hz)
            bandpass_high: High cutoff frequency for bandpass filter (default: 30.0 Hz)
            normalize_by_n_neurons: Whether to normalize EEG by number of neurons (default: True)
            n_neurons: Number of excitatory neurons for normalization (default: 800)
            
        Raises:
            ValueError: If parameters are invalid
        """
        # Validate parameters
        if sampling_rate <= 0:
            raise ValueError(f"sampling_rate must be positive, got {sampling_rate}")
        if window_size <= 0:
            raise ValueError(f"window_size must be positive, got {window_size}")
        if update_interval <= 0:
            raise ValueError(f"update_interval must be positive, got {update_interval}")
        if alpha_band[0] >= alpha_band[1]:
            raise ValueError(
                f"alpha_band[0] must be < alpha_band[1], got {alpha_band}"
            )
        if alpha_band[0] < 0:
            raise ValueError(f"alpha_band frequencies must be positive, got {alpha_band}")
        
        # Check Nyquist frequency
        nyquist = sampling_rate / 2.0
        if alpha_band[1] >= nyquist:
            raise ValueError(
                f"alpha_band[1] ({alpha_band[1]} Hz) must be < Nyquist frequency "
                f"({nyquist} Hz)"
            )
        
        self.sampling_rate = sampling_rate
        self.window_size = window_size
        self.update_interval = update_interval
        self.alpha_band = alpha_band
        self.apply_bandpass = apply_bandpass
        self.bandpass_low = bandpass_low
        self.bandpass_high = bandpass_high
        self.normalize_by_n_neurons = normalize_by_n_neurons
        self.n_neurons = n_neurons
        
        # Initialize state variables
        self.paf = None
        self.uaf_band = None
        self.baseline_mean = None
        self.baseline_std = None
        self.baseline_uaf_values = None  # Store for distribution analysis
        
        # Initialize sliding window buffer
        self.eeg_buffer = deque(maxlen=window_size)
        
        # Pre-compute frequency array for FFT
        # rfft returns frequencies from 0 to Nyquist
        self.freqs = np.fft.rfftfreq(window_size, d=1.0 / sampling_rate)
        
        # Pre-compute Hamming window
        self.hamming_window = sp_signal.windows.hamming(window_size)
        
        # Design bandpass filter if needed
        if apply_bandpass:
            if bandpass_low >= bandpass_high:
                raise ValueError(
                    f"bandpass_low must be < bandpass_high, "
                    f"got {bandpass_low} >= {bandpass_high}"
                )
            if bandpass_high >= nyquist:
                raise ValueError(
                    f"bandpass_high ({bandpass_high} Hz) must be < Nyquist "
                    f"frequency ({nyquist} Hz)"
                )
            
            # Normalize frequencies to Nyquist frequency
            low_norm = bandpass_low / nyquist
            high_norm = bandpass_high / nyquist
            
            # Ensure normalized frequencies are in valid range (0, 1)
            low_norm = np.clip(low_norm, 0.01, 0.99)
            high_norm = np.clip(high_norm, 0.01, 0.99)
            
            # Design Butterworth bandpass filter (4th order)
            self.b, self.a = sp_signal.butter(
                4, [low_norm, high_norm], btype='band'
            )
        else:
            self.b, self.a = None, None
        
        # Statistics tracking
        self._n_spectra_computed = 0
        self._last_uaf_power = None
        
    def add_sample(self, eeg_value: float) -> None:
        """
        Add EEG sample to sliding window buffer.
        
        Args:
            eeg_value: EEG signal value at current time step
            
        Raises:
            ValueError: If eeg_value is NaN or Inf
            
        Examples:
            >>> analyzer = SpectralAnalyzer()
            >>> analyzer.add_sample(-52000.0)  # Typical EEG value
            >>> print(len(analyzer.eeg_buffer))  # 1
        """
        # Validate input
        if not np.isfinite(eeg_value):
            raise ValueError(
                f"eeg_value must be finite, got {eeg_value}"
            )
        
        self.eeg_buffer.append(float(eeg_value))
    
    def add_samples(self, eeg_values: np.ndarray) -> None:
        """
        Add multiple EEG samples to buffer.
        
        Args:
            eeg_values: Array of EEG signal values
            
        Examples:
            >>> analyzer = SpectralAnalyzer()
            >>> eeg_data = np.random.randn(1000) * 10000
            >>> analyzer.add_samples(eeg_data)
            >>> print(len(analyzer.eeg_buffer))  # 1000 (up to maxlen=1024)
        """
        eeg_values = np.asarray(eeg_values).flatten()
        
        if not np.all(np.isfinite(eeg_values)):
            raise ValueError("eeg_values contains NaN or Inf")
        
        for value in eeg_values:
            self.eeg_buffer.append(float(value))
    
    def _preprocess_signal(self, signal: np.ndarray) -> np.ndarray:
        """
        Preprocess EEG signal before spectral analysis.
        
        Preprocessing steps:
        1. Normalize by number of neurons (optional)
        2. Remove DC component (mean)
        3. Apply bandpass filter (optional)
        
        Args:
            signal: Input EEG signal array
            
        Returns:
            Preprocessed signal array
        """
        # Make a copy to avoid modifying original
        processed = signal.copy()
        
        # Step 1: Normalize by number of excitatory neurons
        # This brings the signal to a reasonable range (e.g., -100 to +100)
        if self.normalize_by_n_neurons:
            processed = processed / self.n_neurons
        
        # Step 2: Remove DC component (mean) to avoid 0 Hz peak
        processed = processed - np.mean(processed)
        
        # Step 3: Apply bandpass filter if enabled
        if self.apply_bandpass and self.b is not None:
            try:
                # Use filtfilt for zero-phase filtering
                processed = sp_signal.filtfilt(self.b, self.a, processed)
            except ValueError as e:
                warnings.warn(
                    f"Bandpass filtering failed: {e}. Using unfiltered signal.",
                    RuntimeWarning
                )
        
        return processed
    
    def compute_power_spectrum(
        self, 
        signal: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute power spectral density.
        
        Args:
            signal: Optional signal array. If None, uses current buffer.
            
        Returns:
            Tuple of (frequencies, power_spectral_density) arrays
            
        Raises:
            ValueError: If signal/buffer is too short
            
        Examples:
            >>> analyzer = SpectralAnalyzer()
            >>> # Fill buffer...
            >>> freqs, psd = analyzer.compute_power_spectrum()
            >>> print(freqs.shape, psd.shape)  # (513,) (513,) for 1024-point FFT
        """
        # Get signal
        if signal is None:
            if len(self.eeg_buffer) < self.window_size:
                raise ValueError(
                    f"Buffer size {len(self.eeg_buffer)} < {self.window_size}. "
                    "Need more samples before computing spectrum."
                )
            window_data = np.array(self.eeg_buffer)
        else:
            signal = np.asarray(signal).flatten()
            if len(signal) < self.window_size:
                raise ValueError(
                    f"Signal length {len(signal)} < {self.window_size}"
                )
            # Use last window_size samples
            window_data = signal[-self.window_size:]
        
        # Preprocess signal
        window_data = self._preprocess_signal(window_data)
        
        # Apply Hamming window to reduce spectral leakage
        windowed = window_data * self.hamming_window
        
        # Compute FFT (real FFT for real-valued signals)
        fft_result = np.fft.rfft(windowed)
        
        # Compute power spectral density
        # PSD = |FFT|² / N
        psd = np.abs(fft_result)**2 / self.window_size
        
        # Track statistics
        self._n_spectra_computed += 1
        
        return self.freqs, psd
    
    def find_peak_alpha_frequency(
        self, 
        eeg_signal: Optional[np.ndarray] = None
    ) -> float:
        """
        Find peak alpha frequency (PAF) from EEG signal.
        
        Computes spectrum and finds frequency with maximum power in the
        alpha band (default: 8-12 Hz).
        
        Args:
            eeg_signal: Optional array of EEG signal values. If None, uses buffer.
            
        Returns:
            Peak alpha frequency in Hz
            
        Raises:
            ValueError: If signal is too short or no alpha peak found
            
        Examples:
            >>> analyzer = SpectralAnalyzer()
            >>> baseline_eeg = np.random.randn(300000) * 10000
            >>> paf = analyzer.find_peak_alpha_frequency(baseline_eeg)
            >>> print(f"PAF: {paf:.2f} Hz")  # e.g., 10.25 Hz
        """
        # Get power spectrum
        freqs, psd = self.compute_power_spectrum(signal=eeg_signal)
        
        # Find peak in alpha band
        alpha_mask = (freqs >= self.alpha_band[0]) & (freqs <= self.alpha_band[1])
        
        if not np.any(alpha_mask):
            raise ValueError(
                f"No frequencies found in alpha band {self.alpha_band} Hz. "
                f"Available frequencies: {freqs[0]:.2f} - {freqs[-1]:.2f} Hz"
            )
        
        alpha_freqs = freqs[alpha_mask]
        alpha_psd = psd[alpha_mask]
        
        if len(alpha_psd) == 0:
            raise ValueError("No power values in alpha band")
        
        # Find frequency with maximum power
        peak_idx = np.argmax(alpha_psd)
        paf = alpha_freqs[peak_idx]
        
        # Validate PAF is reasonable
        if not (self.alpha_band[0] <= paf <= self.alpha_band[1]):
            warnings.warn(
                f"Detected PAF ({paf:.2f} Hz) outside alpha band {self.alpha_band}",
                RuntimeWarning
            )
        
        return float(paf)
    
    def set_baseline(
        self,
        eeg_signal: np.ndarray,
        paf: Optional[float] = None
    ) -> Tuple[float, float]:
        """
        Establish baseline statistics from EEG signal.
        
        Computes:
        1. PAF (if not provided)
        2. UAF band [PAF, PAF+2 Hz]
        3. Baseline UAF power statistics (mean and std)
        
        Args:
            eeg_signal: Array of baseline EEG signal values
            paf: Optional pre-computed peak alpha frequency
            
        Returns:
            Tuple of (baseline_mean, baseline_std) for UAF power
            
        Raises:
            ValueError: If signal is too short or baseline computation fails
            
        Examples:
            >>> analyzer = SpectralAnalyzer()
            >>> baseline_eeg = []
            >>> for t in range(300000):  # 5 minutes
            ...     baseline_eeg.append(eeg_generator.step())
            >>> 
            >>> mean, std = analyzer.set_baseline(np.array(baseline_eeg))
            >>> print(f"Baseline UAF: {mean:.2f} ± {std:.2f}")
        """
        eeg_signal = np.asarray(eeg_signal).flatten()
        
        if len(eeg_signal) < self.window_size:
            raise ValueError(
                f"Baseline signal length {len(eeg_signal)} < window_size "
                f"({self.window_size}). Need at least {self.window_size} samples."
            )
        
        # Find PAF if not provided
        if paf is None:
            self.paf = self.find_peak_alpha_frequency(eeg_signal)
        else:
            if not (self.alpha_band[0] <= paf <= self.alpha_band[1]):
                warnings.warn(
                    f"Provided PAF ({paf:.2f} Hz) outside alpha band {self.alpha_band}",
                    RuntimeWarning
                )
            self.paf = float(paf)
        
        # Define UAF band: [PAF, PAF+2] Hz (Zoefel et al., 2011)
        self.uaf_band = (self.paf, self.paf + 2.0)
        
        # Validate UAF band is within valid frequency range
        nyquist = self.sampling_rate / 2.0
        if self.uaf_band[1] >= nyquist:
            warnings.warn(
                f"UAF band upper limit ({self.uaf_band[1]:.2f} Hz) >= Nyquist "
                f"frequency ({nyquist} Hz). Results may be unreliable.",
                RuntimeWarning
            )
        
        # Compute UAF power for multiple windows in baseline
        baseline_uaf_values = []
        
        # Slide window through baseline signal
        step_size = self.update_interval  # ms (typically 100)
        n_windows = (len(eeg_signal) - self.window_size) // step_size + 1
        
        if n_windows < 10:
            warnings.warn(
                f"Only {n_windows} windows available in baseline. "
                f"Consider longer baseline recording for better statistics.",
                RuntimeWarning
            )
        
        for i in range(0, len(eeg_signal) - self.window_size + 1, step_size):
            window = eeg_signal[i:i + self.window_size]
            
            # Preprocess window
            window = self._preprocess_signal(window)
            
            # Apply Hamming window
            windowed = window * self.hamming_window
            
            # Compute FFT
            fft_result = np.fft.rfft(windowed)
            psd = np.abs(fft_result)**2 / self.window_size
            
            # Compute band power in UAF
            band_mask = (
                (self.freqs >= self.uaf_band[0]) & 
                (self.freqs <= self.uaf_band[1])
            )
            
            if np.any(band_mask):
                # Mean PSD in UAF band
                uaf_power = np.mean(psd[band_mask])
                baseline_uaf_values.append(uaf_power)
        
        if len(baseline_uaf_values) == 0:
            raise ValueError(
                "No valid UAF power values computed from baseline. "
                "Check signal length and UAF band definition."
            )
        
        # Store baseline UAF values for distribution analysis
        self.baseline_uaf_values = np.array(baseline_uaf_values)
        
        # Compute statistics
        self.baseline_mean = np.mean(self.baseline_uaf_values)
        self.baseline_std = np.std(self.baseline_uaf_values)
        
        # Validate baseline statistics
        if self.baseline_mean <= 0:
            warnings.warn(
                f"Baseline mean ({self.baseline_mean}) <= 0. "
                f"This may indicate preprocessing issues.",
                RuntimeWarning
            )
        
        return self.baseline_mean, self.baseline_std
    
    def compute_uaf_power(self, signal: Optional[np.ndarray] = None) -> float:
        """
        Compute current UAF band power.
        
        Args:
            signal: Optional signal array. If None, uses current buffer.
            
        Returns:
            Mean power spectral density in UAF band
            
        Raises:
            ValueError: If UAF band not defined or buffer too short
            
        Examples:
            >>> analyzer = SpectralAnalyzer()
            >>> # After set_baseline()...
            >>> analyzer.add_samples(new_eeg_data)
            >>> uaf_power = analyzer.compute_uaf_power()
            >>> print(f"UAF power: {uaf_power:.4f}")
        """
        if self.uaf_band is None:
            raise ValueError(
                "UAF band not defined. Call set_baseline() first."
            )
        
        # Get power spectrum
        _, psd = self.compute_power_spectrum(signal=signal)
        
        # Compute band power in UAF
        band_mask = (
            (self.freqs >= self.uaf_band[0]) & 
            (self.freqs <= self.uaf_band[1])
        )
        
        if not np.any(band_mask):
            warnings.warn(
                f"No frequencies found in UAF band {self.uaf_band}",
                RuntimeWarning
            )
            return 0.0
        
        uaf_power = np.mean(psd[band_mask])
        
        # Store for monitoring
        self._last_uaf_power = float(uaf_power)
        
        return float(uaf_power)
    
    def get_feedback(
        self, 
        threshold: Optional[float] = None,
        signal: Optional[np.ndarray] = None
    ) -> bool:
        """
        Compute feedback signal based on UAF power threshold.
        
        Args:
            threshold: Optional threshold value (default: baseline_mean)
            signal: Optional signal array. If None, uses current buffer.
            
        Returns:
            True if UAF power > threshold (positive feedback), False otherwise
            
        Raises:
            ValueError: If baseline not established
            
        Examples:
            >>> analyzer = SpectralAnalyzer()
            >>> # After set_baseline()...
            >>> feedback = analyzer.get_feedback()
            >>> print("Reward!" if feedback else "No reward")
            >>> 
            >>> # With custom threshold (e.g., 75th percentile)
            >>> threshold = np.percentile(baseline_uaf_values, 75)
            >>> feedback = analyzer.get_feedback(threshold=threshold)
        """
        if self.baseline_mean is None:
            raise ValueError(
                "Baseline not established. Call set_baseline() first."
            )
        
        # Use baseline mean as default threshold
        if threshold is None:
            threshold = self.baseline_mean
        
        # Compute current UAF power
        uaf_power = self.compute_uaf_power(signal=signal)
        
        # Binary feedback: reward if above threshold
        return uaf_power > threshold
    
    def compute_uaf_distribution(
        self,
        eeg_signal: np.ndarray
    ) -> np.ndarray:
        """
        Compute UAF power distribution from EEG signal.
        
        Slides window through signal and computes UAF power at each position.
        Useful for analyzing training effects and comparing pre/post distributions.
        
        Args:
            eeg_signal: Array of EEG signal values
            
        Returns:
            Array of UAF power values
            
        Raises:
            ValueError: If UAF band not defined or signal too short
            
        Examples:
            >>> analyzer = SpectralAnalyzer()
            >>> # After set_baseline()...
            >>> 
            >>> # Compute post-training distribution
            >>> post_uaf_dist = analyzer.compute_uaf_distribution(post_training_eeg)
            >>> 
            >>> # Compare to baseline
            >>> import matplotlib.pyplot as plt
            >>> plt.hist(analyzer.baseline_uaf_values, bins=50, alpha=0.5, label='Baseline')
            >>> plt.hist(post_uaf_dist, bins=50, alpha=0.5, label='Post')
            >>> plt.legend()
        """
        if self.uaf_band is None:
            raise ValueError(
                "UAF band not defined. Call set_baseline() first."
            )
        
        eeg_signal = np.asarray(eeg_signal).flatten()
        
        if len(eeg_signal) < self.window_size:
            raise ValueError(
                f"Signal length {len(eeg_signal)} < window_size ({self.window_size})"
            )
        
        uaf_values = []
        step_size = self.update_interval
        
        for i in range(0, len(eeg_signal) - self.window_size + 1, step_size):
            window = eeg_signal[i:i + self.window_size]
            
            # Preprocess window
            window = self._preprocess_signal(window)
            
            # Apply Hamming window
            windowed = window * self.hamming_window
            
            # Compute FFT
            fft_result = np.fft.rfft(windowed)
            psd = np.abs(fft_result)**2 / self.window_size
            
            # Compute band power in UAF
            band_mask = (
                (self.freqs >= self.uaf_band[0]) & 
                (self.freqs <= self.uaf_band[1])
            )
            
            if np.any(band_mask):
                uaf_power = np.mean(psd[band_mask])
                uaf_values.append(uaf_power)
        
        return np.array(uaf_values)
    
    def get_paf(self) -> Optional[float]:
        """
        Get detected peak alpha frequency.
        
        Returns:
            PAF in Hz, or None if not yet computed
        """
        return self.paf
    
    def get_uaf_band(self) -> Optional[Tuple[float, float]]:
        """
        Get upper alpha frequency band.
        
        Returns:
            Tuple of (low, high) frequencies in Hz, or None if not yet defined
        """
        return self.uaf_band
    
    def get_baseline_stats(self) -> Tuple[Optional[float], Optional[float]]:
        """
        Get baseline statistics.
        
        Returns:
            Tuple of (baseline_mean, baseline_std), or (None, None) if not set
        """
        return self.baseline_mean, self.baseline_std
    
    def get_baseline_distribution(self) -> Optional[np.ndarray]:
        """
        Get baseline UAF power distribution.
        
        Returns:
            Array of baseline UAF power values, or None if baseline not set
        """
        return self.baseline_uaf_values
    
    def get_statistics(self) -> Dict[str, any]:
        """
        Get analyzer statistics.
        
        Returns:
            Dictionary containing:
                - 'paf': Peak alpha frequency (Hz)
                - 'uaf_band': Upper alpha band (low, high) Hz
                - 'baseline_mean': Baseline UAF power mean
                - 'baseline_std': Baseline UAF power std
                - 'n_baseline_windows': Number of baseline windows
                - 'last_uaf_power': Most recent UAF power
                - 'n_spectra_computed': Total spectra computed
                - 'buffer_size': Current buffer size
                - 'buffer_full': Whether buffer is full
        """
        return {
            'paf': self.paf,
            'uaf_band': self.uaf_band,
            'baseline_mean': self.baseline_mean,
            'baseline_std': self.baseline_std,
            'n_baseline_windows': (
                len(self.baseline_uaf_values) 
                if self.baseline_uaf_values is not None 
                else 0
            ),
            'last_uaf_power': self._last_uaf_power,
            'n_spectra_computed': self._n_spectra_computed,
            'buffer_size': len(self.eeg_buffer),
            'buffer_full': len(self.eeg_buffer) == self.window_size,
            'window_size': self.window_size,
            'sampling_rate': self.sampling_rate,
            'alpha_band': self.alpha_band
        }
    
    def is_ready(self) -> bool:
        """
        Check if analyzer is ready for feedback computation.
        
        Returns:
            True if baseline is set and buffer is full
        """
        return (
            self.baseline_mean is not None and
            len(self.eeg_buffer) == self.window_size
        )
    
    def reset(self, clear_baseline: bool = False) -> None:
        """
        Reset analyzer state.
        
        Args:
            clear_baseline: If True, also clear baseline statistics.
                          If False, keep baseline for continued feedback.
                          
        Examples:
            >>> analyzer = SpectralAnalyzer()
            >>> # After training session 1...
            >>> analyzer.reset(clear_baseline=False)  # Keep baseline for session 2
            >>> 
            >>> # Starting new participant
            >>> analyzer.reset(clear_baseline=True)  # Clear everything
        """
        # Always clear buffer
        self.eeg_buffer.clear()
        self._n_spectra_computed = 0
        self._last_uaf_power = None
        
        # Optionally clear baseline
        if clear_baseline:
            self.paf = None
            self.uaf_band = None
            self.baseline_mean = None
            self.baseline_std = None
            self.baseline_uaf_values = None
    
    def __repr__(self) -> str:
        """String representation of the analyzer."""
        stats = self.get_statistics()
        paf_str = f"{stats['paf']:.2f}" if stats['paf'] is not None else "None"
        return (
            f"SpectralAnalyzer("
            f"PAF={paf_str} Hz, "
            f"UAF={stats['uaf_band']}, "
            f"baseline={stats['baseline_mean']:.4f} ± {stats['baseline_std']:.4f} "
            if stats['baseline_mean'] is not None else
            f"baseline=Not set, "
            f"buffer={stats['buffer_size']}/{stats['window_size']})"
        )


# ==============================================================================
# UNIT TESTS
# ==============================================================================

def _test_initialization():
    """Test initialization and parameter validation."""
    print("Testing initialization...")
    
    # Valid initialization
    analyzer = SpectralAnalyzer()
    assert analyzer.sampling_rate == 1000.0
    assert analyzer.window_size == 1024
    assert len(analyzer.eeg_buffer) == 0
    
    # Test invalid parameters
    try:
        SpectralAnalyzer(sampling_rate=0)
        assert False, "Should raise ValueError"
    except ValueError:
        pass
    
    try:
        SpectralAnalyzer(alpha_band=(12, 8))  # Reversed
        assert False, "Should raise ValueError"
    except ValueError:
        pass
    
    print("✓ Initialization tests passed")


def _test_buffer_management():
    """Test buffer operations."""
    print("Testing buffer management...")
    
    analyzer = SpectralAnalyzer(window_size=100)
    
    # Add single samples
    for i in range(50):
        analyzer.add_sample(float(i))
    assert len(analyzer.eeg_buffer) == 50
    
    # Add multiple samples
    analyzer.add_samples(np.arange(60))
    assert len(analyzer.eeg_buffer) == 100  # maxlen
    
    # Test buffer overflow
    analyzer.add_sample(1000.0)
    assert len(analyzer.eeg_buffer) == 100
    assert analyzer.eeg_buffer[-1] == 1000.0
    
    print("✓ Buffer management tests passed")


def _test_spectral_analysis():
    """Test spectrum computation."""
    print("Testing spectral analysis...")
    
    analyzer = SpectralAnalyzer(sampling_rate=1000, window_size=1024)
    
    # Generate synthetic signal with 10 Hz component
    t = np.arange(1024) / 1000.0  # 1.024 seconds
    signal = np.sin(2 * np.pi * 10.0 * t) * 10000  # 10 Hz sine wave
    
    analyzer.add_samples(signal)
    
    # Compute spectrum
    freqs, psd = analyzer.compute_power_spectrum()
    
    # Check shape
    assert len(freqs) == 513  # rfft of 1024 points
    assert len(psd) == 513
    
    # Find peak frequency
    peak_idx = np.argmax(psd[10:100])  # Exclude DC
    peak_freq = freqs[10 + peak_idx]
    print(f"  Detected peak: {peak_freq:.2f} Hz (expected: 10.0 Hz)")
    assert 9.5 < peak_freq < 10.5, f"Peak at {peak_freq} Hz, expected ~10 Hz"
    
    print("✓ Spectral analysis tests passed")


def _test_paf_detection():
    """Test PAF detection."""
    print("Testing PAF detection...")
    
    analyzer = SpectralAnalyzer()
    
    # Generate signal with alpha component at 10 Hz
    t = np.arange(5000) / 1000.0
    signal = np.sin(2 * np.pi * 10.0 * t) * 10000
    
    # Detect PAF
    paf = analyzer.find_peak_alpha_frequency(signal)
    print(f"  Detected PAF: {paf:.2f} Hz")
    assert 9.5 < paf < 10.5, f"PAF at {paf} Hz, expected ~10 Hz"
    
    print("✓ PAF detection tests passed")


def _test_baseline_establishment():
    """Test baseline statistics computation."""
    print("Testing baseline establishment...")
    
    analyzer = SpectralAnalyzer()
    
    # Generate 5-second baseline with 10 Hz alpha
    t = np.arange(5000) / 1000.0
    baseline_signal = np.sin(2 * np.pi * 10.0 * t) * 10000
    baseline_signal += np.random.randn(5000) * 1000  # Add noise
    
    # Set baseline
    mean, std = analyzer.set_baseline(baseline_signal)
    
    print(f"  PAF: {analyzer.get_paf():.2f} Hz")
    print(f"  UAF band: {analyzer.get_uaf_band()}")
    print(f"  Baseline: {mean:.4f} ± {std:.4f}")
    
    assert analyzer.get_paf() is not None
    assert analyzer.get_uaf_band() is not None
    assert mean > 0
    assert std > 0
    
    print("✓ Baseline establishment tests passed")


def _test_feedback_generation():
    """Test feedback signal generation."""
    print("Testing feedback generation...")
    
    analyzer = SpectralAnalyzer()
    
    # Generate baseline
    t = np.arange(5000) / 1000.0
    baseline = np.sin(2 * np.pi * 10.0 * t) * 10000
    analyzer.set_baseline(baseline)
    
    # Generate signal with higher power
    signal_high = np.sin(2 * np.pi * 10.0 * t) * 15000  # Higher amplitude
    analyzer.add_samples(signal_high[-1024:])
    
    feedback = analyzer.get_feedback()
    print(f"  Feedback (high power): {feedback}")
    
    # Generate signal with lower power
    signal_low = np.sin(2 * np.pi * 10.0 * t) * 5000  # Lower amplitude
    analyzer.eeg_buffer.clear()
    analyzer.add_samples(signal_low[-1024:])
    
    feedback = analyzer.get_feedback()
    print(f"  Feedback (low power): {feedback}")
    
    print("✓ Feedback generation tests passed")


def _test_distribution_computation():
    """Test UAF distribution computation."""
    print("Testing distribution computation...")
    
    analyzer = SpectralAnalyzer()
    
    # Generate baseline
    t = np.arange(5000) / 1000.0
    baseline = np.sin(2 * np.pi * 10.0 * t) * 10000
    analyzer.set_baseline(baseline)
    
    # Compute distribution
    distribution = analyzer.compute_uaf_distribution(baseline)
    
    print(f"  Distribution size: {len(distribution)}")
    print(f"  Mean: {np.mean(distribution):.4f}")
    print(f"  Std: {np.std(distribution):.4f}")
    
    assert len(distribution) > 0
    assert np.all(distribution >= 0)
    
    print("✓ Distribution computation tests passed")


if __name__ == "__main__":
    print("=" * 70)
    print("RUNNING SPECTRAL ANALYZER TESTS")
    print("=" * 70)
    
    _test_initialization()
    _test_buffer_management()
    _test_spectral_analysis()
    _test_paf_detection()
    _test_baseline_establishment()
    _test_feedback_generation()
    _test_distribution_computation()
    
    print("\n" + "=" * 70)
    print("ALL TESTS PASSED ✓")
    print("=" * 70)