"""
Davelaar (2018) Simulation Study 1 - Library Modules.

This package contains the core components for the neurofeedback simulation:
- EEG generator (Izhikevich spiking neural network)
- Striatal learning system (MSN units with reward-modulated plasticity)
- Spectral analyzer (FFT-based analysis for neurofeedback)
- Neurofeedback model (integration and training protocol)
"""

from .IzhikevichEEGGenerator import IzhikevichEEGGenerator
from .ActivationBuffer import ActivationBuffer
from .StriatalLearning import StriatalLearning
from .SpectralAnalyzer import SpectralAnalyzer
from .NeurofeedbackSimulation import NeurofeedbackSimulation

__all__ = [
    'IzhikevichEEGGenerator',
    'StriatalLearning',
    'ActivationBuffer',
    'SpectralAnalyzer',
    'NeurofeedbackSimulation',
]

