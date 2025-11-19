"""
Example: Using custom input currents I_exc for excitatory neurons.

This demonstrates how to provide custom input currents for all excitatory neurons
instead of using the default thalamic modulation.
"""

import numpy as np
import sys
from pathlib import Path

# Add parent directory to path to import modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from modules.IzhikevichEEGGenerator import IzhikevichEEGGenerator

def example_custom_input():
    """Example of using custom input currents."""
    
    # Create EEG generator
    eeg_gen = IzhikevichEEGGenerator(random_seed=42)
    
    # Warm up
    eeg_gen.warmup(duration_ms=1000)
    
    print("Example 1: Using default thalamic modulation")
    print("-" * 50)
    for i in range(5):
        eeg = eeg_gen.step(thalamic_modulation=0.0)
        print(f"Step {i+1}: EEG = {eeg:.2f}")
    
    print("\nExample 2: Using custom input currents I_exc")
    print("-" * 50)
    
    # Create custom input currents for all 800 excitatory neurons
    # Example: uniform input of 5.0 for all neurons
    I_exc_custom = np.full(800, 5.0)
    
    for i in range(5):
        eeg = eeg_gen.step(I_exc=I_exc_custom)
        print(f"Step {i+1}: EEG = {eeg:.2f}")
    
    print("\nExample 3: Using heterogeneous input currents")
    print("-" * 50)
    
    # Example: different input for each neuron (e.g., from striatal learning)
    # This could come from the Davelaar model based on MSN activations
    I_exc_heterogeneous = np.random.normal(5.0, 1.0, 800)
    
    for i in range(5):
        eeg = eeg_gen.step(I_exc=I_exc_heterogeneous)
        print(f"Step {i+1}: EEG = {eeg:.2f}")
    
    print("\nExample 4: Modulating specific neurons")
    print("-" * 50)
    
    # Example: increase input to first 100 neurons (e.g., target MSN active)
    I_exc_modulated = np.random.normal(5.0, 1.0, 800)
    I_exc_modulated[:100] += 1.0  # Add extra input to first 100 neurons
    
    for i in range(5):
        eeg = eeg_gen.step(I_exc=I_exc_modulated)
        print(f"Step {i+1}: EEG = {eeg:.2f}")


if __name__ == '__main__':
    example_custom_input()

