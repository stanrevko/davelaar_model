# Davelaar (2018) Simulation Study 1

This repository contains a complete Python implementation of Simulation Study 1 from:

**Davelaar, E.J. (2018). Mechanisms of Neurofeedback: A Computation-theoretic Approach. Neuroscience, 378, 175-188.**

The simulation demonstrates successful EEG neurofeedback learning through a spiking neural network model with reward-modulated plasticity.

## Author

**Stanislav Revko**  
PhD Student  
Lesya Ukrainka Volyn National University

This implementation is part of a PhD dissertation research project on computational modeling of neurofeedback processes based on interactions in the thalamus-striatum-hippocampus-frontal-occipital cortex system.

## Overview

The simulation implements a multi-phase neurofeedback training protocol:

1. **Baseline (5 minutes)**: Record EEG to establish peak alpha frequency (PAF) and baseline statistics
2. **Training (5 sessions × 5 minutes each)**: Neurofeedback learning with reward-modulated striatal plasticity across multiple training sessions
3. **Post-training (5 minutes)**: Measure training effects on EEG spectral properties

## Components

### 1. EEG Generator (`modules/IzhikevichEEGGenerator.py`)
- Izhikevich spiking neural network (pure Python/NumPy implementation)
- 800 excitatory + 200 inhibitory neurons
- All-to-all connectivity
- Low-pass filtered sum of excitatory potentials as EEG proxy
- Thalamic input noise (Gaussian, σ=1.0)
- Optional measurement noise (white Gaussian, disabled by default)
- Warmup period to reduce initialization transients
- Class: `IzhikevichEEGGenerator`

### 2. Striatal Learning (`modules/StriatalLearning.py`)
- 1000 binary MSN (Medium Spiny Neuron) units
- Reward-modulated plasticity
- Stochastic activation sampling
- Class: `StriatalLearning`

### 2a. Activation Buffer (`modules/ActivationBuffer.py`)
- Circular buffer for tracking MSN activations
- Enables credit assignment over feedback window
- Class: `ActivationBuffer`

### 3. Spectral Analyzer (`modules/SpectralAnalyzer.py`)
- Real-time FFT-based spectral analysis
- Peak alpha frequency (PAF) detection (8-12 Hz)
- Upper alpha frequency (UAF) band power computation
- Feedback thresholding
- Class: `SpectralAnalyzer`

### 4. Neurofeedback Model (`modules/NeurofeedbackSimulation.py`)
- Integrates all components
- Implements complete training protocol
- Tracks learning progress and results
- Class: `NeurofeedbackSimulation`

### 5. Main Script (`run_simulation.py`)
- Command-line interface
- Visualization generation (Figures 4 & 5)
- Results saving (plots, data files)

## Quick Start

Get up and running in minutes:

```bash
# 1. Clone the repository
git clone git@github.com-stanrevko:stanrevko/davelaar_model.git
cd davelaar_model

# 2. Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run simulation (default: 5 training sessions × 5 minutes each)
python run_simulation.py

# 5. View results
# Results are saved in results/ directory:
#   - simulation_results.png: Comprehensive visualization
#   - results.npz: All simulation data
#   - learning_curve.csv: Learning progress over time
```

The simulation will run with default parameters:
- Baseline: 5 minutes
- Training: 5 sessions × 5 minutes each (25 minutes total)
- Post-training: 5 minutes
- Total duration: ~35 minutes

## Installation

### Requirements

- Python 3.7+
- NumPy >= 1.19.0
- SciPy >= 1.5.0
- Matplotlib >= 3.3.0
- tqdm >= 4.50.0

**Note**: This implementation uses pure Python/NumPy (no Brian2 dependency), making it faster and easier to modify.

### Install Dependencies

```bash
pip install -r requirements.txt
```

Or install manually:

```bash
pip install numpy scipy matplotlib tqdm
```

### Virtual Environment (Recommended)

```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Usage

### Basic Usage

Run the simulation with default parameters:

```bash
python run_simulation.py
```

### Command-Line Options

```bash
python run_simulation.py --help
```

Options:
- `--seed SEED`: Random seed for reproducibility (default: 42)
- `--output DIR`: Output directory for results (default: results/)
- `--baseline-duration SEC`: Baseline duration in seconds (default: 300 = 5 min)
- `--training-duration SEC`: Training duration per session in seconds (default: 300 = 5 min)
- `--n-training-phases N`: Number of training sessions (default: 5)
- `--post-duration SEC`: Post-training duration in seconds (default: 300 = 5 min)

### Example

```bash
# Run with default settings (results saved to results/ folder)
python run_simulation.py

# Run with custom seed
python run_simulation.py --seed 123

# Run with custom output directory
python run_simulation.py --output my_results/

# Run with shorter durations for testing
python run_simulation.py --baseline-duration 60 --training-duration 60 --n-training-phases 2 --post-duration 60

# Run with custom number of training sessions
python run_simulation.py --n-training-phases 10 --training-duration 300
```

## Output Files

After running the simulation, all results are saved in the `results/` folder (created automatically). The following files are generated:

1. **`results/simulation_results.png`**: Multi-panel figure showing:
   - EEG time series (before/after)
   - Power spectral density
   - UAF distributions (histograms)
   - Learning curves
   - Summary statistics

2. **`results/results.npz`**: NumPy archive containing all simulation data:
   - EEG signals (baseline, training, post)
   - UAF distributions
   - Learning curves
   - Target probability history
   - Feedback history
   - All statistics

3. **`results/learning_curve.csv`**: CSV file with target probability over time

## Testing and Benchmarking

### Test EEG Generator

Test the EEG generator independently:

```bash
python test_eeg_generator.py --duration 5000
```

This generates:
- `results/eeg_generator_test.png`: Visualization of the EEG signal
- Console output with signal statistics

### Benchmark Performance

Compare performance of different implementations:

```bash
# Benchmark pure Python/NumPy implementation
python benchmark_eeg_generator.py --duration 10000

# Compare with Brian2 (if installed)
python benchmark_eeg_generator.py --duration 10000 --compare-brian2
```

The pure Python/NumPy implementation typically achieves **~16-17x real-time speed** (simulates 16-17 seconds per real second).

## Examples

Example scripts demonstrating various features are available in the `examples/` folder:

```bash
# Run example showing custom input currents
python examples/example_custom_input.py
```

See `examples/README.md` for more details on available examples.

## Expected Results

### Success Criteria

Learning is considered successful if:
1. ✅ Final target probability > 0.1 (100× initial)
2. ✅ Post-training UAF mean > baseline UAF mean
3. ✅ Post-training UAF variance > baseline variance
4. ✅ Feedback rate during training: 40-70%

### Typical Output

```
============================================================
NEUROFEEDBACK SIMULATION (Davelaar 2018)
============================================================

Initializing simulation...
  MSN units: 1000
  Cortical neurons: 800E + 200I
  Target MSN: 447 (hidden)

Phase 1: Baseline Recording
  Duration: 5.0 minutes
Baseline: 100%|████████████████████| 300000/300000

Baseline established:
  PAF: 10.34 Hz
  UAF band: [10.34, 12.34] Hz
  Baseline UAF power: 45.67 ± 8.23

Phase 2: Training (5 sessions × 5 minutes)
  Session 1/5: 100%|████████████████████| 300000/300000
  Session 2/5: 100%|████████████████████| 300000/300000
  Session 3/5: 100%|████████████████████| 300000/300000
  Session 4/5: 100%|████████████████████| 300000/300000
  Session 5/5: 100%|████████████████████| 300000/300000

Training results:
  Initial P(target): 0.0010
  Final P(target): 0.3456
  Increase: 345.6×
  Feedback rate: 42.5%
  Weight updates: 3000 (600 per session)

Phase 3: Post-Training
  Duration: 5.0 minutes
Post-training: 100%|████████████████████| 300000/300000

============================================================
RESULTS
============================================================
UAF Power:
  Before: 45.67 ± 8.23
  After:  52.34 ± 12.45
  Change: +6.67 (+14.6%)

Learning Success: YES
  Target probability increased 345×
============================================================

Plots saved to: results/simulation_results.png
Results saved to: results/results.npz
Learning curve saved to: results/learning_curve.csv

Simulation complete!
```

## Code Structure

```
davelaar_simulation_study1/
├── modules/                   # Library modules
│   ├── __init__.py            # Package initialization
│   ├── IzhikevichEEGGenerator.py  # IzhikevichEEGGenerator class
│   ├── ActivationBuffer.py    # ActivationBuffer class
│   ├── StriatalLearning.py    # StriatalLearning class
│   ├── SpectralAnalyzer.py    # SpectralAnalyzer class
│   └── NeurofeedbackSimulation.py  # NeurofeedbackSimulation class
├── examples/                  # Example scripts
│   ├── README.md              # Examples documentation
│   └── example_custom_input.py # Custom input currents example
├── run_simulation.py          # Main executable script
├── test_eeg_generator.py      # Test script for EEG generator
├── benchmark_eeg_generator.py # Performance benchmarking script
├── requirements.txt           # Python dependencies
├── README.md                  # This file
├── venv/                      # Virtual environment (created locally)
└── results/                   # Output directory (created automatically)
```

## Key Parameters

### Network Parameters
- Excitatory neurons: 800 (Regular Spiking)
- Inhibitory neurons: 200 (Fast Spiking)
- Synaptic weights: Random, fixed throughout simulation
- Thalamic input: Gaussian noise (μ=5.0 for E, μ=2.0 for I, σ=1.0)
- Measurement noise: Optional white Gaussian noise on EEG signal (default: disabled, σ=0.0)
- Warmup period: 1000 ms (reduces initialization transients)
- Time step: 1 ms (Euler integration)

### Learning Parameters
- MSN units: 1000
- Learning rate: 0.1
- Expected active units: ~10 per time step
- Feedback window: 1024 ms
- Update interval: 100 ms

### Spectral Analysis
- Sampling rate: 1000 Hz
- Window size: 1024 samples (1024 ms)
- Alpha band: 8-12 Hz
- UAF band: [PAF, PAF+2] Hz

## Troubleshooting

### Memory Issues
If you encounter memory problems with long simulations:
- Reduce simulation durations for testing
- The code uses efficient buffering, but very long runs may require optimization

### Performance

The pure Python/NumPy implementation is optimized for speed:
- **~16-17x real-time**: Simulates 16-17 seconds per real second
- Efficient NumPy operations for neuron updates
- No external simulation framework overhead

For faster testing, use shorter durations:
```bash
python run_simulation.py --baseline-duration 60 --training-duration 60 --n-training-phases 2 --post-duration 60
```

The simulation can still be slow for full-length runs due to:
- Large network size (1000 neurons)
- High temporal resolution (1 ms steps)
- Long baseline/post phases (5 minutes each)

### No Learning Observed
If target probability doesn't increase:
- Check feedback rate (should be 40-70%)
- Verify baseline threshold is reasonable
- Try different random seeds
- Check console output for warnings

## References

1. **Davelaar, E.J. (2018)**. Mechanisms of Neurofeedback: A Computation-theoretic Approach. *Neuroscience*, 378, 175-188. DOI: 10.1016/j.neuroscience.2017.05.052

2. **Izhikevich, E.M. (2003)**. Simple model of spiking neurons. *IEEE Transactions on Neural Networks*, 14(6), 1569-1572.

3. **Zoefel, B., et al. (2011)**. Neurofeedback training of the upper alpha frequency band in EEG improves cognitive performance. *Neuroimage*, 54(2), 1427-1431.

## Implementation Notes

- **Pure Python/NumPy**: This implementation uses pure Python/NumPy instead of Brian2, making it faster (~16x real-time) and easier to modify
- **Euler Integration**: Izhikevich neuron dynamics are integrated using Euler method with 1 ms time step
- **Warmup Period**: Network runs for 1000 ms before recording to reduce initialization transients
- **Noise Sources**: 
  - Thalamic input noise: Gaussian noise (σ=1.0) drives network activity (as specified in Davelaar 2018)
  - Measurement noise: Optional white Gaussian noise can be added to model sensor artifacts (disabled by default, not in original paper)
- **Modular Design**: All components are separated into modules in the `modules/` folder for easy modification

## License

This code is provided for research and educational purposes. Please cite the original paper if you use this implementation in your work.

## Contact

For questions or issues, please open an issue on GitHub or contact:

**Stanislav Revko**  
Lesya Ukrainka Volyn National University  
GitHub: [@stanrevko](https://github.com/stanrevko)

