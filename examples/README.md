# Examples

This folder contains example scripts demonstrating how to use the Davelaar (2018) simulation components.

## Available Examples

### `example_custom_input.py`

Demonstrates how to use custom input currents `I_exc` for all excitatory neurons in the EEG generator.

**Usage:**
```bash
cd examples
python example_custom_input.py
```

**What it shows:**
- Default thalamic modulation (backward compatible)
- Uniform custom input for all neurons
- Heterogeneous input (different for each neuron)
- Selective modulation of specific neurons

## Running Examples

All examples should be run from the project root directory:

```bash
# From project root
python examples/example_custom_input.py
```

Or from within the examples directory:

```bash
cd examples
python example_custom_input.py
```

## Adding New Examples

When adding new examples:
1. Place them in this `examples/` folder
2. Use imports: `from modules.ClassName import ClassName` (files use PascalCase)
3. Add a description in this README
4. Include usage instructions in the script docstring

