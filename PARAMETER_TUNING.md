# Parameter Tuning to Match Davelaar (2018) Results

## Current Issues vs. Expected Values

### 1. UAF Power Scaling Issue

**Expected (from paper):**
- Baseline: Mean ≈ 65, Std ≈ 18, Range: 15-151
- Post-training: Mean ≈ 92, Std ≈ 31, Range: 15-195

**Current simulation:**
- Baseline: Mean ≈ 648,683, Std ≈ 564,623
- Post-training: Mean ≈ 525,215, Std ≈ 382,295

**Problem:** UAF power values are ~10,000× too large.

**Root cause:** The EEG signal is the sum of 800 membrane potentials (each ~-65 to -50 mV), giving values around -40,000 to -50,000. The PSD scales with the square of signal amplitude, leading to very large power values.

**Solution:** Need to normalize or scale the EEG signal before computing PSD, or scale the PSD values after computation.

### 2. Learning Not Occurring

**Expected (from paper):**
- Initial P(target) ≈ 0.001 (1/1000)
- Final P(target) ≈ 0.4-0.9 (400-900× increase)

**Current simulation:**
- Initial P(target) ≈ 0.01
- Final P(target) ≈ 0.01 (no increase)

**Problem:** Target probability not increasing during training.

**Possible causes:**
1. Feedback rate might be too high (80% vs expected 40-70%)
2. Learning rate might need adjustment
3. Weight update mechanism might need refinement
4. Baseline threshold might be set incorrectly

### 3. Power Spectrum Scaling

**Expected (from paper):**
- Peak power: ~0.08-0.14 (before/after)
- Power range: 0-80 on plot

**Current simulation:**
- Peak power: Much higher values
- Need normalization

## Proposed Fixes

### Fix 1: Normalize EEG Signal Before Spectral Analysis

Scale the EEG signal to a reasonable range before computing PSD. Options:
- Option A: Normalize by number of neurons (divide by 800)
- Option B: Normalize to unit variance
- Option C: Scale PSD values after computation

### Fix 2: Adjust Learning Parameters

1. **Check feedback threshold**: Ensure baseline threshold is appropriate
2. **Adjust learning rate**: May need to increase from 0.1
3. **Verify weight updates**: Ensure updates are being applied correctly
4. **Check feedback rate**: Should be 40-70%, not 80%

### Fix 3: Scale Power Spectrum

Normalize PSD values to match paper's scale (0-80 range).

## Implementation Priority

1. **High Priority**: Fix UAF power scaling (affects all results)
2. **High Priority**: Fix learning mechanism (core functionality)
3. **Medium Priority**: Scale power spectrum (visualization)

