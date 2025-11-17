# Reproducibility Note

I attemped to include

```python
# Initialize RNG
self.local_rng = np.random.default_rng(seed)
self.local_rng_state = copy.deepcopy(self.local_rng.bit_generator.state)

if rng_state is not None:
    self.local_rng.bit_generator.state = copy.deepcopy(rng_state)
    self.local_rng_state = copy.deepcopy(rng_state)
    if seed is not None:
        warnings.warn("RNG state provided; seed will be ignored.")
```

so that I can save and restore the RNG state for reproducibility. However, I noticed that even after restoring the RNG state, the result is not exactly the same across runs.

Need further investigation to ensure that the RNG state restoration works as intended.
