# Release preparation checks

Checked on 2026-10-25 with Python 3.10.19, PyTorch 2.9.1, and NumPy 2.2.6.

- Seven exported checkpoints strictly load all model parameters and buffers.
  Every tensor is bit-for-bit equal to its original research checkpoint.
- All 17,799 test windows (174,490 actor-windows) have exactly the same float32
  coordinates and ordering as the research evaluation inputs.
- For all seven models, CPU predictions were compared with the original frozen
  experimental code on 61 real windows and seven synthetic batches. Final and
  initial predictions, gates, coefficients, and refinement histories matched
  exactly (maximum absolute difference: 0).
- The checks included a 57-agent window, stationary/moving trajectories, and
  padded actors. Seven interface/metric tests passed.

These checks establish release equivalence; they are not a new full-dataset
evaluation. The README reports the original recorded full test scores. Device
and numerical-library versions may cause small floating-point differences.

To run the lightweight tests:

```bash
python -m pip install -e ".[test]"
python -m pytest -q
```
