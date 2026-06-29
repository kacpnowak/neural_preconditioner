# Training FESOM Neural Preconditioner

- `[x]` Create `train_fesom_preconditioner.py`
  - `[x]` Implement FESOM data loading (batch of LCOR2 SST vectors)
  - `[x]` Re-use `make_smooth_numpy` matrix building logic
  - `[x]` Setup PyAMG hierarchy construction
  - `[x]` Hook into `GNP_JAX_Model.train()`
- `[x]` Modify `GNP_JAX.py` to format training data directly via custom inputs.
- `[x]` Execute the training script to generate the FESOM-optimized checkpoint.
- `[x]` Validate the trained checkpoint by running the modified `run_fesom_filter.py`.
- `[x]` Run full script evaluating Baseline vs Fully-trained Preconditioner.
- `[x]` Create relative residual convergence plot comparing the solvers.
- `[x]` Present final results and walkthrough to user.
