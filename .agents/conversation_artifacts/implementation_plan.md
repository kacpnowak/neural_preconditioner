# Training the Neural Preconditioner on FESOM CORE2 Data

The current Neural Preconditioner checkpoint was trained on synthetic regular grids and deployed zero-shot onto the FESOM mesh. To achieve massive reductions in FGMRES solver iterations, we must explicitly fine-tune the GraphUNet weights on the physical FESOM grid topology and real ocean temperature distributions.

## Proposed Changes

We will create a specialized training pipeline `train_fesom_preconditioner.py`.

### [NEW] [train_fesom_preconditioner.py](file:///Users/kanowa001/soft/neural_preconditioner/train_fesom_preconditioner.py)
This script will implement the JAX-based training loop specifically targeted for the FESOM mesh:
1. **Matrix Construction:** Use the highly optimized pure-NumPy `create_fesom_matrix_fast` to instantly assemble the unstructured Bi-Laplacian matrix for the 126k node FESOM CORE2 mesh.
2. **Hierarchy Generation:** Build the multi-level topological hierarchy (`A`, `P`, `R`) using PyAMG's Smoothed Aggregation.
3. **Data-Driven Loss Distribution:** Extract multiple time steps (snapshots) from `LCOR2_SST.nc`. These physical SST states will be used as the right-hand side (`b`) target vectors during the curriculum training phase. Training the neural network to approximate $M(r) \approx A^{-1} r$ directly on physical residual distributions heavily biases the solver toward rapid convergence for these specific physical problems.
4. **JAX Checkpointing:** Train the `GraphUNet_JAX` using Optax (`adamw`) and save the final Flax `TrainState` checkpoint specifically tuned for the CORE2 grid.

## Open Questions

> [!IMPORTANT]
> **Epochs and Runtime:** Training on a 126k-node grid takes considerably more memory and computation than the synthetic 10k-node patches. Do you want to run a quick, shallow training pass (e.g., 5-10 epochs) to verify it trains correctly, or should we set it up for a deep convergence run (e.g., 50+ epochs)?

## Verification Plan

### Automated Verification
- The training script will track and print the aggregated L1/L2 approximation loss.
- Once trained, we will modify `run_fesom_filter.py` to load this new FESOM-specific checkpoint and compare the FGMRES iterations against the baseline (200+ iterations).
