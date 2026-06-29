# Neural Preconditioner Project Rules & Context

Welcome, future AI agent! This repository contains a deep learning JAX-based Neural Preconditioner specifically designed to accelerate the iterative FGMRES convergence of unstructured oceanography models (specifically, the Bi-Laplacian filter for the FESOM CORE2 mesh).

## Repository Context
- This project ported Jie Chen's Neural Preconditioner architecture from PyTorch to JAX to natively target massive unstructured sparse meshes (126k+ nodes).
- The GraphUNet architecture relies heavily on JAX sparse operations (`jax.experimental.sparse.BCOO`) and utilizes PyAMG for Smoothed Aggregation hierarchy generation.
- The repository evaluates the neural preconditioner against the unmodified baseline FGMRES solver.

## Critical Design Decisions
1. **Compilation Trace Hangs**: `jax.lax.scan` causes catastrophic XLA compiler memory hangs when unrolling sparse loops over large embeddings (`embed=128`). We fixed this by manually pulling the `train_step` into a standard Python `for` loop and JIT-compiling *only* the single step. If you modify the training loop, DO NOT use `jax.lax.scan` for epoch-level iteration.
2. **Spectral Radius Scaling**: The FESOM matrix coefficients are massive. To prevent the UNet gradients from exploding, the input matrix is dynamically scaled by its spectral radius (`A_scaled = A / spectral_radius`), and the RHS vectors are similarly divided before training/evaluation. This must be maintained for numerical stability.

## Conversation History & Artifacts
To gain a full understanding of the scaling discrepancies, synthetic benchmarks, and training trajectories established in previous sessions, review the markdown artifacts located in `.agents/conversation_artifacts/`:
- `implementation_plan.md`: The original design schema.
- `research_notes.md`: Detailed mathematical teardown of the PyTorch codebase and how we matched it in JAX.
- `walkthrough.md`: Final evaluations, plot context, and step-by-step documentation of our process.
- `task.md`: Checklist of everything accomplished.

## Adding to Git
If you modify these architectures, make sure to add the checkpoint directories (`checkpoints_*/`) to `.gitignore` to prevent heavy weight files from overloading the repository.
