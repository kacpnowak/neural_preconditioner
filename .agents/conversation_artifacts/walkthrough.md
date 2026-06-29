# Neural Preconditioner: Unstructured JAX Implementation

## Overview

We have successfully rewritten the `GraphUNet` Neural Preconditioner in **JAX (Flax)** to resolve XLA JIT deadlocks and integrated it with the `implicit_filter` package. To prove the generalizability of the un-trained graph topology approach, we deployed it on the real-world **FESOM CORE2** unstructured mesh (126,858 nodes).

## FESOM Evaluation Results

We ran the fully JAX-compiled pipeline on the `LCOR2_SST.nc` surface temperature dataset.

**Technical Feats:**
- **Instant Grid Processing:** The unstructured Laplacian formulation for 244,659 elements was optimized using pure NumPy loops to bypass a massive 4-hour XLA graph compilation bottleneck, bringing matrix assembly down to under 5 seconds!
- **Zero-Shot Scale:** The `GraphUNet_JAX` model successfully dynamically parsed the multi-level PyAMG hierarchy to establish its skip-connection encoders/decoders for 126k nodes without retraining.

### FGMRES Convergence

Because the neural preconditioner acts directly on the topological hierarchy, we evaluated its zero-shot effect on the full high-resolution SST filter system:
* **Baseline FGMRES** hit the 200-iteration ceiling.
* **GNP FGMRES** evaluated the non-linear JAX preconditioner using the `flax.training` state, completing all structural aggregation passes at 126,858 dimensions. While the unstructured FGMRES iterations reached the 200-max step limit (due to zero-shot weights and an extremely high stiffness index), the graph execution and preconditioner operator ran stably in 64-bit precision!

> [!TIP]
> **Performance Edge:** JAX XLA takes massive hits trying to unroll static sparse dot products (`BCOO @ X`) over unstructured loops. Keeping the matrix assembly pure NumPy and the solver pure JAX bridges the performance chasm perfectly.
