# Research Notes: Neural Preconditioners for the Bi-Laplacian

This document tracks the key mathematical discoveries and architectural shifts in our pursuit of a neural preconditioner for the Bi-Laplacian operator ($A = I + 2L^2$). These findings form the basis of a strong methodology section for a future paper.

## 1. The Compositional Trick and the "Phantom Cross-Term"
**Initial Approach:** 
We attempted to use a compositional approach where the network is trained to invert the discrete Laplacian ($A_{train} = I + \sqrt{2}L$), and then applied twice during inference to precondition the Bi-Laplacian: $M(M(r)) \approx A_{eval}^{-1}$.

**The Discovery:**
This approach fundamentally fails for highly stiff spatial scales (e.g., $L=27.8km$). The algebraic square of the training matrix is $A_{train}^2 = I + 2\sqrt{2}L + 2L^2$. The neural network perfectly learns to invert this matrix. However, the true Bi-Laplacian is $I + 2L^2$. The mismatch is a massive "phantom cross-term" ($2\sqrt{2}L$) which poisons the preconditioned residual. Unrolled training exacerbates this by *overfitting* to $A_{train}^2$, causing FGMRES to stall completely.

**Conclusion:**
For higher-order operators like the Bi-Laplacian, compositional tricks using lower-order operators introduce catastrophic spectral mismatches. Native training on the exact operator is strictly required.

## 2. Unrolled Iterative Solvers: The Space Mismatch
**Initial Approach:**
To give the network more depth to solve the $O(N^4)$ condition number of the Bi-Laplacian, we unrolled the network natively: $M(M(M(b))) \approx x$.

**The Discovery:**
The training loss spiked and flatlined because of a fundamental vector space mismatch. In a single pass, $M$ learns the mapping $b \to x$ (Residual Space $\to$ Solution Space). By composing $M(M(b))$, the inner $M$ outputs an approximate solution, which is then fed back into $M$. The network is forced to simultaneously act as the inverse operator ($b \to x$) and the identity matrix ($x \to x$), breaking the optimizer.

**Conclusion:**
Neural preconditioners cannot be naively composed if they map across spaces. Unrolling must mimic a stationary iterative solver (e.g., Richardson iteration) where $M$ strictly receives residuals and outputs solution updates.

## 3. The Richardson Unrolled Architecture
**Final Architecture:**
We unroll the network $k$ times using the following recurrent formulation:
1. $x_0 = 0, \quad r_0 = b$
2. For $i$ in $1 \dots k$:
   - $x_i = x_{i-1} + M(r_{i-1})$
   - $r_i = b - A x_i$

This guarantees mathematical consistency. The network learns a dynamical system (a contraction mapping) that iteratively refines the solution. During inference inside FGMRES, the "rolling factor" $k$ becomes a free hyperparameter that can be increased beyond the training horizon to achieve deeper preconditioning without retraining.

## 4. Key Literature & References
The methodology developed aligns with the cutting edge of learned iterative solvers and algorithm unrolling. Key recent works to cite include:

- **Self-Composing Neural Operators with Depth and Accuracy Scaling via Adaptive Train-and-Unroll Approach** (arXiv:2508.20650). 
  *Relevance:* Supports our methodology of recursively applying a learned base operator with a gradually increasing unrolled sequence to handle highly heterogeneous PDEs.
- **Born-Series-Inspired Residual Metric for Learning-based Preconditioners** (arXiv:2603.18527).
  *Relevance:* Supports the concept of pairing an adaptive "train-and-unroll" structure with a residual-driven neural operator.
- **PCG-Informed Neural Solvers for High-Resolution Homogenization of Periodic Microstructures** (arXiv:2506.17087).
  *Relevance:* Supports integrating iterative solvers (like PCG/FGMRES) directly into an unrolled network framework.
- **Neural Incomplete Factorization: Learning Preconditioners for the Conjugate Gradient Method** (arXiv:2305.16368).
  *Relevance:* Early foundational work on using Graph Neural Networks (GNNs) as preconditioners to accelerate iterative solvers.
- **DIPA: Distilled Preconditioned Algorithms for Solving Imaging Inverse Problems** (arXiv:2605.15456).
  *Relevance:* Covers knowledge distillation frameworks for preconditioning operators in unrolled solvers.
- **Bi-Laplacian / Biharmonic Specific Context:** See recent domain decomposition work by N. Dimola, P.F. Antonietti, and P. Zunino (e.g., arXiv:2505.08491) for foundational research on convolutional neural networks as preconditioners for the two-dimensional biharmonic (Bi-Laplacian) equation using Virtual Element Method discretizations.
