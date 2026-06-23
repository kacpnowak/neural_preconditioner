import numpy as np
import matplotlib.pyplot as plt

scales = [10.0, 16.7, 27.8, 46.4, 77.4, 129.2, 215.4, 359.4, 599.5, 1000.0]
iters_no_precon = [869, 3000, 3000, 3000, 3000, 3000, 3000, 3000, 3000, 3000]
iters_jacobi = [1103, 3000, 3000, 3000, 3000, 3000, 3000, 3000, 3000, 3000]
iters_gnp = [307, 511, 1386, 3000, 3000, 3000, 3000, 3000, 3000, 3000]

plt.figure(figsize=(10, 6))
plt.plot(scales, iters_no_precon, "o--", label="CG (No Preconditioner)", color="red")
plt.plot(scales, iters_jacobi, "s--", label="CG + Jacobi Preconditioner", color="blue")
plt.plot(scales, iters_gnp, "d-", label="CG + GNP (Neural Preconditioner)", color="green")

plt.xscale("log")
plt.yscale("log")
plt.xlabel("Filter Scale L [km]")
plt.ylabel("JAX CG Iterations")
plt.title("JAX FGMRES Convergence vs. Scale L")
plt.grid(True, which="both", ls="-", alpha=0.5)
plt.legend()

plot_path = "./cg_convergence_scaling_jax.png"
plt.savefig(plot_path, dpi=300)
print("Plot generated successfully!")
