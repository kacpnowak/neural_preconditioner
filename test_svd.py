import jax.numpy as jnp
import numpy as np

print("Testing SVD on NaNs...")
A = jnp.array([[np.nan, 1.0], [1.0, np.nan]])
u, s, v = jnp.linalg.svd(A)
print("NaN SVD Done!")

print("Testing SVD on Infs...")
B = jnp.array([[np.inf, 1.0], [1.0, np.inf]])
u, s, v = jnp.linalg.svd(B)
print("Inf SVD Done!")
