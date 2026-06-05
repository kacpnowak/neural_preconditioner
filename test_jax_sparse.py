import jax
import jax.numpy as jnp
from jax.experimental import sparse

# Create dummy sparse matrices
A = sparse.BCOO.fromdense(jnp.array([[1.0, 0.0], [0.0, 1.0]]))
B = sparse.BCOO.fromdense(jnp.array([[2.0, 0.0], [0.0, 2.0]]))
try:
    C = sparse.bcoo_dot_general(A, B, dimension_numbers=(((1,), (0,)), ((), ())))
    print("Sparse-sparse dot general worked!")
except Exception as e:
    print(f"Sparse-sparse failed: {e}")

try:
    C = A @ B
    print("Sparse-sparse matmul worked!")
except Exception as e:
    print(f"Sparse-sparse matmul failed: {e}")
