import jax
import jax.numpy as jnp
from jax.experimental import sparse

A = sparse.BCOO.fromdense(jnp.array([[1.0, 0.0], [0.0, 1.0]]))
B = sparse.BCOO.fromdense(jnp.array([[2.0, 0.0], [0.0, 2.0]]))

@jax.jit
def mul(x, y):
    return x @ y

try:
    C = mul(A, B)
    print("JIT sparse-sparse worked!")
except Exception as e:
    print(f"JIT failed: {e}")
