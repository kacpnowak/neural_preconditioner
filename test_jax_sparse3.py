import jax
import jax.numpy as jnp
from jax.experimental import sparse

# R shape (2, 4)
R = sparse.BCOO.fromdense(jnp.array([[1.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 1.0]]))
# x shape (4, 3, 2)
x = jnp.ones((4, 3, 2))

# R @ x
try:
    # R is (2, 4). x is (4, 3, 2). Dot product over axis 1 of R and axis 0 of x.
    out = sparse.bcoo_dot_general(R, x, dimension_numbers=(((1,), (0,)), ((), ())))
    print("out shape:", out.shape)
except Exception as e:
    print("Failed:", e)
