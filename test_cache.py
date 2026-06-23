import jax
import jax.numpy as jnp
from jax.experimental import sparse

# Define a simple function that takes a hierarchy
@jax.jit
def train_step(dyn_hier):
    print("Compiling train_step!!!")
    return dyn_hier['A'][0] @ jnp.ones(dyn_hier['A'][0].shape[1])

# Create two identical hierarchies but with different object IDs
A0 = sparse.BCOO((jnp.ones(10, dtype=jnp.float64), jnp.zeros((10, 2), dtype=jnp.int32)), shape=(10, 10))
hier0 = {'A': [A0]}

A1 = sparse.BCOO((jnp.ones(10, dtype=jnp.float64), jnp.zeros((10, 2), dtype=jnp.int32)), shape=(10, 10))
hier1 = {'A': [A1]}

print("Running Matrix 0...")
res0 = train_step(hier0)
res0 = jax.block_until_ready(res0)

print("Running Matrix 1...")
res1 = train_step(hier1)
res1 = jax.block_until_ready(res1)
print("Done!")
