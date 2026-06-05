import jax
import jax.numpy as jnp
from jax.experimental import sparse
from GraphUNet_JAX import GraphUNet_JAX

embed = 128
dtype = jnp.float64
net = GraphUNet_JAX(embed=embed, K=3, dtype=dtype)
dummy_input = jnp.ones((1, 1), dtype=dtype)
dummy_A = sparse.BCOO((jnp.ones(1, dtype=dtype), jnp.zeros((1, 2), dtype=jnp.int32)), shape=(1, 1))
dummy_hier = {'A': [dummy_A]*6, 'P': [dummy_A]*6, 'R': [dummy_A]*6}

import time
t0 = time.time()
print("Starting init...")
key = jax.random.PRNGKey(42)
params = net.init({'params': key, 'dropout': key}, dummy_input, dummy_hier, train=False)['params']
print(f"Init finished in {time.time() - t0:.2f} seconds")
