import jax
import jax.numpy as jnp
import jax.random as random
import numpy as np
import math
from scipy.sparse import csc_matrix, identity
import pyamg
from jax.experimental import sparse
from functools import partial
import time

jax.config.update("jax_enable_x64", True)

Lx = 1000
dxm = 10
n2d = np.arange(0, Lx + 1, dxm, dtype="float32").shape[0]**2
scales = np.logspace(1, 3, 10)
kc = 2 * math.pi / scales

from synthetic_data_generator import create_synthetic_matrix
ss, ii, jj, tri, xcoord, ycoord = create_synthetic_matrix(Lx, dxm, False)

Smat1 = csc_matrix((ss * (1.0 / np.square(kc[0])), (ii, jj)), shape=(n2d, n2d))
Smat_eval = identity(n2d) + 2.0 * (Smat1 ** 2)
ml = pyamg.smoothed_aggregation_solver(Smat_eval)

coo = ml.levels[0].A.tocoo()
A_bcoo = sparse.BCOO((jnp.array(coo.data, dtype=jnp.float64), 
                      jnp.column_stack((coo.row, coo.col))), 
                      shape=coo.shape)

@partial(jax.jit, static_argnames=['m'])
def arnoldi_build_jit(A, m, key):
    n = A.shape[0]
    V = jnp.zeros((n, m + 1), dtype=jnp.float64)
    H = jnp.zeros((m + 1, m), dtype=jnp.float64)

    v0 = random.normal(key, (n,), dtype=jnp.float64)
    v0 = v0 / jnp.linalg.norm(v0)
    V = V.at[:, 0].set(v0)

    def outer_loop(j, val):
        V, H = val
        v = A @ V[:, j]
        
        def inner_loop(k, val_inner):
            w, H_inner = val_inner
            h_k_j = jnp.dot(V[:, k], w)
            H_inner = H_inner.at[k, j].set(h_k_j)
            w = w - h_k_j * V[:, k]
            return w, H_inner

        w, H = jax.lax.fori_loop(0, j + 1, inner_loop, (v, H))
        
        h_j_plus_1_j = jnp.linalg.norm(w)
        H = H.at[j + 1, j].set(h_j_plus_1_j)
        V = V.at[:, j + 1].set(w / h_j_plus_1_j)
        return V, H

    V, H = jax.lax.fori_loop(0, m, outer_loop, (V, H))
    return V, H

print("Compiling arnoldi_build_jit...")
t0 = time.time()
V, H = arnoldi_build_jit(A_bcoo, m=80, key=random.PRNGKey(0))
V = jax.block_until_ready(V)
print(f"Done in {time.time()-t0:.2f} seconds!")
