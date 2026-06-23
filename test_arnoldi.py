import jax
import jax.numpy as jnp
import jax.random as random
import numpy as np
import math
from scipy.sparse import csc_matrix, identity
import pyamg
from jax.experimental import sparse
import time

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

print("Converting to BCOO...")
coo = ml.levels[0].A.tocoo()
A_bcoo = sparse.BCOO((jnp.array(coo.data, dtype=jnp.float64), 
                      jnp.column_stack((coo.row, coo.col))), 
                      shape=coo.shape)

from GNP_JAX import arnoldi_build

print("Running arnoldi_build...")
t0 = time.time()
V, H = arnoldi_build(A_bcoo, m=80, key=random.PRNGKey(0))
print(f"Done in {time.time()-t0:.2f} seconds!")
