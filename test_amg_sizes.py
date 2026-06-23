import numpy as np
import math
from scipy.sparse import csc_matrix, identity
import pyamg

Lx = 1000
dxm = 10
n2d = np.arange(0, Lx + 1, dxm, dtype="float32").shape[0]**2
scales = np.logspace(1, 3, 10)
kc = 2 * math.pi / scales

from synthetic_data_generator import create_synthetic_matrix
ss, ii, jj, tri, xcoord, ycoord = create_synthetic_matrix(Lx, dxm, False)

for i, k in enumerate(kc):
    Smat1 = csc_matrix((ss * (1.0 / np.square(k)), (ii, jj)), shape=(n2d, n2d))
    Smat_eval = identity(n2d) + 2.0 * (Smat1 ** 2)
    ml = pyamg.smoothed_aggregation_solver(Smat_eval)
    levels = len(ml.levels)
    print(f"Matrix {i} (Scale L={scales[i]:.1f}km): {levels} levels")
