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
    
    print(f"Matrix {i} (Scale L={scales[i]:.1f}km):")
    for lvl_idx, lvl in enumerate(ml.levels):
        A_nnz = lvl.A.nnz
        print(f"  Level {lvl_idx}: A_nnz = {A_nnz}, shape = {lvl.A.shape}")
        if hasattr(lvl, 'P'):
            print(f"    P_nnz = {lvl.P.nnz}, shape = {lvl.P.shape}")
            print(f"    R_nnz = {lvl.R.nnz}, shape = {lvl.R.shape}")
