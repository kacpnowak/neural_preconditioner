import jax
import jax.numpy as jnp
from jax import config
# Enable 64-bit precision for stable FGMRES
config.update("jax_enable_x64", True)

import numpy as np
import math
import xarray as xr
import torch
import pyamg
from scipy.sparse import csc_matrix, identity

from synthetic_data_generator import neighboring_triangles, neighbouring_nodes, areas
from GNP_JAX import GNP_JAX_Model
from GraphUNet_JAX import GraphUNet_JAX
from ResGCN_JAX import scale_A_by_spectral_radius_jax
from jax_fgmres import jax_fgmres_solve
from scipy.sparse.linalg import eigs

def make_smooth_numpy(Mt, elem_area, dx, dy, nn_num, nn_pos, tri, n2d, e2d, full=False):
    smooth_m = np.zeros(nn_pos.shape, dtype=np.float32)
    metric = np.zeros(nn_pos.shape, dtype=np.float32)
    aux = np.zeros(n2d, dtype=np.int32)
    
    for j in range(e2d):
        enodes = tri[j, :]
        for n in range(3):
            row = enodes[n]
            cc = nn_num[row]
            # fill aux
            for i in range(cc):
                aux[nn_pos[i, row]] = i
            
            for m in range(3):
                col = enodes[m]
                pos = aux[col]
                tmp_x = dx[j, m] * dx[j, n]
                tmp_y = dy[j, n] * dy[j, m]
                
                if m == n and full:
                    smooth_m[pos, row] += (tmp_x + tmp_y) * elem_area[j] + (Mt[j]**2) * elem_area[j] / 3.0
                else:
                    smooth_m[pos, row] += (tmp_x + tmp_y) * elem_area[j]
                    
                metric[pos, row] += Mt[j] * (dx[j, n] - dx[j, m]) * elem_area[j] / 3.0
    return smooth_m, metric

def make_smat_numpy(nn_pos, nn_num, smooth_m, n2d, nza):
    ss = np.zeros(nza, dtype=np.float32)
    ii = np.zeros(nza, dtype=np.int32)
    jj = np.zeros(nza, dtype=np.int32)
    
    idx = 0
    for n in range(n2d):
        for m in range(nn_num[n]):
            ss[idx] = smooth_m[m, n]
            ii[idx] = n
            jj[idx] = nn_pos[m, n]
            idx += 1
            
    return ss, ii, jj

def create_fesom_matrix_fast(xcoord, ycoord, tri):
    tri = np.array(tri)
    n2d = len(xcoord)
    e2d = len(tri[:, 1])
    cyclic_length = 360 * math.pi / 180
    ne_num, ne_pos = neighboring_triangles(n2d, e2d, tri)
    nn_num, nn_pos = neighbouring_nodes(n2d, tri, ne_num, ne_pos)
    area, elem_area, dx, dy, Mt = areas(n2d, e2d, tri, xcoord, ycoord, ne_num, ne_pos, "r", False, cyclic_length)
    
    smooth_m, metric = make_smooth_numpy(Mt, elem_area, dx, dy, nn_num, nn_pos, tri, n2d, e2d, False)
    nza = int(np.sum(nn_num))
    ss, ii, jj = make_smat_numpy(nn_pos, nn_num, smooth_m, n2d, nza)
    return ss, ii, jj, tri, xcoord, ycoord

def main():
    print("Loading FESOM CORE2 Mesh...")
    mesh_ds = xr.open_dataset("/Users/kanowa001/soft/AWI/data/core2/fesom.mesh.diag.nc")
    xcoord = mesh_ds['nodes'].values[0, :]
    ycoord = mesh_ds['nodes'].values[1, :]
    tri = mesh_ds['elem'].T.values - 1
    n2d = xcoord.shape[0]
    print(f"Mesh loaded. Nodes: {n2d}, Elements: {tri.shape[0]}")

    print("Building unstructured Bi-Laplacian matrix using Numpy (Fast)...")
    ss, ii, jj, tri, xcoord, ycoord = create_fesom_matrix_fast(xcoord, ycoord, tri)
    
    # Scale parameter for the Laplacian
    k = 0.05
    Smat1 = csc_matrix((ss * (1.0 / np.square(k)), (ii, jj)), shape=(n2d, n2d))
    A_scipy = identity(n2d) + 2.0 * (Smat1 ** 2)
    
    print("Loading LCORE2 FESOM SST Data...")
    dataset = xr.open_dataset("/Users/kanowa001/soft/AWI/data/LCORE2/temp.fesom.1948.nc")
    temp_var = dataset['temp'].values
    
    if len(temp_var.shape) == 3:
        sst = temp_var[0, :, 0]
    elif len(temp_var.shape) == 2:
        sst = temp_var[0, :]
    else:
        sst = temp_var
        
    # Make sure we only take the nodes if it's somehow larger
    sst = sst[:n2d]
    b = jnp.array(sst, dtype=jnp.float64)
    x0 = jnp.zeros_like(b)
    
    # Check for NaN in data
    valid_mask = ~jnp.isnan(b)
    b = jnp.where(valid_mask, b, 0.0) # Fill NaNs with 0 for the filter
    
    # Setup JAX matrices
    # We will scale A to have spectral radius ~1 to ensure Unet stability
    scaled_A_BCOO, spectral_radius = scale_A_by_spectral_radius_jax(A_scipy)
    scaled_b = b / spectral_radius
    from jax.experimental import sparse
    A_BCOO = sparse.BCOO.from_scipy_sparse(A_scipy)
    
    # Create PyAMG hierarchy
    print("Building PyAMG Smoothed Aggregation hierarchy...")
    ml = pyamg.smoothed_aggregation_solver(A_scipy, max_levels=3, keep=True)
    amg_hierarchy = {'A': [], 'P': [], 'R': []}
    amg_hierarchy['A'].append(scaled_A_BCOO) # Must use the scaled matrix!
    for i, lvl in enumerate(ml.levels):
        if i > 0:
            coo = lvl.A.tocoo()
            amg_hierarchy['A'].append(sparse.BCOO((jnp.array(coo.data, dtype=jnp.float64), jnp.column_stack((coo.row, coo.col))), shape=coo.shape))
        if hasattr(lvl, 'P'):
            coo = lvl.P.tocoo()
            amg_hierarchy['P'].append(sparse.BCOO((jnp.array(coo.data, dtype=jnp.float64), jnp.column_stack((coo.row, coo.col))), shape=coo.shape))
        if hasattr(lvl, 'R'):
            coo = lvl.R.tocoo()
            amg_hierarchy['R'].append(sparse.BCOO((jnp.array(coo.data, dtype=jnp.float64), jnp.column_stack((coo.row, coo.col))), shape=coo.shape))

    print("Initializing GraphUNet_JAX...")
    net = GraphUNet_JAX(embed=128, K=3, dtype=jnp.float64)
    key = jax.random.PRNGKey(42)
    key, init_key = jax.random.split(key)
    
    num_l = len(amg_hierarchy['A'])
    dummy_A = sparse.BCOO((jnp.ones(1, dtype=jnp.float64), jnp.zeros((1, 2), dtype=jnp.int32)), shape=(1, 1))
    dummy_hier = {'A': [dummy_A]*num_l, 'P': [dummy_A]*num_l, 'R': [dummy_A]*num_l}
    dummy_input = jnp.ones((1, 1), dtype=jnp.float64)
    
    params = net.init(init_key, dummy_input, dummy_hier, train=False)['params']
    
    from flax.training import train_state, checkpoints
    import optax
    state = train_state.TrainState.create(apply_fn=net.apply, params=params, tx=optax.adam(1e-3))
    
    # We could evaluate un-trained random weights, but let's load a checkpoint trained on the synthetic 16.7km matrix
    # and see if it generalizes zero-shot to FESOM real data!
    print("Restoring pretrained checkpoint...")
    state = checkpoints.restore_checkpoint(ckpt_dir='./checkpoints_cg_jax/phase_1/gnp_model_1977', target=state)
    
    @jax.jit
    def net_apply_jit(params, x, hierarchy, **kwargs):
        return net.apply(params, x, hierarchy=hierarchy, train=False)
        
    gnp = GNP_JAX_Model(A=scaled_A_BCOO, net_apply=net_apply_jit, net_params=state.params, training_data='x_mix', m=80)
    gnp.hierarchy = amg_hierarchy
    precond_apply = jax.jit(gnp.get_preconditioner_apply())
    
    @jax.jit
    def M_gnp(r):
        z = jnp.zeros_like(r)
        r_curr = r
        for _ in range(4): # max_p for eval
            update = precond_apply(state.params, r_curr).astype(jnp.float64)
            z = z + update
            r_curr = r_curr - (scaled_A_BCOO @ update)
        return z
        
    @jax.jit
    def A_op(x): return scaled_A_BCOO @ x

    @jax.jit
    def A_op_baseline(x): return A_BCOO @ x

    print("Executing baseline FGMRES (No Preconditioner)...")
    baseline_result = jax_fgmres_solve(A_op_baseline, b, M_op=None, tol=1e-5, restart=50, max_iters=200, dtype=jnp.float64)
    baseline_iters = baseline_result[1]
    
    print(f"Baseline FGMRES Iterations: {baseline_iters}")
    
    print("Executing FGMRES with Pretrained Neural Preconditioner...")
    gnp_result = jax_fgmres_solve(A_op, scaled_b, M_op=M_gnp, tol=1e-5, restart=50, max_iters=200, dtype=jnp.float64)
    gnp_iters = gnp_result[1]
    
    print(f"GNP FGMRES Iterations: {gnp_iters}")

if __name__ == "__main__":
    main()
