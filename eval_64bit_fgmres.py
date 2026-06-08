import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import jax.random as random
import numpy as np
import math
from scipy.sparse import csc_matrix, identity
import pyamg
from jax.experimental import sparse
from flax.training import train_state, checkpoints
import optax
import time

from GraphUNet_JAX import GraphUNet_JAX
from GNP_JAX import GNP_JAX_Model
from jax_fgmres import jax_fgmres_solve
from synthetic_data_generator import create_synthetic_matrix
from ResGCN_JAX import scale_A_by_spectral_radius_jax

def run_64bit_eval():
    print("Loading Matrix Scale L=16.7km (Index 1)...")
    Lx = 1000
    dxm = 10
    n2d = np.arange(0, Lx + 1, dxm, dtype="float32").shape[0]**2
    scales = np.logspace(1, 3, 10)
    kc = 2 * math.pi / scales[1]

    ss, ii, jj, tri, xcoord, ycoord = create_synthetic_matrix(Lx, dxm, False)
    Smat1 = csc_matrix((ss * (1.0 / np.square(kc)), (ii, jj)), shape=(n2d, n2d))
    Smat_eval = identity(n2d) + 2.0 * (Smat1 ** 2)

    # Scale the matrix by its spectral radius just like we do during training!
    print("Scaling matrix by spectral radius for Neural Preconditioner stability...")
    A_jax_val, _ = scale_A_by_spectral_radius_jax(Smat_eval)

    # Note: We must also run the AMG on the spectral-scaled matrix, 
    # OR we can just use PyAMG on the unscaled and scale the JAX output.
    # In run_cg_preconditioner_jax.py, PyAMG was built on Smat_eval (unscaled).
    # Then we scaled Smat_eval into A_jax_val. 
    # Let's ensure the AMG hierarchy uses A_jax_val for the first level!
    ml = pyamg.smoothed_aggregation_solver(Smat_eval)
    
    amg_hierarchy = {'A': [], 'P': [], 'R': []}
    
    # We MUST replace the first level 'A' with our scaled A_jax_val
    amg_hierarchy['A'].append(A_jax_val)
    
    for i, lvl in enumerate(ml.levels):
        if i > 0:
            coo = lvl.A.tocoo()
            A_bcoo = sparse.BCOO((jnp.array(coo.data, dtype=jnp.float64), 
                                  jnp.column_stack((coo.row, coo.col))), shape=coo.shape)
            amg_hierarchy['A'].append(A_bcoo)
            
        if hasattr(lvl, 'P'):
            P_coo = lvl.P.tocoo()
            P_bcoo = sparse.BCOO((jnp.array(P_coo.data, dtype=jnp.float64), 
                                  jnp.column_stack((P_coo.row, P_coo.col))), shape=P_coo.shape)
            amg_hierarchy['P'].append(P_bcoo)
            
        if hasattr(lvl, 'R'):
            R_coo = lvl.R.tocoo()
            R_bcoo = sparse.BCOO((jnp.array(R_coo.data, dtype=jnp.float64), 
                                  jnp.column_stack((R_coo.row, R_coo.col))), shape=R_coo.shape)
            amg_hierarchy['R'].append(R_bcoo)
            
    A_eval_bcoo = amg_hierarchy['A'][0]
    
    print("Initializing Model and Loading 64-bit Checkpoint...")
    net = GraphUNet_JAX(embed=128, K=3, dtype=jnp.float64)
    key = random.PRNGKey(42)
    key, init_key = random.split(key)
    
    num_l = len(amg_hierarchy['A'])
    dummy_A = sparse.BCOO((jnp.ones(1, dtype=jnp.float64), jnp.zeros((1, 2), dtype=jnp.int32)), shape=(1, 1))
    dummy_hier = {'A': [dummy_A]*num_l, 'P': [dummy_A]*num_l, 'R': [dummy_A]*num_l}
    dummy_input = jnp.ones((1, 1), dtype=jnp.float64)
    
    params = net.init(init_key, dummy_input, dummy_hier, train=False)['params']
    
    state = train_state.TrainState.create(apply_fn=net.apply, params=params, tx=optax.adam(1e-3))
    
    # Reload from checkpoint
    state = checkpoints.restore_checkpoint(ckpt_dir='./checkpoints_cg_jax/phase_1/gnp_model_1977', target=state)
    
    print("Model initialized and checkpoint restored!")
    
    @jax.jit
    def net_apply_jit(params, x, hierarchy, **kwargs):
        return net.apply(params, x, hierarchy=hierarchy, train=False)
        
    gnp = GNP_JAX_Model(A=A_eval_bcoo, net_apply=net_apply_jit, net_params=state.params, training_data='x_mix', m=80)
    gnp.hierarchy = amg_hierarchy
    precond_apply = jax.jit(gnp.get_preconditioner_apply())
    
    @jax.jit
    def M_gnp(r):
        z = jnp.zeros_like(r)
        r_curr = r
        for _ in range(4): # max_p for eval
            update = precond_apply(state.params, r_curr).astype(jnp.float64)
            z = z + update
            r_curr = r_curr - (A_eval_bcoo @ update)
        return z
        
    @jax.jit
    def A_op(x): return A_eval_bcoo @ x
    
    key, subkey = random.split(key)
    tt = random.normal(subkey, (n2d,), dtype=jnp.float64)
    b_eval = tt / jnp.linalg.norm(tt)
    
    print("\n--- Running Custom FGMRES in 64-bit ---")
    t0 = time.time()
    x_fgmres, iters = jax_fgmres_solve(A_op, b_eval, M_op=M_gnp, tol=1e-6, restart=200, max_iters=3000, dtype=jnp.float64)
    x_fgmres = jax.block_until_ready(x_fgmres)
    t1 = time.time()
    res_fgmres = jnp.linalg.norm(b_eval - A_op(x_fgmres))
    print(f"64-bit FGMRES finished in {t1-t0:.2f}s | Iterations: {iters} | Final Residual: {res_fgmres:.4e}")

if __name__ == "__main__":
    run_64bit_eval()
