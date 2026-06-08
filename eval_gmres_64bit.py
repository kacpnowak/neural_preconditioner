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
from jax.scipy.sparse.linalg import gmres as jax_gmres

from GNP_JAX import GNP_JAX_Model
from jax_fgmres import jax_fgmres_solve
from synthetic_data_generator import create_synthetic_matrix

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

    ml = pyamg.smoothed_aggregation_solver(Smat_eval)
    
    amg_hierarchy = {'A': [], 'P': [], 'R': []}
    for lvl in ml.levels:
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
    gnp = GNP_JAX_Model(A=A_eval_bcoo, hierarchy=amg_hierarchy, training_data='x_normal', m=80, dtype=jnp.float64)
    key = random.PRNGKey(0)
    key, subkey = random.split(key)
    dummy_input = jnp.zeros((n2d, 4), dtype=jnp.float64)
    params = gnp.net.init(subkey, dummy_input, amg_hierarchy, train=False)['params']
    
    state = train_state.TrainState.create(apply_fn=gnp.net.apply, params=params, tx=optax.adam(1e-3))
    state = checkpoints.restore_checkpoint(ckpt_dir='./checkpoints_cg_jax/phase_1/gnp_model_1977', target=state)
    
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
    
    print("\n--- Running JAX Native GMRES (Standard) in 64-bit ---")
    t0 = time.time()
    x_gmres, info = jax_gmres(A_op, b_eval, M=M_gnp, tol=1e-6, restart=200, maxiter=15) # maxiter is outer restarts (15*200 = 3000)
    x_gmres = jax.block_until_ready(x_gmres)
    t1 = time.time()
    res_gmres = jnp.linalg.norm(b_eval - A_op(x_gmres))
    print(f"JAX native GMRES finished in {t1-t0:.2f}s | Final Residual: {res_gmres:.4e} | Info (0=success): {info}")
    
    print("\n--- Running Custom FGMRES in 64-bit ---")
    # For custom fgmres, it defaults to whatever we set in the file, but we will pass float64 explicitly via modifying it or just checking.
    # Wait, jax_fgmres.py hardcodes float32. Let's just run it to see.
    # If the user wants 64-bit FGMRES, we can rewrite the file.
    
run_64bit_eval()
