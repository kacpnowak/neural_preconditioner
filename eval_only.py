import os
import jax
import jax.numpy as jnp
import jax.random as random
from flax.training import train_state
from flax.training import checkpoints
import optax
import numpy as np
import math
from scipy.sparse import csc_matrix, identity
import matplotlib.pyplot as plt

from synthetic_data_generator import create_synthetic_matrix, create_synthetic_data
from GNP_JAX import GNP_JAX_Model
from GraphUNet_JAX import GraphUNet_JAX
from jax_fgmres import jax_fgmres_solve
from ResGCN_JAX import scale_A_by_spectral_radius_jax

def run_eval():
    Lx = 1000
    dxm = 10
    n2d = np.arange(0, Lx + 1, dxm, dtype="float32").shape[0]**2
    
    ss, ii, jj, tri, xcoord, ycoord = create_synthetic_matrix(Lx, dxm, False)
    tt = create_synthetic_data(Lx, dxm)
    b = jnp.array(tt, dtype=jnp.float32)
    b = b / jnp.linalg.norm(b)

    scales = np.logspace(1, 3, 10)
    kc = 2 * math.pi / scales
    
    As_jax_eval = []
    diags_As_eval = []
    
    for k in kc:
        Smat1 = csc_matrix((ss * (1.0 / np.square(k)), (ii, jj)), shape=(n2d, n2d))
        Smat_eval = identity(n2d) + 2.0 * (Smat1 ** 2)
        A_jax_eval, gamma_eval = scale_A_by_spectral_radius_jax(Smat_eval)
        As_jax_eval.append(A_jax_eval)
        diag_Ai_eval = jnp.array(Smat_eval.diagonal() / gamma_eval, dtype=jnp.float32)
        diags_As_eval.append(diag_Ai_eval)

    net = GraphUNet_JAX(embed=128, K=3, dtype=jnp.float32)
    key = random.PRNGKey(42)
    key, init_key, dropout_key = random.split(key, 3)
    dummy_input = jnp.ones((n2d, 1), dtype=jnp.float32)
    
    # Init params
    params = net.init({'params': init_key, 'dropout': dropout_key}, dummy_input, As_jax_eval[-1], train=False)['params']
    optimizer = optax.adamw(learning_rate=0.0036, weight_decay=0.0024)
    state = train_state.TrainState.create(apply_fn=net.apply, params=params, tx=optimizer)
    
    # Load Phase 3 checkpoint
    ckpt_dir = './checkpoints_cg_jax/phase_3'
    if os.path.exists(ckpt_dir):
        state = checkpoints.restore_checkpoint(ckpt_dir=ckpt_dir, target=state)
        print("Successfully loaded Phase 3 checkpoint!")
    else:
        print("Checkpoint not found!")
        return

    gnp = GNP_JAX_Model(A=As_jax_eval[-1], net_apply=net.apply, net_params=state.params, training_data='x_mix', m=80)
    
    def python_fgmres_solve(A_op, b, M_op=None, max_iters=5000, tol=1e-6):
        return jax_fgmres_solve(A_op, b, M_op=M_op, restart=1000, max_iters=max_iters, tol=tol)

    print("\nEvaluating FGMRES convergence...")
    
    for i, Ai in enumerate(As_jax_eval):
        @jax.jit
        def A_op(x): return Ai @ x
        
        _, it_none = python_fgmres_solve(A_op, b, M_op=None, max_iters=5000, tol=1e-6)
        
        diag_inv = 1.0 / diags_As_eval[i]
        @jax.jit
        def M_jacobi(r): return diag_inv * r
        _, it_jac = python_fgmres_solve(A_op, b, M_op=M_jacobi, max_iters=5000, tol=1e-6)
        
        gnp.A = Ai
        precond_apply_jit = jax.jit(gnp.get_preconditioner_apply())
        
        @jax.jit
        def M_gnp(r): 
            z = jnp.zeros_like(r)
            r_curr = r
            for _ in range(5):
                update = precond_apply_jit(state.params, r_curr)
                z = z + update
                r_curr = r_curr - (Ai @ update)
            return z
        
        _, it_gnp = python_fgmres_solve(A_op, b, M_op=M_gnp, max_iters=5000, tol=1e-6)
        
        print(f"Scale L={scales[i]:.1f}km | No Precon: {it_none} iters | Jacobi: {it_jac} iters | GNP: {it_gnp} iters")

if __name__ == '__main__':
    run_eval()
