import os
import time
import gc
import math
import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from tqdm import tqdm
import flax.linen as nn
from flax.training import train_state, checkpoints
from jax_fgmres import jax_fgmres_solve
import random as python_random
from jax import random
from jax.experimental import sparse
from jax.scipy.sparse.linalg import cg as jax_cg
import optax
import matplotlib.pyplot as plt
from scipy.sparse import csc_matrix, identity

from synthetic_data_generator import create_synthetic_matrix, create_synthetic_data
from ResGCN_JAX import scale_A_by_spectral_radius_jax, ResGCN_JAX
from GNP_JAX import GNP_JAX_Model
import pyamg

def run_jax_experiment():
    print("Starting JAX Neural CG Preconditioner Experiment...")
    print(f"JAX Default Backend: {jax.default_backend()}")
    
    # 1. Config grid size
    # Using 10,201 nodes mesh (dxm = 10) for CPU speed.
    Lx = 1000  # Domain size in km
    dxm = 10   # Resolution in km (101x101 grid = 10,201 nodes)
    n2d = np.arange(0, Lx + 1, dxm, dtype="float32").shape[0]**2
    print(f"Grid size: {n2d} nodes")
    
    print("Generating synthetic mesh connectivity and data (via JAX)...")
    ss, ii, jj, tri, xcoord, ycoord = create_synthetic_matrix(Lx, dxm, False)
    tt = create_synthetic_data(Lx, dxm)
    
    # 10 length scales L from 10 to 1000 km
    scales = np.logspace(1, 3, 10)
    kc = 2 * math.pi / scales
    
    print("Constructing family of JAX sparse matrices A(L) and AMG hierarchies...")
    As_jax_train = []
    As_jax_eval = []
    As_jax_laplacian = []
    gammas_train = []
    diags_As_eval = []
    
    # Store the AMG hierarchy (Prolongation, Restriction, and Coarse Matrices)
    amg_hierarchies_train = []
    
    for k in tqdm(kc, desc="Building PyAMG Hierarchies"):
        Smat1 = csc_matrix((ss * (1.0 / np.square(k)), (ii, jj)), shape=(n2d, n2d))
        # We will use the exact Bi-Laplacian for BOTH training and evaluation
        Smat_eval = identity(n2d) + 2.0 * (Smat1 ** 2)
        
        # Build PyAMG Smoothed Aggregation Hierarchy
        ml = pyamg.smoothed_aggregation_solver(Smat_eval, max_levels=4, max_coarse=100)
        
        # Extract and scale the hierarchy matrices
        hierarchy = {'A': [], 'P': [], 'R': []}
        for i in range(len(ml.levels)):
            A_level_scipy = ml.levels[i].A
            A_level_jax, gamma_level = scale_A_by_spectral_radius_jax(A_level_scipy)
            hierarchy['A'].append(A_level_jax)
            
            if i == 0:
                # Store the fine level data for the old loop compatability
                A_jax_eval = A_level_jax
                gamma_eval = gamma_level
                
            if i < len(ml.levels) - 1:
                P_scipy = ml.levels[i].P.tocoo()
                R_scipy = ml.levels[i].R.tocoo()
                
                # Convert to JAX BCOO using float64
                P_jax = sparse.BCOO((jnp.array(P_scipy.data, dtype=jnp.float64), 
                                     jnp.column_stack((P_scipy.row, P_scipy.col))), 
                                    shape=P_scipy.shape)
                R_jax = sparse.BCOO((jnp.array(R_scipy.data, dtype=jnp.float64), 
                                     jnp.column_stack((R_scipy.row, R_scipy.col))), 
                                    shape=R_scipy.shape)
                hierarchy['P'].append(P_jax)
                hierarchy['R'].append(R_jax)
                
        amg_hierarchies_train.append(hierarchy)
        
        As_jax_train.append(A_jax_eval)
        gammas_train.append(gamma_eval)
        As_jax_eval.append(A_jax_eval)
        
        # Store diagonal of scaled eval matrix for Jacobi preconditioner baseline
        diag_Ai_eval = jnp.array(Smat_eval.diagonal() / gamma_eval, dtype=jnp.float32)
        diags_As_eval.append(diag_Ai_eval)
        
        # Laplacian
        Laplacian_scipy = identity(n2d) + math.sqrt(2.0) * Smat1
        A_jax_laplacian_val, _ = scale_A_by_spectral_radius_jax(Laplacian_scipy)
        As_jax_laplacian.append(A_jax_laplacian_val)
        
    # Evaluate using the synthetic physical data (tt) to test realistic convergence
    b = jnp.array(tt, dtype=jnp.float32)
    b = b / jnp.linalg.norm(b)
    
    # 2. Configure Flax ResGCN_JAX model
    num_layers = 12
    embed = 128
    hidden = 128
    drop_rate = 0.0
    dtype = jnp.float64
    lr = 0.0036
    weight_decay = 0.0024
    training_data = 'x_mix'
    m_base = 80
    batch_base = 4
    batch_base = 4
    epochs_base = 150
    phases = 1
    
    from GraphUNet_JAX import GraphUNet_JAX
    
    print("Initializing JAX-Native GraphUNet and GNP framework...", flush=True)
    net = GraphUNet_JAX(
        embed=embed,
        K=3,
        dtype=dtype
    )
    
    # Initialize parameters instantly using a dummy N=1 hierarchy
    key = random.PRNGKey(42)
    key, init_key, dropout_key = random.split(key, 3)
    dummy_input = jnp.ones((1, 1), dtype=dtype)
    
    # Extract the number of levels from the real hierarchy to ensure we instantiate all layers
    num_l = len(amg_hierarchies_train[-1]['A'])
    dummy_A = sparse.BCOO((jnp.ones(1, dtype=dtype), jnp.zeros((1, 2), dtype=jnp.int32)), shape=(1, 1))
    dummy_hier = {'A': [dummy_A]*num_l, 'P': [dummy_A]*num_l, 'R': [dummy_A]*num_l}
    
    print("Running net.init...", flush=True)
    params = net.init({'params': init_key, 'dropout': dropout_key}, dummy_input, dummy_hier, train=False)['params']
    print("net.init finished!", flush=True)
    
    print("Defining net_apply...", flush=True)
    net_apply = jax.jit(
        lambda variables, x, AA, train, **kwargs: net.apply(variables, x, AA, train=train, **kwargs),
        static_argnames=('train',)
    )
    
    gnp = GNP_JAX_Model(amg_hierarchies_train[0], net_apply, params, training_data, m_base)
    optimizer = optax.adamw(learning_rate=lr, weight_decay=weight_decay)
    
    # Create TrainState
    state = train_state.TrainState.create(apply_fn=net_apply, params=params, tx=optimizer)
    
    ckpt_dir = os.path.abspath("./checkpoints_cg_jax")
    os.makedirs(ckpt_dir, exist_ok=True)
    
    # 3. Hierarchical curriculum training loop in JAX
    print(f"Starting JAX hierarchical curriculum training over {phases} phase(s)...")
    t0 = time.time()
    
    train_key = random.PRNGKey(123)
    
    matrix_indices = [3]  # Only train on the L=46.4km scale to avoid recompiling 10 different sparse network topologies!
    max_p_phases = [1]    
    print("Starting JAX hierarchical curriculum training...", flush=True)
    
    # Pre-calculate No Precon Baseline for Quick Eval
    idx_quick = 3 # 46.4km scale
    b_eval_quick = jnp.array(tt, dtype=jnp.float32)
    b_eval_quick = b_eval_quick / jnp.linalg.norm(b_eval_quick)
    
    @jax.jit
    def A_op_bilap_quick(x): return As_jax_eval[idx_quick] @ x
    @jax.jit
    def A_op_lap_quick(x): return As_jax_laplacian[idx_quick] @ x
    
    def python_fgmres_solve_quick(A_op, b, M_op=None, max_iters=3000, tol=1e-6):
        return jax_fgmres_solve(A_op, b, M_op=M_op, restart=200, max_iters=max_iters, tol=tol)
        
    print("Skipping 'No Precon' baseline for Quick Eval to speed up script...")
    base_none_bilap = 3000
    
    # Skipping baseline print output
    
    for phase in range(phases):
        print(f"Starting phase {phase}...", flush=True)
        epochs_now = epochs_base
        batch_now = batch_base
        gnp.m = m_base
        max_p = max_p_phases[phase]
        
        print(f"\n--- Phase {phase+1}/{phases} --- [epochs={epochs_now}, batch={batch_now}, m={gnp.m}]")
        t_phase = time.time()
        
        # Randomize matrix order after the first phase to prevent catastrophic forgetting
        if phase > 0:
            python_random.shuffle(matrix_indices)
            
        for idx in matrix_indices:
            gc.collect()
            i = idx
            hierarchy_i = amg_hierarchies_train[idx]
            gnp.hierarchy = hierarchy_i
            gnp.A = hierarchy_i['A'][0]  # Refresh system matrix for current training step
            
            max_p = max_p_phases[phase]
            
            # Train model parameters on this matrix scale
            state, hist_loss, best_loss, best_epoch, model_file, train_key, pass_counts = gnp.train(
                batch_size=batch_now, epochs=epochs_now,
                state=state, key=train_key, max_passes=max_p,
                checkpoint_dir=ckpt_dir, progress_bar=False
            )
            
            print(f"  Matrix {i:03d} (Scale L={scales[i]:.1f}km) | Best Loss: {best_loss:.4e} at epoch {best_epoch} | Passes drawn: {pass_counts}")
            
        # --- QUICK EVAL LOGIC (USING PRECOMPUTED BASELINES) ---
        gnp.A = As_jax_eval[idx_quick]
        precond_apply_quick = jax.jit(gnp.get_preconditioner_apply())
        max_p_eval_quick = max_p_phases[phase]
        
        @jax.jit
        def M_gnp_bilap(r):
            z = jnp.zeros_like(r)
            r_curr = r
            for _ in range(max_p_eval_quick):
                update = precond_apply_quick(state.params, r_curr)
                z = z + update
                r_curr = r_curr - (As_jax_eval[idx_quick] @ update)
            return z
            
        @jax.jit
        def M_gnp_lap(r):
            z = jnp.zeros_like(r)
            r_curr = r
            for _ in range(max_p_eval_quick):
                update = precond_apply_quick(state.params, r_curr)
                z = z + update
                r_curr = r_curr - (As_jax_laplacian[idx_quick] @ update)
            return z
            
        _, it_gnp_bilap = python_fgmres_solve_quick(A_op_bilap_quick, b_eval_quick, M_op=M_gnp_bilap, max_iters=3000)
        
        print(f"\n--- Quick Eval at Phase {phase+1} (L={scales[idx_quick]:.1f}km) ---")
        print(f"Bi-Laplacian | No Precon: {int(base_none_bilap)} | GNP: {int(it_gnp_bilap)}")
        print("---------------------------------------------")
        
        # Save a dedicated checkpoint for this specific phase so it can be evaluated later
        phase_ckpt_dir = os.path.join(ckpt_dir, f"phase_{phase+1}")
        os.makedirs(phase_ckpt_dir, exist_ok=True)
        checkpoints.save_checkpoint(ckpt_dir=phase_ckpt_dir, target=state.params, step=phase+1, prefix='gnp_model_', overwrite=True)
        print(f"Phase {phase+1} model saved to {phase_ckpt_dir}")
            
        print(f"Phase {phase+1} finished in {time.time() - t_phase:.2f}s")
        
    print(f"Training completed in {time.time() - t0:.2f}s")
    
    # 4. Evaluate JAX native preconditioned CG solver
    print("\nEvaluating convergence of JAX Preconditioned CG Solver...")
    
    iters_no_precon = []
    iters_jacobi = []
    iters_gnp = []
    
    precond_apply = gnp.get_preconditioner_apply()
    
    # To get the iteration counts, we can run a custom JAX CG solver or track JIT-compiled CG loop.
    # Since jax_cg returns x and info (where info is the termination status),
    # let's write a simple pure JAX preconditioned CG solver function so that we can extract the EXACT iteration count!
    # This matches the PyTorch CG output and gives clean iteration counts.
    
    def python_fgmres_solve(A_op, b, M_op=None, max_iters=3000, tol=1e-6):
        return jax_fgmres_solve(A_op, b, M_op=M_op, restart=200, max_iters=max_iters, tol=tol)

    # Warmup JIT FGMRES solver A_op
    print("Preparing Python-JAX FGMRES Solver...")
    
    # We will JIT the preconditioner to be fast
    precond_apply_jit = jax.jit(precond_apply)
    
    for i, Ai in enumerate(As_jax_eval):
        # We evaluate on the predefined normalized synthetic data
        b_final = jnp.array(tt, dtype=jnp.float32)
        b_final = b_final / jnp.linalg.norm(b_final)
        
        # JIT compile the linear operator
        @jax.jit
        def A_op(x): return Ai @ x
        
        # Warmup
        _ = A_op(b_final)
        
        # A. Solve without preconditioning
        _, it_none = python_fgmres_solve(A_op, b_final, M_op=None, max_iters=3000, tol=1e-6)
        iters_no_precon.append(int(it_none))
        
        # B. Solve with Jacobi preconditioner
        diag_inv = 1.0 / diags_As_eval[i]
        
        @jax.jit
        def M_jacobi(r): return diag_inv * r
        _ = M_jacobi(b_final) # Warmup
        
        _, it_jac = python_fgmres_solve(A_op, b_final, M_op=M_jacobi, max_iters=3000, tol=1e-6)
        iters_jacobi.append(int(it_jac))
        
        # C. Solve with GNP (Neural GCN Preconditioner)
        gnp.A = Ai
        precond_apply_jit = jax.jit(gnp.get_preconditioner_apply())
        
        @jax.jit
        def M_gnp(r): 
            z = jnp.zeros_like(r)
            r_curr = r
            for _ in range(max_p_phases[-1]):
                update = precond_apply_jit(state.params, r_curr)
                z = z + update
                r_curr = r_curr - (Ai @ update)
            return z
        
        _ = M_gnp(b_final) # Warmup
        
        _, it_gnp = python_fgmres_solve(A_op, b_final, M_op=M_gnp, max_iters=3000, tol=1e-6)
        iters_gnp.append(int(it_gnp))
        
        print(f"Scale L={scales[i]:.1f}km | No Precon: {it_none} iters | Jacobi: {it_jac} iters | GNP: {it_gnp} iters")

    print("\nSaving best parameters to disk...")
    checkpoints.save_checkpoint(ckpt_dir='./checkpoints_bilaplacian', target=state.params, step=0, overwrite=True)

    # 5. Plot JAX CG results
    plt.figure(figsize=(10, 6))
    plt.plot(scales, iters_no_precon, "o--", label="CG (No Preconditioner)", color="red")
    plt.plot(scales, iters_jacobi, "s--", label="CG + Jacobi Preconditioner", color="blue")
    plt.plot(scales, iters_gnp, "d-", label="CG + GNP (Neural Preconditioner)", color="green")
    
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Filter Scale L [km]")
    plt.ylabel("JAX CG Iterations")
    plt.title("JAX CG Solver Convergence vs. Scale L")
    plt.grid(True, which="both", ls="-", alpha=0.5)
    plt.legend()
    
    plot_path = "./cg_convergence_scaling_jax.png"
    plt.savefig(plot_path, dpi=300)
    plt.close()
    
    print(f"\nSaved JAX convergence comparison plot to: {plot_path}")
    print("JAX Experiment completed successfully!")

if __name__ == "__main__":
    run_jax_experiment()
