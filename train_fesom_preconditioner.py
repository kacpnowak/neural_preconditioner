import os
import gc
import time
import numpy as np
import xarray as xr
from scipy.sparse import csc_matrix, identity
import jax
import jax.numpy as jnp
from jax import random
from flax.training import train_state
import optax
import pyamg

# Import custom modules
from GNP_JAX import GNP_JAX_Model
from GraphUNet_JAX import GraphUNet_JAX
from run_fesom_filter import create_fesom_matrix_fast

def count_parameters(params):
    return sum(x.size for x in jax.tree_util.tree_leaves(params))

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
    
    k = 0.05 # Scale parameter
    Smat1 = csc_matrix((ss * (1.0 / np.square(k)), (ii, jj)), shape=(n2d, n2d))
    A_scipy = identity(n2d) + 2.0 * (Smat1 ** 2)
    
    from run_fesom_filter import scale_A_by_spectral_radius_jax
    
    print("Scaling FESOM matrix by spectral radius to stabilize UNet...")
    A_jax_Lap, spectral_radius = scale_A_by_spectral_radius_jax(A_scipy)
    
    print("Building PyAMG Smoothed Aggregation hierarchy...")
    ml = pyamg.smoothed_aggregation_solver(
        A_scipy,
        max_levels=5,
        max_coarse=10,
        keep=True
    )
    
    hierarchy_dict = {
        'A': [],
        'P': [],
        'R': []
    }
    
    hierarchy_dict['A'].append(A_jax_Lap)
    
    for i in range(len(ml.levels)):
        if i > 0:
            A_level = ml.levels[i].A.tocoo()
            A_jax = jax.experimental.sparse.BCOO(
                (A_level.data.astype(np.float64), np.column_stack((A_level.row, A_level.col))),
                shape=A_level.shape
            )
            hierarchy_dict['A'].append(A_jax)
        
        if i < len(ml.levels) - 1:
            P_level = ml.levels[i].P.tocoo()
            P_jax = jax.experimental.sparse.BCOO(
                (P_level.data.astype(np.float64), np.column_stack((P_level.row, P_level.col))),
                shape=P_level.shape
            )
            hierarchy_dict['P'].append(P_jax)
            
            R_level = ml.levels[i].R.tocoo()
            R_jax = jax.experimental.sparse.BCOO(
                (R_level.data.astype(np.float64), np.column_stack((R_level.row, R_level.col))),
                shape=R_level.shape
            )
            hierarchy_dict['R'].append(R_jax)
    
    hierarchy_list = [hierarchy_dict]
    
    model = GraphUNet_JAX(
        embed=128,
        K=3,
        dtype=jnp.float64
    )
    
    print("Initializing GraphUNet_JAX...")
    # Initialize parameters
    rng = random.PRNGKey(0)
    rng, init_rng = random.split(rng)
    dummy_x = jnp.ones((n2d, 1), dtype=jnp.float64)
    
    init_variables = jax.jit(model.init)(
        {'params': init_rng, 'dropout': init_rng}, 
        dummy_x, 
        hierarchy_dict,
        train=False
    )
    params = init_variables['params']
    
    total_params = count_parameters(params)
    print(f"Total model parameters: {total_params}")
    
    # Optional: Load pretrained checkpoint as starting point instead of scratch
    # Wait, the user said "make preconditioner for bi-laplacian on this data". Let's start from the 1977 pretrained checkpoint to leverage synthetic features.
    from flax.training import checkpoints
    try:
        ckpt_dir_old = os.path.abspath("./checkpoints_bilaplacian")
        restored_state = checkpoints.restore_checkpoint(ckpt_dir=ckpt_dir_old, target=None)
        if restored_state is not None:
            params = restored_state['params']
            print("Successfully loaded pre-trained synthetic weights to fine-tune!")
    except Exception as e:
        print("Could not load pre-trained weights, starting from scratch. ", e)
    
    # Full training capacity from original paper configuration
    epochs = 300
    batch_size = 4
    m_base = 10
    lr = 5e-4
    weight_decay = 1e-4
    
    gnp = GNP_JAX_Model(
        A=hierarchy_dict['A'][0], 
        net_apply=model.apply, 
        net_params=params, 
        training_data='x_mix', 
        m=m_base
    )
    gnp.hierarchy = hierarchy_dict
    
    optimizer = optax.adamw(learning_rate=lr, weight_decay=weight_decay)
    state = train_state.TrainState.create(apply_fn=model.apply, params=params, tx=optimizer)
    
    ckpt_dir_new = os.path.abspath("./checkpoints_fesom_300")
    os.makedirs(ckpt_dir_new, exist_ok=True)
    
    print(f"Starting FESOM Preconditioner JAX Training for {epochs} epochs...")
    t0 = time.time()
    train_key = random.PRNGKey(123)
    
    state, hist_loss, best_loss, best_epoch, model_file, train_key, pass_counts = gnp.train(
        batch_size=batch_size, 
        epochs=epochs,
        state=state, 
        key=train_key, 
        max_passes=1,
        checkpoint_dir=ckpt_dir_new, 
        progress_bar=True
    )
    
    t1 = time.time()
    print(f"Training completed in {t1-t0:.2f} seconds!")
    print(f"Best Loss: {best_loss:.4e} at epoch {best_epoch}")
    print(f"Model saved to: {model_file}")

if __name__ == "__main__":
    main()
