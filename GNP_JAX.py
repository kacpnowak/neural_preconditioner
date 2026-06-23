import jax
import jax.numpy as jnp
from jax import random
from flax.training import train_state, checkpoints
import optax
import os
import numpy as np
from tqdm import tqdm
from typing import Callable, Optional, Generator


from functools import partial

@partial(jax.jit, static_argnames=['m'])
def arnoldi_build(A, m, key):
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


def create_streaming_dataset(A: jnp.ndarray, batch_size: int, training_data: str, m: int,
                             key: random.PRNGKey) -> Generator:
    """szaa
    A generator that yields batches of data, replacing the PyTorch IterableDataset.
    """
    n = A.shape[0]
    Q = None
    if training_data in ['x_subspace', 'x_mix']:
        key, subkey = random.split(key)
        Vm1, barHm = arnoldi_build(A, m=m, key=subkey)
        _W, S, Zh = jnp.linalg.svd(barHm, full_matrices=False)
        Q = (Vm1[:, :-1] @ Zh.T) / S.reshape(1, m)
        Q = jax.block_until_ready(Q)

    while True:
        key, subkey = random.split(key)
        if training_data == 'x_normal':
            yield random.normal(subkey, (n, batch_size), dtype=jnp.float64)
        elif training_data == 'x_subspace':
            e = random.normal(subkey, (m, batch_size), dtype=jnp.float64)
            yield Q @ e
        elif training_data == 'x_mix':
            batch_size1 = batch_size // 2
            batch_size2 = batch_size - batch_size1
            key1, key2 = random.split(subkey)
            e = random.normal(key1, (m, batch_size1), dtype=jnp.float64)
            x1 = Q @ e
            x2 = random.normal(key2, (n, batch_size2), dtype=jnp.float64)
            yield jnp.concatenate([x1, x2], axis=1)
        else:  # 'no_x'
            yield random.normal(subkey, (n, batch_size), dtype=jnp.float64)


class GNP_JAX_Model:
    """
    Graph Neural Preconditioner implementation using JAX and Flax.
    """

    def __init__(self, A: jnp.ndarray, net_apply: Callable, net_params, training_data: str, m: int, custom_b_train: Optional[jnp.ndarray] = None):
        self.A = A
        self.net_apply = net_apply
        self.training_data = training_data
        self.m = m
        self.custom_b_train = custom_b_train
        self.dtype = next(iter(jax.tree_util.tree_leaves(net_params))).dtype

    def train(self, batch_size: int, epochs: int, state: train_state.TrainState,
              checkpoint_dir: Optional[str] = None, progress_bar: bool = True, max_passes: int = 1, key=None):

        # JIT compiled sparse matrix multiplication
        A_mat = self.A
        @jax.jit
        def apply_A(v):
            return A_mat @ v

        @jax.jit
        def train_step(state, x_or_b, k_idx, dropout_key, dyn_A, dyn_hier):
            x_base = x_or_b.astype(self.dtype)
            
            def l1_loss(pred, target):
                return jnp.mean(jnp.abs(pred - target))

            def loss_fn(params):
                def apply_M(val):
                    return state.apply_fn({'params': params}, val, dyn_hier, train=True, rngs={'dropout': dropout_key})
                
                def apply_A(v):
                    return dyn_A @ v
                
                b_target = apply_A(x_base) if self.training_data not in ['no_x', 'b_custom'] else x_base
                
                # Since we restricted training to 1 pass (V-cycle), we can just hardcode the single pass natively!
                # This completely eliminates JAX scan list-capturing unroll blowup.
                x_out = apply_M(b_target)
                b_out = apply_A(x_out)
                return l1_loss(b_out, b_target)

            loss, grads = jax.value_and_grad(loss_fn)(state.params)
            state = state.apply_gradients(grads=grads)
            return state, loss

        # Pre-split keys for the scan loop
        key, scan_key = random.split(key)
        epoch_keys = random.split(scan_key, epochs)
        
        # Pre-generate k_idxs
        key, pass_key = random.split(key)
        k_passes_arr = random.randint(pass_key, shape=(epochs,), minval=1, maxval=max_passes + 1)
        k_idxs_arr = k_passes_arr - 1
        
        # Compute pass counts for logging
        unique, counts = np.unique(np.array(k_passes_arr), return_counts=True)
        pass_counts = dict(zip(unique, counts))

        # Generate Q once eagerly if needed
        Q = jnp.zeros((1, 1), dtype=self.dtype)
        if self.training_data in ['x_subspace', 'x_mix']:
            key, subkey = random.split(key)
            Vm1, barHm = arnoldi_build(self.A, m=self.m, key=subkey)
            _W, S, Zh = jnp.linalg.svd(barHm, full_matrices=False)
            Q = (Vm1[:, :-1] @ Zh.T) / S.reshape(1, self.m)
            Q = jax.block_until_ready(Q)

        A_mat = self.A
        
        # Chop up the JIT: We only JIT the train_step. The training loop runs in Python.
        # This prevents XLA from attempting to globally optimize 2000 epochs of the GraphUNet.
        hist_loss = []
        for i in range(epochs):
            epoch_key = epoch_keys[i]
            k_idx = k_idxs_arr[i]
            
            # Generate data
            if self.training_data == 'x_normal':
                batch = random.normal(epoch_key, (A_mat.shape[0], batch_size), dtype=self.dtype)
            elif self.training_data == 'x_subspace':
                e = random.normal(epoch_key, (self.m, batch_size), dtype=self.dtype)
                batch = jnp.dot(Q, e)
            elif self.training_data == 'x_mix':
                batch_size1 = batch_size // 2
                batch_size2 = batch_size - batch_size1
                key1, key2 = random.split(epoch_key)
                
                batch1 = random.normal(key1, (A_mat.shape[0], batch_size1), dtype=self.dtype)
                e = random.normal(key2, (self.m, batch_size2), dtype=self.dtype)
                batch2 = jnp.dot(Q, e)
                batch = jnp.concatenate((batch1, batch2), axis=1)
            elif self.training_data == 'no_x':
                batch = random.normal(epoch_key, (A_mat.shape[0], batch_size), dtype=self.dtype)
            elif self.training_data == 'b_custom':
                idx = random.randint(epoch_key, shape=(batch_size,), minval=0, maxval=self.custom_b_train.shape[1])
                batch = self.custom_b_train[:, idx]

            # Drop key for dropout
            drop_key, _ = random.split(epoch_key)
            
            state, loss_val = train_step(state, batch, k_idx, drop_key, A_mat, self.hierarchy)
            
            if progress_bar and i % 10 == 0:
                print(f"Epoch {i}/{epochs} | Step Loss: {loss_val:.6e}", flush=True)
                
            hist_loss.append(loss_val)
            
        state = jax.block_until_ready(state)
        
        hist_loss = np.array(hist_loss)
        best_epoch = np.argmin(hist_loss)
        best_loss = hist_loss[best_epoch]
        best_params = state.params
        
        if checkpoint_dir:
            os.makedirs(checkpoint_dir, exist_ok=True)
            checkpoints.save_checkpoint(ckpt_dir=checkpoint_dir, target=state.params, step=int(best_epoch),
                                        prefix='gnp_model_', overwrite=True)

        checkpoint_file = os.path.join(checkpoint_dir, 'gnp_model_' + str(best_epoch)) if checkpoint_dir and best_epoch != -1 else None
        return state.replace(params=best_params), hist_loss, best_loss, best_epoch, checkpoint_file, key, pass_counts

    def get_preconditioner_apply(self) -> Callable:
        """ Returns a callable for applying the preconditioner. """
        
        # We must capture the hierarchy dynamically inside the wrapper
        # to prevent XLA from constant folding the sparse indices at compile time.
        
        @jax.jit
        def apply_fn(params, r, dyn_hier):
            r = r.reshape(-1, 1)
            z = self.net_apply({'params': params}, r, dyn_hier, train=False)
            return z.flatten()

        def wrapper_apply(params, r):
            return apply_fn(params, r, self.hierarchy)

        return wrapper_apply


if __name__ == '__main__':
    from flax import linen as nn


    class SimpleNet(nn.Module):
        @nn.compact
        def __call__(self, x):
            x = nn.Dense(features=128)(x)
            x = nn.relu(x)
            x = nn.Dense(features=x.shape[-1])(x)
            return x


    key = random.PRNGKey(42)
    n, m, batch_size, epochs = 100, 20, 16, 100

    key, subkey = random.split(key)
    A = random.normal(subkey, (n, n), dtype=jnp.float32)
    A = A + A.T

    net = SimpleNet()
    key, subkey1, subkey2 = random.split(key, 3)
    dummy_input = random.normal(subkey1, (n, 1), dtype=jnp.float32)
    params = net.init(subkey2, dummy_input)['params']

    gnp = GNP(A=A, net_apply=net.apply, net_params=params, training_data='x_mix', m=m)

    optimizer = optax.adam(learning_rate=1e-3)
    hist_loss, best_loss, best_epoch, ckpt_file = gnp.train(
        batch_size=batch_size, grad_accu_steps=1, epochs=epochs,
        optimizer=optimizer, params=params, checkpoint_dir='./checkpoints'
    )

    print(f"\nTraining finished.")
    print(f"Best Loss: {best_loss:.4f} at epoch {best_epoch}")
    print(f"Best checkpoint file: {ckpt_file}")

    if ckpt_file and os.path.exists(ckpt_file):
        loaded_params = checkpoints.restore_checkpoint(ckpt_dir='./checkpoints', target=params, prefix='gcn_precond')

        key, subkey = random.split(key)
        r_vector = random.normal(subkey, (n,), dtype=jnp.float32)

        preconditioner_apply = gnp.apply
        z_vector = preconditioner_apply(loaded_params, r_vector)

        print("\nApplied preconditioner to a random vector:")
        print("Input shape:", r_vector.shape)
        print("Output shape:", z_vector.shape)