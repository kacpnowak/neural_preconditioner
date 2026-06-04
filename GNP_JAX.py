import jax
import jax.numpy as jnp
from jax import random
from flax.training import train_state, checkpoints
import optax
import os
import numpy as np
from tqdm import tqdm
from typing import Callable, Optional, Generator


def arnoldi_build(A: jnp.ndarray, m: int = 10, v0: Optional[jnp.ndarray] = None,
                  key: Optional[random.PRNGKey] = None):
    """
    Arnoldi iteration using standard loops.
    """
    n = A.shape[0]
    if v0 is None:
        v0 = random.normal(key, (n,), dtype=A.dtype)

    beta = jnp.linalg.norm(v0)
    V = jnp.zeros((n, m + 1), dtype=A.dtype)
    H = jnp.zeros((m + 1, m), dtype=A.dtype)

    V = V.at[:, 0].set(v0 / beta)

    for j in range(m):
        w = A @ V[:, j]
        for k in range(j + 1):
            h_k_j = jnp.dot(V[:, k], w)
            H = H.at[k, j].set(h_k_j)
            w = w - h_k_j * V[:, k]
        h_j_plus_1_j = jnp.linalg.norm(w)
        H = H.at[j + 1, j].set(h_j_plus_1_j)
        V = V.at[:, j + 1].set(w / h_j_plus_1_j)

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

    def __init__(self, A: jnp.ndarray, net_apply: Callable, net_params, training_data: str, m: int):
        self.A = A
        self.net_apply = net_apply
        self.training_data = training_data
        self.m = m
        self.dtype = next(iter(jax.tree_util.tree_leaves(net_params))).dtype

    def train(self, batch_size: int, epochs: int, state: train_state.TrainState,
              checkpoint_dir: Optional[str] = None, progress_bar: bool = True, max_passes: int = 1, key=None):

        # JIT compiled sparse matrix multiplication
        A_mat = self.A
        @jax.jit
        def apply_A(v):
            return A_mat @ v

        @jax.jit
        def train_step(state, x_or_b, k_idx, dropout_key):
            x_base = x_or_b.astype(self.dtype)
            
            def l1_loss(pred, target):
                return jnp.mean(jnp.abs(pred - target))

            def loss_fn(params):
                def apply_M(val):
                    return state.apply_fn({'params': params}, val, A_mat, train=True, rngs={'dropout': dropout_key})
                
                b_target = apply_A(x_base) if self.training_data != 'no_x' else x_base
                
                def pass_1(_):
                    x_out = apply_M(b_target)
                    b_out = apply_A(x_out)
                    return l1_loss(b_out, b_target)
                    
                def pass_2(_):
                    x_1 = apply_M(b_target)
                    r_1 = b_target - apply_A(x_1)
                    x_out = x_1 + apply_M(r_1)
                    b_out = apply_A(x_out)
                    return l1_loss(b_out, b_target)

                def pass_3(_):
                    x_1 = apply_M(b_target)
                    r_1 = b_target - apply_A(x_1)
                    x_2 = x_1 + apply_M(r_1)
                    r_2 = b_target - apply_A(x_2)
                    x_out = x_2 + apply_M(r_2)
                    b_out = apply_A(x_out)
                    return l1_loss(b_out, b_target)
                    
                def pass_4(_):
                    x_1 = apply_M(b_target)
                    r_1 = b_target - apply_A(x_1)
                    x_2 = x_1 + apply_M(r_1)
                    r_2 = b_target - apply_A(x_2)
                    x_3 = x_2 + apply_M(r_2)
                    r_3 = b_target - apply_A(x_3)
                    x_out = x_3 + apply_M(r_3)
                    b_out = apply_A(x_out)
                    return l1_loss(b_out, b_target)

                def pass_5(_):
                    x_1 = apply_M(b_target)
                    r_1 = b_target - apply_A(x_1)
                    x_2 = x_1 + apply_M(r_1)
                    r_2 = b_target - apply_A(x_2)
                    x_3 = x_2 + apply_M(r_2)
                    r_3 = b_target - apply_A(x_3)
                    x_4 = x_3 + apply_M(r_3)
                    r_4 = b_target - apply_A(x_4)
                    x_out = x_4 + apply_M(r_4)
                    b_out = apply_A(x_out)
                    return l1_loss(b_out, b_target)
                    
                branches = [pass_1, pass_2, pass_3, pass_4, pass_5]
                return jax.lax.switch(k_idx, branches, None)

            loss, grads = jax.value_and_grad(loss_fn)(state.params)
            state = state.apply_gradients(grads=grads)
            return state, loss

        if key is None:
            key = random.PRNGKey(0)
            
        key, data_key = random.split(key)
        # Assuming run_cg_preconditioner imports create_streaming_dataset_jax locally now
        # We will keep the default data generation if called normally
        try:
            from synthetic_data_generator import create_streaming_dataset_jax
            data_generator = create_streaming_dataset_jax(self.A, batch_size, self.training_data, self.m, data_key)
        except:
            data_generator = create_streaming_dataset(self.A, batch_size, self.training_data, self.m, key)

        hist_loss = []
        best_loss = jnp.inf
        best_epoch = -1
        best_params = state.params

        if checkpoint_dir:
            os.makedirs(checkpoint_dir, exist_ok=True)

        pbar = tqdm(total=epochs, desc='Train') if progress_bar else None
        
        pass_counts = {}

        for epoch in range(epochs):
            batch = next(data_generator)
            key, k_key, drop_key = random.split(key, 3)
            
            k_passes = random.randint(k_key, shape=(), minval=1, maxval=max_passes + 1).item()
            k_idx = k_passes - 1
            pass_counts[k_passes] = pass_counts.get(k_passes, 0) + 1
            
            state, loss_val = train_step(state, batch, k_idx, drop_key)

            loss_item = loss_val.item()
            hist_loss.append(loss_item)

            if loss_item < best_loss:
                best_loss = loss_item
                best_epoch = epoch
                best_params = state.params
                if checkpoint_dir:
                    checkpoints.save_checkpoint(ckpt_dir=checkpoint_dir, target=state.params, step=best_epoch,
                                                prefix='gnp_model_', overwrite=True)

            if progress_bar:
                pbar.set_description(f'Train loss {loss_item:.1e}')
                pbar.update()

        if progress_bar:
            pbar.close()

        checkpoint_file = os.path.join(checkpoint_dir, 'gnp_model_' + str(best_epoch)) if checkpoint_dir and best_epoch != -1 else None
        return state.replace(params=best_params), hist_loss, best_loss, best_epoch, checkpoint_file, key, pass_counts

    def get_preconditioner_apply(self) -> Callable:
        """ Returns a callable for applying the preconditioner. """
        
        A_mat = self.A

        @jax.jit
        def apply_fn(params, r):
            r = r.reshape(-1, 1)
            z = self.net_apply({'params': params}, r, A_mat, train=False)
            return z.flatten()

        return apply_fn


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