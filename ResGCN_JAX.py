import jax
import jax.numpy as jnp
from jax.experimental import sparse
from flax import linen as nn

def scale_A_by_spectral_radius_jax(A_scipy):
    """
    Compute spectral radius estimate and scale matrix in SciPy,
    then convert to JAX BCOO sparse matrix.
    """
    abs_A = abs(A_scipy)
    row_sum = abs_A.sum(axis=1).A1
    col_sum = abs_A.sum(axis=0).A1
    gamma = min(max(row_sum), max(col_sum))
    scaled_A = A_scipy / gamma
    
    A_jax = sparse.BCOO.from_scipy_sparse(scaled_A)
    return A_jax, gamma

class MLP_JAX(nn.Module):
    in_dim: int
    out_dim: int
    num_layers: int
    hidden: int
    drop_rate: float
    dtype: jnp.dtype = jnp.float32
    use_layernorm: bool = False
    is_output_layer: bool = False

    @nn.compact
    def __call__(self, R, train: bool = True):
        for i in range(self.num_layers):
            is_last = (i == self.num_layers - 1)
            R = nn.Dense(features=self.out_dim if is_last else self.hidden, dtype=self.dtype)(R)
            if not is_last or not self.is_output_layer:
                if self.use_layernorm:
                    R = nn.LayerNorm(dtype=self.dtype)(R)
                R = nn.relu(R)
                if self.drop_rate > 0:
                    R = nn.Dropout(rate=self.drop_rate, deterministic=not train)(R)
        return R

class ChebConv_JAX(nn.Module):
    in_dim: int
    out_dim: int
    K: int = 3
    dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(self, R, AA):
        n, batch_size, in_dim = R.shape
        X = R.reshape(n, batch_size * in_dim)
        
        Z = [X]
        if self.K > 1:
            Z.append(AA @ X)
            for k in range(2, self.K):
                Z.append(2.0 * (AA @ Z[-1]) - Z[-2])
                
        out = 0.0
        for k in range(self.K):
            Z_reshaped = Z[k].reshape(n, batch_size, in_dim)
            out = out + nn.Dense(features=self.out_dim, dtype=self.dtype, name=f"lin_{k}")(Z_reshaped)
        return out

class GraphPool_JAX(nn.Module):
    pool_ratio: float = 0.5
    dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(self, X, A):
        n, batch_size, in_dim = X.shape
        k = max(1, int(n * self.pool_ratio))
        
        # Projection vector for scoring
        p = self.param('p', nn.initializers.normal(1.0), (in_dim,))
        p = p / (jnp.linalg.norm(p) + 1e-6)
        
        # Score computation (average score across batch)
        y = jnp.mean(X @ p, axis=1) # shape: (n,)
        
        # Select top-k nodes
        topk_vals, topk_idx = jax.lax.top_k(y, k)
        gate = jax.nn.sigmoid(topk_vals)
        gate = jnp.expand_dims(gate, axis=(1, 2)) # shape: (k, 1, 1)
        
        # Pool features
        X_pool = X[topk_idx] * gate
        
        # Convert A to dense for pooling
        A_dense = A.todense() if isinstance(A, sparse.BCOO) else A
        
        # Mathematically preserve graph connectivity across dropped nodes!
        # Standard Graph U-Net pooling squares the adjacency matrix (A^2) to connect 2-hop neighbors,
        # ensuring that if node B is dropped in the chain A-B-C, A and C remain connected!
        A_squared = A_dense @ A_dense
        A_pool = A_squared[topk_idx][:, topk_idx]
        
        # Normalize and clip back to standard adjacency bounds
        A_pool = jnp.clip(A_pool, 0.0, 1.0)
        
        return X_pool, A_pool, topk_idx

class GraphUnpool_JAX(nn.Module):
    @nn.compact
    def __call__(self, X_pool, topk_idx, target_n):
        k, batch_size, in_dim = X_pool.shape
        X_unpool = jnp.zeros((target_n, batch_size, in_dim), dtype=X_pool.dtype)
        X_unpool = X_unpool.at[topk_idx].set(X_pool)
        return X_unpool

class GAT_JAX(nn.Module):
    out_dim: int
    heads: int = 4
    dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(self, X, A):
        n, batch_size, in_dim = X.shape
        head_dim = self.out_dim // self.heads
        
        # Transform features
        W = self.param('W', nn.initializers.glorot_uniform(), (in_dim, self.heads * head_dim))
        X_flat = X.reshape(n * batch_size, in_dim)
        h = (X_flat @ W).reshape(n, batch_size, self.heads, head_dim)
        
        # For a massive graph, computing full attention is O(N^2).
        # We only apply this at the bottleneck where N is small, so we use a dense attention matrix.
        # Compute dense adjacency mask
        A_dense = A.todense() if isinstance(A, sparse.BCOO) else A
        mask = jnp.where(A_dense > 0, 0.0, -1e9) # mask disconnected edges
        
        # Self-attention formulation
        a_l = self.param('a_l', nn.initializers.glorot_uniform(), (head_dim, self.heads))
        a_r = self.param('a_r', nn.initializers.glorot_uniform(), (head_dim, self.heads))
        
        # Compute scores per node
        # h: (N, B, H, D)
        # We want to compute h @ a_l for each head.
        score_l = jnp.einsum('nbhd,dh->nbh', h, a_l) # (N, B, H)
        score_r = jnp.einsum('nbhd,dh->nbh', h, a_r) # (N, B, H)
        
        # Pairwise combinations via broadcasting
        # Combine and take mean over batch for the structural attention map
        score_l_mean = jnp.mean(score_l, axis=1) # (N, H)
        score_r_mean = jnp.mean(score_r, axis=1) # (N, H)
        
        e = score_l_mean[:, None, :] + score_r_mean[None, :, :] # (N, N, H)
        e = jax.nn.leaky_relu(e, negative_slope=0.2)
        
        # Apply structural mask
        e = e + jnp.expand_dims(mask, axis=-1)
        
        # Softmax
        alpha = jax.nn.softmax(e, axis=1) # (N, N, H)
        
        # Aggregate
        # alpha: (N, N, H), h: (N, B, H, D)
        # We want out: (N, B, H, D)
        out = jnp.einsum('nih,ibhd->nbhd', alpha, h)
        out = out.reshape(n, batch_size, self.out_dim)
        return nn.relu(out)

class GraphUNet_JAX(nn.Module):
    embed: int = 64
    hidden: int = 96
    drop_rate: float = 0.0
    dtype: jnp.dtype = jnp.float32
    scale_input: bool = True
    pool_ratio: float = 0.5
    depth: int = 3

    @nn.compact
    def __call__(self, r, AA, train: bool = False):
        n, batch_size = r.shape
        
        if self.scale_input:
            # Dynamically scale input to avoid float32 precision floor
            # We do NOT subtract the mean because that breaks linearity for M(r).
            # We only divide by std, and we MUST multiply it back at the end!
            std = jnp.std(r, axis=0, keepdims=True) + 1e-8
            r = r / std
            R = r.reshape(n, batch_size, 1)
        else:
            R = r.reshape(n, batch_size, 1)
        
        # Initial encoding
        R = MLP_JAX(in_dim=1, out_dim=self.embed, num_layers=2, 
                    hidden=self.hidden, drop_rate=self.drop_rate, dtype=self.dtype)(R, train=train)
        
        A_curr = AA
        
        # Encoder (Downsampling)
        encoded_features = []
        encoded_indices = []
        encoded_sizes = []
        encoded_As = []
        
        for d in range(self.depth):
            # Smooth
            R = ChebConv_JAX(in_dim=self.embed, out_dim=self.embed, K=3, dtype=self.dtype)(R, A_curr)
            R = nn.LayerNorm(dtype=self.dtype)(R)
            R = nn.relu(R)
            
            # Save state for skip connection
            encoded_features.append(R)
            encoded_sizes.append(A_curr.shape[0])
            encoded_As.append(A_curr)
            
            # Pool
            R, A_curr, idx = GraphPool_JAX(pool_ratio=self.pool_ratio, dtype=self.dtype)(R, A_curr)
            encoded_indices.append(idx)
            
        # Bottleneck (GAT)
        R = GAT_JAX(out_dim=self.embed, heads=4, dtype=self.dtype)(R, A_curr)
        R = nn.LayerNorm(dtype=self.dtype)(R)
        R = nn.relu(R)
        
        # Decoder (Upsampling)
        for d in reversed(range(self.depth)):
            # Unpool
            idx = encoded_indices[d]
            target_n = encoded_sizes[d]
            R = GraphUnpool_JAX()(R, idx, target_n)
            
            # Skip Connection
            R = R + encoded_features[d]
            
            # Smooth
            A_decode = encoded_As[d]
            R = ChebConv_JAX(in_dim=self.embed, out_dim=self.embed, K=3, dtype=self.dtype)(R, A_decode)
            R = nn.LayerNorm(dtype=self.dtype)(R)
            R = nn.relu(R)
            
        # Final projection layer
        z = MLP_JAX(in_dim=self.embed, out_dim=1, num_layers=4, 
                    hidden=self.hidden, drop_rate=self.drop_rate, dtype=self.dtype,
                    is_output_layer=True)(R, train=train)
        
        z = z.reshape(n, batch_size)
        if self.scale_input:
            z = z * std
            
        return z

class ResGCN_JAX(nn.Module):
    embed: int = 128
    hidden: int = 128
    drop_rate: float = 0.0
    dtype: jnp.dtype = jnp.float32
    scale_input: bool = True
    depth: int = 6
    K: int = 8

    @nn.compact
    def __call__(self, r, AA, train: bool = False):
        n, batch_size = r.shape
        
        if self.scale_input:
            std = jnp.std(r, axis=0, keepdims=True) + 1e-8
            r = r / std
            R = r.reshape(n, batch_size, 1)
        else:
            R = r.reshape(n, batch_size, 1)
        
        # Initial encoding
        R = MLP_JAX(in_dim=1, out_dim=self.embed, num_layers=2, 
                    hidden=self.hidden, drop_rate=self.drop_rate, dtype=self.dtype)(R, train=train)
        
        # Deep Residual ChebConv
        for d in range(self.depth):
            R_skip = R
            R = ChebConv_JAX(in_dim=self.embed, out_dim=self.embed, K=self.K, dtype=self.dtype)(R, AA)
            R = nn.LayerNorm(dtype=self.dtype)(R)
            R = nn.relu(R)
            R = R + R_skip
            
        # Final projection layer
        z = MLP_JAX(in_dim=self.embed, out_dim=1, num_layers=3, 
                    hidden=self.hidden, drop_rate=self.drop_rate, dtype=self.dtype,
                    is_output_layer=True)(R, train=train)
        
        z = z.reshape(n, batch_size)
        if self.scale_input:
            z = z * std
            
        return z
