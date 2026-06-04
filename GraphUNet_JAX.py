import jax
import jax.numpy as jnp
from jax.experimental import sparse
import flax.linen as nn
from ResGCN_JAX import ChebConv_JAX

class GraphPool_JAX(nn.Module):
    """
    JAX-Native Top-K Pooling layer. 
    It computes a projection score, keeps the Top-K nodes, and filters the adjacency matrix.
    Crucially, it uses an argsort-and-slice trick to physically shrink the BCOO `nse` 
    (number of sparse elements) so the network actually saves FLOPs and Memory at coarse levels,
    while remaining 100% XLA compilable.
    """
    pool_ratio: float = 0.5
    nse_ratio: float = 0.5  # How much to shrink the number of edges
    
    @nn.compact
    def __call__(self, X, A: sparse.BCOO):
        n = X.shape[0]
        nse = A.indices.shape[0]
        k = max(1, int(n * self.pool_ratio))
        k_nse = max(1, int(nse * self.nse_ratio))
        
        # 1. Projection score
        p = self.param('p', nn.initializers.glorot_uniform(), (X.shape[-1], 1))
        p_norm = p / (jnp.linalg.norm(p) + 1e-8)
        # X is (n, batch, embed). p_norm is (embed, 1). 
        # (X @ p_norm) is (n, batch, 1). We squeeze to (n, batch) and mean across batch.
        y = jnp.mean((X @ p_norm).squeeze(-1), axis=1) # shape: (n,)
        
        # 2. Top-K Selection
        topk_vals, topk_idx = jax.lax.top_k(y, k)
        topk_idx = jnp.sort(topk_idx)  # Sort to preserve spatial locality
        
        # 3. Create boolean mask
        keep_mask = jnp.zeros(n, dtype=jnp.bool_).at[topk_idx].set(True)
        
        # 4. Filter Adjacency Matrix
        row_idx = A.indices[:, 0]
        col_idx = A.indices[:, 1]
        valid_edges = keep_mask[row_idx] & keep_mask[col_idx]
        
        # Remap indices to the smaller graph (0 to K-1)
        new_indices_map = jnp.zeros(n, dtype=jnp.int32)
        new_indices_map = new_indices_map.at[topk_idx].set(jnp.arange(k, dtype=jnp.int32))
        
        new_row = new_indices_map[row_idx]
        new_col = new_indices_map[col_idx]
        new_data = jnp.where(valid_edges, A.data, 0.0)
        
        # 5. Shrink the NSE dynamically!
        # We sort by ~valid_edges so all True (valid) edges move to the front of the array.
        sort_idx = jnp.argsort(~valid_edges)
        
        # Apply the sort
        sorted_row = new_row[sort_idx]
        sorted_col = new_col[sort_idx]
        sorted_data = new_data[sort_idx]
        sorted_valid = valid_edges[sort_idx]
        
        # Static slice to physically shrink the matrix for XLA!
        sliced_row = sorted_row[:k_nse]
        sliced_col = sorted_col[:k_nse]
        sliced_data = sorted_data[:k_nse]
        sliced_valid = sorted_valid[:k_nse]
        
        # Any edges in the slice that were originally False (if k_nse > valid edges) 
        # must be zeroed out so they don't corrupt the graph.
        final_data = jnp.where(sliced_valid, sliced_data, 0.0)
        final_indices = jnp.stack([sliced_row, sliced_col], axis=-1)
        
        A_pool = sparse.BCOO((final_data, final_indices), shape=(k, k))
        
        # Gate the pooled features
        gate = jax.nn.sigmoid(y[topk_idx])
        gate = gate[:, None, None] # Expand to (k, 1, 1) for correct broadcasting
        X_pool = X[topk_idx] * gate
        
        return X_pool, A_pool, topk_idx

class GraphUnpool_JAX(nn.Module):
    @nn.compact
    def __call__(self, X_pool, topk_idx, target_shape):
        # Scatter the coarse features back to their original positions in the fine graph
        X_unpool = jnp.zeros(target_shape, dtype=X_pool.dtype)
        X_unpool = X_unpool.at[topk_idx].set(X_pool)
        return X_unpool

class GraphUNet_JAX(nn.Module):
    embed: int = 32
    K: int = 3
    dtype: jnp.dtype = jnp.float32
    
    @nn.compact
    def __call__(self, x, A: sparse.BCOO, train: bool = False):
        n, batch_size = x.shape
        
        # Dynamically scale input to avoid float32 precision floor / explosion
        std = jnp.std(x, axis=0, keepdims=True) + 1e-8
        x_scaled = x / std
        x_in = x_scaled.reshape(n, batch_size, 1)
        
        # Initial Embedding
        x_emb = nn.Dense(self.embed, dtype=self.dtype)(x_in)
        x_emb = nn.LayerNorm(dtype=self.dtype)(x_emb)
        x_emb = nn.relu(x_emb)
        
        # Level 1 (Fine)
        x1 = ChebConv_JAX(in_dim=self.embed, out_dim=self.embed, K=self.K, dtype=self.dtype)(x_emb, A)
        x1 = nn.LayerNorm(dtype=self.dtype)(x1)
        x1 = nn.relu(x1)
        x1_pool, A1, idx1 = GraphPool_JAX(pool_ratio=0.5, nse_ratio=0.5)(x1, A)
        
        # Level 2 (Mid)
        x2 = ChebConv_JAX(in_dim=self.embed, out_dim=self.embed, K=self.K, dtype=self.dtype)(x1_pool, A1)
        x2 = nn.LayerNorm(dtype=self.dtype)(x2)
        x2 = nn.relu(x2)
        x2_pool, A2, idx2 = GraphPool_JAX(pool_ratio=0.5, nse_ratio=0.5)(x2, A1)
        
        # Level 3 (Bottleneck)
        x3 = ChebConv_JAX(in_dim=self.embed, out_dim=self.embed, K=5, dtype=self.dtype)(x2_pool, A2)
        x3 = nn.LayerNorm(dtype=self.dtype)(x3)
        x3 = nn.relu(x3)
        
        # Decoder Unpool 2
        x3_unpool = GraphUnpool_JAX()(x3, idx2, x2.shape)
        x2_out = x2 + x3_unpool  # Skip connection
        x2_out = ChebConv_JAX(in_dim=self.embed, out_dim=self.embed, K=self.K, dtype=self.dtype)(x2_out, A1)
        x2_out = nn.LayerNorm(dtype=self.dtype)(x2_out)
        x2_out = nn.relu(x2_out)
        
        # Decoder Unpool 1
        x2_unpool = GraphUnpool_JAX()(x2_out, idx1, x1.shape)
        x1_out = x1 + x2_unpool  # Skip connection
        x1_out = ChebConv_JAX(in_dim=self.embed, out_dim=self.embed, K=self.K, dtype=self.dtype)(x1_out, A)
        x1_out = nn.LayerNorm(dtype=self.dtype)(x1_out)
        x1_out = nn.relu(x1_out)
        
        # Projection to scalar output (No LayerNorm or ReLU on final output!)
        out = nn.Dense(1, dtype=self.dtype)(x1_out)
        out = out.reshape(n, batch_size)
        
        # Restore true physical magnitude
        out = out * std.reshape(1, batch_size)
        return out
