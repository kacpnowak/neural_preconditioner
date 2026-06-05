import jax
import jax.numpy as jnp
from jax.experimental import sparse
import flax.linen as nn
from ResGCN_JAX import ChebConv_JAX

class GraphUNet_JAX(nn.Module):
    embed: int = 32
    K: int = 3
    dtype: jnp.dtype = jnp.float32
    
    @nn.compact
    def __call__(self, x, hierarchy: dict, train: bool = False):
        """
        hierarchy: dict containing 'A' (list of BCOO matrices), 'P' (list of Prolongation), 'R' (list of Restriction)
        """
        n, batch_size = x.shape
        As = hierarchy['A']
        Ps = hierarchy['P']
        Rs = hierarchy['R']
        num_levels = len(As)
        
        # Dynamically scale input to avoid float32 precision floor / explosion
        std = jnp.std(x, axis=0, keepdims=True) + 1e-8
        x_scaled = x / std
        x_in = x_scaled.reshape(n, batch_size, 1)
        
        # Initial Embedding
        x_emb = nn.Dense(self.embed, dtype=self.dtype)(x_in)
        x_emb = nn.LayerNorm(dtype=self.dtype)(x_emb)
        x_emb = nn.relu(x_emb)
        
        # Encoder Pass
        x_levels = []
        x_current = x_emb
        
        for i in range(num_levels - 1):
            # Pre-smoothing (2 layers)
            x_tmp = ChebConv_JAX(in_dim=self.embed, out_dim=self.embed, K=self.K, dtype=self.dtype)(x_current, As[i])
            x_tmp = nn.LayerNorm(dtype=self.dtype)(x_tmp)
            x_tmp = nn.relu(x_tmp)
            
            x_tmp2 = ChebConv_JAX(in_dim=self.embed, out_dim=self.embed, K=self.K, dtype=self.dtype)(x_tmp, As[i])
            x_tmp2 = nn.LayerNorm(dtype=self.dtype)(x_tmp2)
            x_tmp2 = nn.relu(x_tmp2)
            
            x_current = x_current + x_tmp2 # Skip connection across smoother
            
            # Store for skip connection across U-Net
            x_levels.append(x_current)
            
            # Restrict to coarse level (R @ x)
            # R[i] shape: (coarse, fine), x_current shape: (fine, batch, embed)
            x_current = sparse.bcoo_dot_general(Rs[i], x_current, dimension_numbers=(((1,), (0,)), ((), ())))
            
        # Bottleneck Conv (Deepest Level)
        x_tmp = ChebConv_JAX(in_dim=self.embed, out_dim=self.embed, K=self.K, dtype=self.dtype)(x_current, As[-1])
        x_tmp = nn.LayerNorm(dtype=self.dtype)(x_tmp)
        x_tmp = nn.relu(x_tmp)
        
        x_tmp2 = ChebConv_JAX(in_dim=self.embed, out_dim=self.embed, K=self.K, dtype=self.dtype)(x_tmp, As[-1])
        x_tmp2 = nn.LayerNorm(dtype=self.dtype)(x_tmp2)
        x_tmp2 = nn.relu(x_tmp2)
        
        x_current = x_current + x_tmp2
        
        # Decoder Pass
        for i in reversed(range(num_levels - 1)):
            # Prolongate to fine level (P @ x)
            # P[i] shape: (fine, coarse), x_current shape: (coarse, batch, embed)
            x_current = sparse.bcoo_dot_general(Ps[i], x_current, dimension_numbers=(((1,), (0,)), ((), ())))
            
            # Skip connection
            x_current = x_current + x_levels[i]
            
            # Post-smoothing (2 layers)
            x_tmp = ChebConv_JAX(in_dim=self.embed, out_dim=self.embed, K=self.K, dtype=self.dtype)(x_current, As[i])
            x_tmp = nn.LayerNorm(dtype=self.dtype)(x_tmp)
            x_tmp = nn.relu(x_tmp)
            
            x_tmp2 = ChebConv_JAX(in_dim=self.embed, out_dim=self.embed, K=self.K, dtype=self.dtype)(x_tmp, As[i])
            x_tmp2 = nn.LayerNorm(dtype=self.dtype)(x_tmp2)
            x_tmp2 = nn.relu(x_tmp2)
            
            x_current = x_current + x_tmp2
            
        # Projection to scalar output (No LayerNorm or ReLU on final output!)
        out = nn.Dense(1, dtype=self.dtype)(x_current)
        out = out.reshape(n, batch_size)
        
        # Restore true physical magnitude
        out = out * std.reshape(1, batch_size)
        return out
