import jax
import jax.numpy as jnp
import numpy as np

def jax_fgmres_solve(A_op, b, M_op=None, restart=150, max_iters=1000, tol=1e-6):
    """
    Mixed-precision Flexible GMRES implemented in pure JAX using jax.lax control flow.
    A_op and M_op should accept and return float32 vectors.
    Orthogonalization (V, Z, H) is performed in float32 to maintain precision.
    """
    n = b.shape[0]
    
    # We will use float32 for the subspace matrices to prevent stalling at 1e-5
    dtype_high = jnp.float32
    dtype_low = jnp.float32

    def fgmres_restart_cond(state):
        x, r, iters, converged = state
        return jnp.logical_and(iters < max_iters, jnp.logical_not(converged))

    def fgmres_restart_body(state):
        x, r, iters, converged = state
        
        # Cast residual to high precision for norm
        r_high = r.astype(dtype_high)
        beta = jnp.linalg.norm(r_high)
        norm_b = jnp.linalg.norm(b.astype(dtype_high))
        norm_b = jnp.where(norm_b == 0.0, 1.0, norm_b)
        
        # If converged already, just exit
        is_converged = (beta / norm_b) < tol
        
        # Preallocate arrays
        V = jnp.zeros((n, restart + 1), dtype=dtype_high)
        Z = jnp.zeros((n, restart), dtype=dtype_high)
        H = jnp.zeros((restart + 1, restart), dtype=dtype_high)
        g = jnp.zeros(restart + 1, dtype=dtype_high)
        c = jnp.zeros(restart, dtype=dtype_high)
        s = jnp.zeros(restart, dtype=dtype_high)
        
        # Safe divide
        v0 = jnp.where(beta == 0.0, r_high, r_high / beta)
        V = V.at[:, 0].set(v0)
        g = g.at[0].set(beta)
        
        # Inner Arnoldi loop state
        inner_state = (V, Z, H, g, c, s, iters, is_converged, 0)
        
        def inner_cond(i_state):
            _, _, _, _, _, _, inner_iters, inner_conv, j = i_state
            return jnp.logical_and(
                jnp.logical_and(j < restart, inner_iters < max_iters),
                jnp.logical_not(inner_conv)
            )
            
        def inner_body(i_state):
            V, Z, H, g, c, s, inner_iters, inner_conv, j = i_state
            
            v_j = V[:, j].astype(dtype_low) # Back to low precision for operators
            
            # Apply preconditioner
            if M_op is not None:
                z_j = M_op(v_j)
            else:
                z_j = v_j
                
            Z = Z.at[:, j].set(z_j.astype(dtype_high))
            
            # Apply matrix
            w = A_op(z_j).astype(dtype_high)
            
            # Modified Gram-Schmidt
            def gs_body(k, val):
                w_curr, H_curr = val
                H_kj = jnp.vdot(V[:, k], w_curr)
                H_curr = H_curr.at[k, j].set(H_kj)
                w_curr = w_curr - H_kj * V[:, k]
                return w_curr, H_curr
                
            w, H = jax.lax.fori_loop(0, j + 1, gs_body, (w, H))
            
            H_j1_j = jnp.linalg.norm(w)
            H = H.at[j + 1, j].set(H_j1_j)
            
            # Safe divide for V
            V = V.at[:, j + 1].set(jnp.where(H_j1_j == 0.0, w, w / H_j1_j))
            
            # Apply previous Givens rotations
            def givens_apply_body(k, H_curr):
                tmp = c[k] * H_curr[k, j] + s[k] * H_curr[k + 1, j]
                H_curr = H_curr.at[k + 1, j].set(-s[k] * H_curr[k, j] + c[k] * H_curr[k + 1, j])
                H_curr = H_curr.at[k, j].set(tmp)
                return H_curr
                
            H = jax.lax.fori_loop(0, j, givens_apply_body, H)
            
            # Compute new Givens rotation
            t = jnp.sqrt(H[j, j]**2 + H[j + 1, j]**2)
            c_new = jnp.where(t == 0.0, 1.0, H[j, j] / t)
            s_new = jnp.where(t == 0.0, 0.0, H[j + 1, j] / t)
            
            c = c.at[j].set(c_new)
            s = s.at[j].set(s_new)
            
            H = H.at[j, j].set(c[j] * H[j, j] + s[j] * H[j + 1, j])
            H = H.at[j + 1, j].set(0.0)
            
            g = g.at[j + 1].set(-s[j] * g[j])
            g = g.at[j].set(c[j] * g[j])
            
            inner_iters += 1
            inner_conv = (jnp.abs(g[j + 1]) / norm_b) < tol
            
            jax.debug.print("Inner Iter {j} | Residual: {r}", j=j, r=jnp.abs(g[j + 1]))
            return V, Z, H, g, c, s, inner_iters, inner_conv, j + 1

        # Run inner loop
        V, Z, H, g, c, s, iters, is_converged, num_inner_steps = jax.lax.while_loop(inner_cond, inner_body, inner_state)
        
        # Update solution if we actually did steps
        def update_x(vals):
            x, r, Z, H, g, num_inner_steps = vals
            # Mask out rows/cols beyond num_inner_steps
            idx = jnp.arange(restart)
            mask_2d = (idx[:, None] < num_inner_steps) & (idx[None, :] < num_inner_steps)
            mask_1d = idx < num_inner_steps
            
            # Make the un-used diagonal 1.0 to prevent singular matrix
            H_full = jnp.where(mask_2d, H[:restart, :restart], 0.0)
            H_full = H_full + jnp.diag(jnp.where(mask_1d, 0.0, 1.0))
            g_full = jnp.where(mask_1d, g[:restart], 0.0)
            
            y = jax.scipy.linalg.solve_triangular(H_full, g_full)
            y = jnp.where(mask_1d, y, 0.0)
            
            x_update = (Z @ y).astype(dtype_low)
            x = x + x_update
            r = b - A_op(x)
            return x, r
            
        def no_update(vals):
            x, r, Z, H, g, num_inner_steps = vals
            return x, r
            
        x, r = jax.lax.cond(num_inner_steps > 0, update_x, no_update, (x, r, Z, H, g, num_inner_steps))
        
        # If we hit max_iters or tolerance inside the inner loop, update converged
        converged = is_converged
        
        current_res_norm = jnp.linalg.norm(r.astype(dtype_high))
        jax.debug.print("FGMRES Step {i} | Residual: {r}", i=iters, r=current_res_norm)
        
        return x, r, iters, converged

    # Initial state
    x0 = jnp.zeros_like(b)
    r0 = b - A_op(x0)
    iters0 = 0
    converged0 = jnp.linalg.norm(r0.astype(dtype_high)) == 0.0

    final_state = jax.lax.while_loop(
        fgmres_restart_cond,
        fgmres_restart_body,
        (x0, r0, iters0, converged0)
    )
    
    x_final, r_final, iters_final, conv_final = final_state
    return x_final, iters_final
