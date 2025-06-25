import jax
import jax.numpy as jnp
from jax.lax import fori_loop, cond, scan
from jax import vmap, jit
from scipy.sparse import csc_matrix, identity
from scipy.sparse.linalg import cg, gmres
import math
import numpy as np
import matplotlib.pyplot as plt
from functools import partial

def make_tri(nodnum: jnp.ndarray, nx: int, ny: int):
    def make_tri(n, nn):
        return jnp.array([nodnum[nn, n], nodnum[nn+1, n], nodnum[nn, n+1]]), jnp.array([nodnum[nn+1, n], nodnum[nn+1, n+1], nodnum[nn, n+1]])

    b = jnp.reshape(vmap(lambda i: jnp.tile(i, nx-1))(jnp.arange(ny-1)), [(nx-1)*(ny-1)])
    a = jnp.tile(jnp.arange(nx-1), nx-1)

    tri = vmap(make_tri)(b, a)
    return jnp.reshape(jnp.concatenate((tri[0], tri[1]), axis=1), [2*(nx-1)*(ny-1), 3])


def neighboring_triangles(n2d: int, e2d: int, tri: np.ndarray):
    # Initialize an array to store the count of neighboring triangles for each node.
    ne_num = np.zeros([n2d], dtype=int)

    # Loop through each triangle (element) in the mesh.
    for n in range(e2d):
        enodes = tri[n, :]
        # Increment the count of neighboring triangles for each node in the current triangle.
        ne_num[enodes] += 1

    # Initialize an array to store the positions of neighboring triangles for each node.
    ne_pos = np.zeros([int(np.max(ne_num)), n2d], dtype=int)

    # Reset the array to store the count of neighboring triangles for each node.
    ne_num = np.zeros([n2d], dtype=int)

    # Loop through each triangle (element) in the mesh.
    for n in range(e2d):
        enodes = tri[n, :]
        # Loop through the nodes of the current triangle.
        for j in range(3):
            # Store the position of the current neighboring triangle for the corresponding node.
            ne_pos[ne_num[enodes[j]], enodes[j]] = n
            # Increment the count of neighboring triangles for the node.
        ne_num[enodes] += 1

    return ne_num, ne_pos


def neighbouring_nodes(n2d: int, tri: np.ndarray, ne_num: np.ndarray, ne_pos: np.ndarray):
    nn_num = np.zeros([n2d], dtype=int)
    check = np.zeros([n2d], dtype=int)
    aux = np.zeros([10], dtype=int)
    for j in range(n2d):
        cc = 0
        for m in range(ne_num[j]):
            el = ne_pos[m, j]
            for k in range(3):
                a = tri[el, k]
                if check[a] == 0:
                    check[a] = 1
                    aux[cc] = a
                    cc += 1

        nn_num[j] = cc
        check[aux[0:cc]] = 0

    nn_pos = np.zeros([np.max(nn_num), n2d], dtype=int)

    for j in range(n2d):
        cc = 0
        for m in range(ne_num[j]):
            el = ne_pos[m, j]
            for k in range(3):
                a = tri[el, k]
                if check[a] == 0:
                    check[a] = 1
                    aux[cc] = a
                    cc += 1

        nn_pos[0:cc, j] = aux[0:cc].T
        check[aux[0:cc]] = 0

    return nn_num, nn_pos


def areas(n2d: int, e2d: int, tri: np.ndarray, xcoord: np.ndarray, ycoord: np.ndarray, ne_num: np.ndarray,
          ne_pos: np.ndarray, meshtype: str, carthesian: bool, cyclic_length):
    dx = np.zeros([e2d, 3], dtype=float)
    dy = np.zeros([e2d, 3], dtype=float)
    elem_area = np.zeros([e2d])
    r_earth = 6400  # Earth's radius, assuming units in kilometers
    Mt = np.ones([e2d])

    if meshtype == 'm':
        for n in range(e2d):
            # Calculate differences in x and y coordinates for triangle vertices.
            x2 = xcoord[tri[n, 1]] - xcoord[tri[n, 0]]
            x3 = xcoord[tri[n, 2]] - xcoord[tri[n, 0]]
            y2 = ycoord[tri[n, 1]] - ycoord[tri[n, 0]]
            y3 = ycoord[tri[n, 2]] - ycoord[tri[n, 0]]

            # Calculate determinant of the Jacobian matrix for this triangle.
            d = x2 * y3 - y2 * x3

            # Calculate x and y derivatives of P1 basis functions.
            dx[n, 0] = (-y3 + y2) / d
            dx[n, 1] = y3 / d
            dx[n, 2] = -y2 / d

            dy[n, 0] = -(-x3 + x2) / d
            dy[n, 1] = -x3 / d
            dy[n, 2] = x2 / d

            # Calculate the area of the triangle.
            elem_area[n] = 0.5 * abs(d)

    elif meshtype == 'r':
        rad = math.pi / 180.0
        if carthesian:
            Mt = np.ones([e2d])
        else:
            Mt = np.cos(np.sum(rad * ycoord[tri], axis=1) / 3.0)

        for n in range(e2d):
            # Calculate differences in longitude and latitude for triangle vertices.
            x2 = rad * (xcoord[tri[n, 1]] - xcoord[tri[n, 0]])
            x3 = rad * (xcoord[tri[n, 2]] - xcoord[tri[n, 0]])
            y2 = r_earth * rad * (ycoord[tri[n, 1]] - ycoord[tri[n, 0]])
            y3 = r_earth * rad * (ycoord[tri[n, 2]] - ycoord[tri[n, 0]])

            # Adjust for cyclic boundaries.
            if x2 > cyclic_length / 2.0:
                x2 = x2 - cyclic_length
            if x2 < -cyclic_length / 2.0:
                x2 = x2 + cyclic_length
            if x3 > cyclic_length / 2.0:
                x3 = x3 - cyclic_length
            if x3 < -cyclic_length / 2.0:
                x3 = x3 + cyclic_length

            # Apply metric factors and calculate x and y derivatives of P1 basis functions.
            x2 = r_earth * x2 * Mt[n]
            x3 = r_earth * x3 * Mt[n]
            d = x2 * y3 - y2 * x3

            dx[n, 0] = (-y3 + y2) / d
            dx[n, 1] = y3 / d
            dx[n, 2] = -y2 / d

            dy[n, 0] = -(-x3 + x2) / d
            dy[n, 1] = -x3 / d
            dy[n, 2] = x2 / d

            # Calculate the area of the triangle.
            elem_area[n] = 0.5 * abs(d)

        if carthesian:
            Mt = np.zeros([e2d])
        else:
            Mt = (np.sin(rad * np.sum(ycoord[tri], axis=1) / 3.0) / Mt) / r_earth

    # Calculate scalar cell (cluster) area for each node.
    area = np.zeros([n2d])
    for n in range(n2d):
        area[n] = np.sum(elem_area[ne_pos[0:ne_num[n], n]]) / 3.0

    return area, elem_area, dx, dy, Mt


def make_smooth(Mt: jnp.ndarray, elem_area: jnp.ndarray, dx: jnp.ndarray, dy: jnp.ndarray, nn_num: jnp.ndarray,
                nn_pos: jnp.ndarray, tri: jnp.ndarray, n2d: int, e2d: int, full: bool = False):

    smooth_m = jnp.zeros(nn_pos.shape, dtype=jnp.float32)
    metric = jnp.zeros(nn_pos.shape, dtype=jnp.float32)
    aux = jnp.zeros((n2d,), dtype=jnp.int32)

    @jit
    def loop_body(j, carry):
        smooth_m, metric, aux, nn_num, nn_pos, elem_area, dx, dy, Mt = carry
        enodes = tri[j, :]

        def inner_loop_body(n, carry):
            smooth_m, metric, aux, enodes, nn_num, nn_pos, elem_area, dx, dy, Mt = carry
            row = enodes[n]
            cc = nn_num[row]

            def fill_xd(i, val):
                row, aux, nn_pos = val
                n = nn_pos[i, row]
                aux = aux.at[n].set(i)
                return row, aux, nn_pos

            row, aux, _ = fori_loop(0, cc, fill_xd, (row, aux, nn_pos))

            def update_smooth_m(m, carry):
                smooth_m, metric, aux, enodes, elem_area, dx, dy, n = carry
                col = enodes[m]
                pos = aux[col]
                tmp_x = dx[m] * dx[n]
                tmp_y = dy[n] * dy[m]
                c1 = m == n

                smooth_m = smooth_m.at[pos, row].add(cond(c1 & full,
                                                          lambda: (tmp_x + tmp_y) * elem_area + jnp.square(
                                                              Mt) * elem_area / 3.0,
                                                          lambda: (tmp_x + tmp_y) * elem_area
                                                          )
                                                     )
                metric = metric.at[pos, row].add(Mt * (dx[n] - dx[m]) * elem_area / 3.0)
                return smooth_m, metric, aux, enodes, elem_area, dx, dy, n

            smooth_m, metric, aux, _, _, _, _, _ = fori_loop(0, 3, update_smooth_m,
                                                             (smooth_m, metric, aux, enodes, elem_area, dx, dy, n))
            return smooth_m, metric, aux, enodes, nn_num, nn_pos, elem_area, dx, dy, Mt

        smooth_m, metric, aux, _, _, _, _, _, _, _ = fori_loop(0, 3, inner_loop_body, (
        smooth_m, metric, aux, enodes, nn_num, nn_pos, elem_area[j], dx[j, :], dy[j, :], Mt[j]))
        return smooth_m, metric, aux, nn_num, nn_pos, elem_area, dx, dy, Mt

    smooth_m, metric, _, _, _, _, _, _, _ = fori_loop(0, e2d, loop_body,
                                                      (smooth_m, metric, aux, nn_num, nn_pos, elem_area, dx, dy, Mt))
    return smooth_m, metric


@partial(jit, static_argnums=[3, 4])
def make_smat(nn_pos: jnp.ndarray, nn_num: jnp.ndarray, smooth_m: jnp.ndarray, n2d: int, nza: int):
    def helper(carry, x):
        n, m = carry
        out = (smooth_m[m, n], n, nn_pos[m, n])
        n, m = cond(m + 1 >= nn_num[n], lambda: cond(n + 1 >= n2d, lambda: (0, 0), lambda: (n + 1, 0)),
                    lambda: (n, m + 1))
        return (n, m), out

    _, tmp = scan(helper, init=(0, 0), xs=jnp.arange(nza))
    ss, ii, jj = tmp

    return ss, ii, jj


def create_synthetic_data(Lx, dxm):
    
    r_earth = 6400.0  # in km; wavenumbers will be in 1/km
    Ly = Lx
    dym = dxm
    cyclic = 0  # 1 if mesh is cyclic
    cyclic_length = 360  # in degrees; if not cyclic, take it larger than  zonal size
    cyclic_length = cyclic_length * math.pi / 180  # DO NOT TOUCH
    meshtype = 'm'  # coordinates can be in physical measure 'm' or in radians 'r'
    
    xx = np.arange(0, Lx + 1, dxm, dtype="float32")
    yy = np.arange(0, Ly + 1, dym, dtype="float32")
    
    # Create uniformity in the mesh
    # xx = np.concatenate((np.arange(0, 4*Lx//20 + 1, dxm), np.arange(5*Lx//20, Lx + 1, dxm)))
    # yy = np.concatenate((np.arange(0, 4*Ly//20 + 1, dym), np.arange(5*Ly//20, Ly + 1, dym)))
    
    nx = len(xx)
    ny = len(yy)
    
    nodnum = np.arange(0, nx * ny)
    xcoord = np.tile(xx, reps=(ny, 1)).T
    ycoord = np.tile(yy, reps=(nx, 1))
    @jit
    def make_ll(x, y):
        return jnp.sqrt(y * y + x * x)
    
    @jit
    def make_tff(ttf, ll):
        return ttf / (jnp.power(ll, 1.5))  # 1.5 for -2 spectrum
    
    tt = 50 * (np.random.random(xcoord.shape) - 0.5)
    ttf = np.fft.fft2(tt)
    # ============
    # Make spectrum red
    # ============
    espectrum = np.zeros((nx // 2 + 1))  # Place for Fourier spectrum
    kk = np.concatenate((np.arange(0, nx // 2 + 1), np.arange(-nx // 2 + 1, 0, 1)))  # Wavenumbers
    
    ll = vmap(vmap(make_ll, in_axes=(None, 0)), in_axes=(0, None))(kk, kk)
    ttf = vmap(vmap(make_tff, in_axes=(0, 0)), in_axes=(0, 0))(ttf, ll)
    ttf = ttf.at[0, 0].set(0.0)
    
    tt = jnp.real(jnp.fft.ifft2(ttf))
    return jnp.reshape(tt, [nx*ny])


def create_synthetic_matrix(Lx, dxm, cartesian):
    def make_tri(nodnum: jnp.ndarray, nx: int, ny: int):
        def make_tri(n, nn):
            return jnp.array([nodnum[nn, n], nodnum[nn+1, n], nodnum[nn, n+1]]), jnp.array([nodnum[nn+1, n], nodnum[nn+1, n+1], nodnum[nn, n+1]])
    
        b = jnp.reshape(vmap(lambda i: jnp.tile(i, nx-1))(jnp.arange(ny-1)), [(nx-1)*(ny-1)])
        a = jnp.tile(jnp.arange(nx-1), nx-1)
    
        tri = vmap(make_tri)(b, a)
        return jnp.reshape(jnp.concatenate((tri[0], tri[1]), axis=1), [2*(nx-1)*(ny-1), 3])
    
    r_earth = 6400.0  # in km; wavenumbers will be in 1/km
    Ly = Lx
    dym = dxm
    cyclic = 0  # 1 if mesh is cyclic
    cyclic_length = 360  # in degrees; if not cyclic, take it larger than  zonal size
    cyclic_length = cyclic_length * math.pi / 180  # DO NOT TOUCH
    meshtype = 'm'  # coordinates can be in physical measure 'm' or in radians 'r'
    
    xx = np.arange(0, Lx + 1, dxm, dtype="float32")
    yy = np.arange(0, Ly + 1, dym, dtype="float32")
    
    # Create uniformity in the mesh
    # xx = np.concatenate((np.arange(0, 4*Lx//20 + 1, dxm), np.arange(5*Lx//20, Lx + 1, dxm)))
    # yy = np.concatenate((np.arange(0, 4*Ly//20 + 1, dym), np.arange(5*Ly//20, Ly + 1, dym)))
    
    nx = len(xx)
    ny = len(yy)
    
    nodnum = np.arange(0, nx * ny)
    xcoord = np.tile(xx, reps=(ny, 1)).T
    ycoord = np.tile(yy, reps=(nx, 1))

    # == == == == == == == =
    # Reshape to 1D arrays
    # == == == == == == == =
    nodnum = np.reshape(nodnum, [ny, nx]).T
    xcoord = np.reshape(xcoord, [nx * ny])
    ycoord = np.reshape(ycoord, [nx * ny])
    
    xcoord /= r_earth
    ycoord /= r_earth
    alpha = math.pi/3 + math.pi/12
    zg = np.sin(ycoord)
    xg = np.cos(ycoord) * np.cos(xcoord)
    yg = np.cos(ycoord) * np.sin(xcoord)
    # Rotate by alpha
    zn = zg * np.cos(alpha) + xg * np.sin(alpha)
    xg = -zg * np.sin(alpha) + xg * np.cos(alpha)
    # New coordinates in radians
    ycoord = np.arcsin(zn)
    xcoord = np.arctan2(yg,xg)
    # New coordinates in degrees
    ycoord = (180/math.pi) * np.arcsin(zn)
    xcoord = (180/math.pi) * np.arctan2(yg,xg)
    
    make_tri = jit(make_tri, static_argnums=[1, 2])
    
    tri = make_tri(nodnum, nx, ny)
    tri = np.array(tri)
    n2d = len(xcoord)  # The number of vertices(nodes)
    e2d = len(tri[:, 1])  # The number of triangles(elements)
    ne_num, ne_pos = neighboring_triangles(n2d, e2d, tri)
    nn_num, nn_pos = neighbouring_nodes(n2d, tri, ne_num, ne_pos)
    area, elem_area, dx, dy, Mt = areas(n2d, e2d, tri, xcoord, ycoord, ne_num, ne_pos, meshtype, cartesian, cyclic_length)
    
    # Move data to JAX
    jelem_area = jnp.array(elem_area)
    jdx = jnp.array(dx)
    jdy = jnp.array(dy)
    jnn_num = jnp.array(nn_num)
    jnn_pos = jnp.array(nn_pos)
    jtri = jnp.array(tri)
    jarea = jnp.array(area)
    jMt = jnp.array(Mt)
    
    with jax.default_device(jax.devices("cpu")[0]): # Force JAX to use CPU
        jsmooth, metric = make_smooth(jMt, jelem_area, jdx, jdy, jnn_num, jnn_pos, jtri, n2d, e2d, False)
        ss, ii, jj = make_smat(jnn_pos, jnn_num, jsmooth, n2d, int(jnp.sum(jnn_num)))
    return ss, ii, jj, tri, xcoord, ycoord