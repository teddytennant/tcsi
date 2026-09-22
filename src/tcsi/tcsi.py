"""TCSI: Tucker tensor completion from a mode-1 subspace estimate.

The reduced unknown is the matrix
    M* = V* C2* (W*^T kron R) in R^{d2 x (d3 r1)},
where C2* is the mode-2 unfolding of the core and R is the orthogonal
Procrustes factor aligning Us with U*. When Us = U*, R = I.

Unfolding convention (matches the proof identity
Yi = e_k (e_g kron e_l)^T (I_{d3} kron Us)):
the mode-2 column multi-index is (g, alpha) with alpha fastest.
Thus [M]_{k, g*r1 + alpha} stores the reduced fiber, and
    Yi = e_k (e_g kron Us[l, :])^T.
With this order, d* = d1*d2*d3 is the unique scale such that
d* E[y Y] = M* when the subspace is exact and the noise is zero.

The PDF step-size line is broken across the trimming threshold. On the
scaled gradient G = (d*/n) sum_i (<M, Yi> - y_i) Yi the population map is
the identity, so the paper's eta = d*/n on the unscaled Euclidean gradient
is a unit step. We take eta = c / op_norm of that scaled Gram operator and
backtrack so the least-squares loss does not increase.
"""

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp


def _f64(x):
    return jnp.asarray(x, dtype=jnp.float64)


def incoherence(U):
    """Incoherence of an N x r matrix: (N/r) * ||U||_{2,inf}^2."""
    U = _f64(U)
    n_rows, r = U.shape
    row_sq = jnp.sum(U * U, axis=1)
    return (n_rows / r) * jnp.max(row_sq)


def project_rows(U, mu, multiple=3.0):
    """Project each row onto the Euclidean ball of radius sqrt(mu0 * r / N).

    mu0 = multiple * mu. The ball is exactly the set of rows with squared
    norm at most mu0 * r / N, which is the incoherence constraint
    Incoh(U) <= mu0. Algorithm 3 writes the same radius as
    sqrt(mu0 * r / d). A row already inside the ball is unchanged.
    """
    U = _f64(U)
    n_rows, r = U.shape
    radius = jnp.sqrt(multiple * mu * r / n_rows)
    row_norm = jnp.linalg.norm(U, axis=1, keepdims=True)
    scale = jnp.minimum(1.0, radius / jnp.maximum(row_norm, 1e-30))
    return U * scale, radius


def trim_factor(U, mu, multiple=3.0):
    """Algorithm 3: row-clip, then re-orthogonalize by the polar factor.

    A column-orthonormal factor that already lies in the ball is unchanged.
    Re-orthogonalization undoes a pure global scale, so a factor whose
    nonzero rows all have the same length is unchanged as well. Unequal
    rows that stick out of the ball are clipped before that step.
    """
    U0, _radius = project_rows(U, mu, multiple=multiple)
    left, _sing, right_h = jnp.linalg.svd(U0, full_matrices=False)
    return left @ right_h


def trunc_rank(M, rank):
    """Best rank-r approximation in Frobenius norm, and its compact SVD."""
    M = _f64(M)
    left, sing, right_h = jnp.linalg.svd(M, full_matrices=False)
    r = min(rank, sing.shape[0])
    left_r = left[:, :r]
    sing_r = sing[:r]
    right_r = right_h[:r, :].T
    approx = (left_r * sing_r) @ right_r.T
    return approx, left_r, sing_r, right_r


def trim_matrix(M, rank, mu, multiple=3.0, trim_right=False):
    """Rank-r truncation, then row-clip of the left singular factor.

    If the left factor's incoherence exceeds `multiple * mu`, its rows are
    projected onto the ball of squared radius `multiple * mu * r / N`.
    The clip is a no-op when every row is already inside the ball, so the
    incoherence check and the projection are the same operation.
    """
    approx, left, sing, right = trunc_rank(M, rank)
    left_t = trim_factor(left, mu, multiple=multiple)
    if trim_right:
        right_t = trim_factor(right, mu, multiple=multiple)
    else:
        right_t = right
    return (left_t * sing) @ right_t.T


def tangent_project(Z, left, right):
    """Project Z onto the tangent space of rank-r matrices at left diag right^T.

    PT(Z) = UU^T Z + Z VV^T - UU^T Z VV^T.
    """
    Z = _f64(Z)
    left = _f64(left)
    right = _f64(right)
    left_part = left @ (left.T @ Z)
    right_part = (Z @ right) @ right.T
    both = left @ ((left.T @ Z) @ right) @ right.T
    return left_part + right_part - both


def procrustes(Us, U):
    """R = argmin_{R orthogonal} ||Us - U R||_F.

    If U^T Us = O1 Sigma O2^T, then R = O1 O2^T. Equivalently, if
    Us^T U = O1 Sigma O2^T as in the paper, R = O2 O1^T.
    """
    Us = _f64(Us)
    U = _f64(U)
    o1, _, o2h = jnp.linalg.svd(U.T @ Us, full_matrices=False)
    return o1 @ o2h


def mode2_core(C):
    """Mode-2 unfolding with column multi-index (c, a), a fastest."""
    C = _f64(C)
    r1, r2, r3 = C.shape
    return jnp.transpose(C, (1, 2, 0)).reshape(r2, r3 * r1)


def build_reduced_matrix(C, V, W, R):
    """M* = V C2 (W^T kron R), shape (d2, d3 * r1)."""
    C2 = mode2_core(C)
    return _f64(V) @ C2 @ jnp.kron(_f64(W).T, _f64(R))


def sensing_matrix(index_l, index_k, index_g, Us, d2, d3):
    """One sensing matrix Yi in R^{d2 x (d3 r1)}.

    Only row k is nonzero, and that row is e_g kron Us[l, :].
    """
    Us = _f64(Us)
    r1 = Us.shape[1]
    Y = jnp.zeros((d2, d3 * r1), dtype=jnp.float64)
    start = int(index_g) * r1
    Y = Y.at[int(index_k), start : start + r1].set(Us[int(index_l)])
    return Y


def predict(M, index_l, index_k, index_g, Us):
    """Vector of inner products <Yi, M>."""
    M = _f64(M)
    Us = _f64(Us)
    r1 = Us.shape[1]
    d3 = M.shape[1] // r1
    blocks = M.reshape(M.shape[0], d3, r1)
    gathered = blocks[index_k, index_g, :]
    return jnp.sum(gathered * Us[index_l], axis=-1)


def scaled_gradient(M, index_l, index_k, index_g, Us, y, d_star):
    """G = (d*/n) sum_i (<M, Yi> - y_i) Yi."""
    M = _f64(M)
    Us = _f64(Us)
    y = _f64(y)
    n = y.shape[0]
    r1 = Us.shape[1]
    d2 = M.shape[0]
    d3 = M.shape[1] // r1
    residual = predict(M, index_l, index_k, index_g, Us) - y
    contrib = residual[:, None] * Us[index_l]
    acc = jnp.zeros((d2, d3, r1), dtype=jnp.float64)
    acc = acc.at[index_k, index_g].add(contrib)
    return (d_star / n) * acc.reshape(d2, d3 * r1)


def least_squares(M, index_l, index_k, index_g, Us, y):
    """(1/2) ||Y(M) - y||^2, the objective in equation (4)."""
    residual = predict(M, index_l, index_k, index_g, Us) - _f64(y)
    return 0.5 * jnp.dot(residual, residual)


def population_average(T, Us):
    """E[y Y] under uniform sampling and zero noise, exact subspace not required.

    Contracts the observed tensor against Us on mode 1. Equals M*/d* when
    the subspace is exact.
    """
    T = _f64(T)
    Us = _f64(Us)
    d1, d2, d3 = T.shape
    d_star = d1 * d2 * d3
    # contracted[alpha, k, g] = sum_l T[l, k, g] Us[l, alpha]
    contracted = jnp.einsum("lkg,la->akg", T, Us)
    arranged = jnp.transpose(contracted, (1, 2, 0))
    return arranged.reshape(d2, d3 * Us.shape[1]) / d_star


def hosvd(T, rank):
    """Standardized HOSVD factors. Left singular vectors do not depend on
    the column order of the unfolding, so any consistent reshape is valid.
    """
    T = _f64(T)
    r1, r2, r3 = rank
    d1, d2, d3 = T.shape
    U = jnp.linalg.svd(T.reshape(d1, d2 * d3), full_matrices=False)[0][:, :r1]
    V = jnp.linalg.svd(
        jnp.transpose(T, (1, 0, 2)).reshape(d2, d1 * d3), full_matrices=False
    )[0][:, :r2]
    W = jnp.linalg.svd(
        jnp.transpose(T, (2, 0, 1)).reshape(d3, d1 * d2), full_matrices=False
    )[0][:, :r3]
    C = jnp.einsum("ijk,ia,jb,kc->abc", T, U, V, W)
    T_hat = jnp.einsum("abc,ia,jb,kc->ijk", C, U, V, W)
    return T_hat, C, U, V, W


def random_tucker(key, shape, rank):
    """Draw a Tucker tensor as in section 5.1 and standardize by HOSVD.

    Factors and core have i.i.d. Unif([0, 1]) entries. The returned tensor
    is the HOSVD reconstruction, which matches the draw when the multilinear
    rank is exactly `rank`.
    """
    d1, d2, d3 = shape
    r1, r2, r3 = rank
    kU, kV, kW, kC = jax.random.split(key, 4)
    U = jax.random.uniform(kU, (d1, r1), dtype=jnp.float64)
    V = jax.random.uniform(kV, (d2, r2), dtype=jnp.float64)
    W = jax.random.uniform(kW, (d3, r3), dtype=jnp.float64)
    C = jax.random.uniform(kC, (r1, r2, r3), dtype=jnp.float64)
    T = jnp.einsum("abc,ia,jb,kc->ijk", C, U, V, W)
    return hosvd(T, rank)


def side_subspace(key, U, n_s, sigma_s):
    """Us = top-r1 left singular vectors of Ms = U C_coef + Es."""
    U = _f64(U)
    d1, r1 = U.shape
    kC, kE = jax.random.split(key)
    coef = jax.random.normal(kC, (r1, n_s), dtype=jnp.float64)
    noise = sigma_s * jax.random.normal(kE, (d1, n_s), dtype=jnp.float64)
    Ms = U @ coef + noise
    Us = jnp.linalg.svd(Ms, full_matrices=False)[0][:, :r1]
    return Us


def subspace_delta(Us, U):
    """delta = ||Us Us^T - U U^T||_F."""
    Us = _f64(Us)
    U = _f64(U)
    return jnp.linalg.norm(Us @ Us.T - U @ U.T, ord="fro")


def observe(key, T, n, sigma=0.0, replace=False):
    """n uniform random entries of T, plus optional Gaussian noise of std sigma.

    Sampling without replacement is a random subset. Sampling with
    replacement matches the i.i.d. model in equation (1) and is required
    when n exceeds the number of entries.
    """
    T = _f64(T)
    d1, d2, d3 = T.shape
    d_star = d1 * d2 * d3
    if (not replace) and n > d_star:
        raise ValueError("n exceeds the number of entries; pass replace=True")
    k_idx, k_noise = jax.random.split(key)
    flat = jax.random.choice(
        k_idx, d_star, shape=(n,), replace=replace
    )
    index_l = flat // (d2 * d3)
    rem = flat % (d2 * d3)
    index_k = rem // d3
    index_g = rem % d3
    y = T[index_l, index_k, index_g]
    if sigma > 0.0:
        y = y + sigma * jax.random.normal(k_noise, (n,), dtype=jnp.float64)
    return index_l, index_k, index_g, y


def reconstruct(M, Us):
    """Inverse mode-2 reduction: reshape M to R^{r1 x d2 x d3} and multiply Us.

    This is T_hat = M_2^{-1}(M) x1 Us. It inverts M* = V C2 (W^T kron R)
    when Us is the subspace used to build the sensing matrices.
    """
    M = _f64(M)
    Us = _f64(Us)
    d2, col = M.shape
    r1 = Us.shape[1]
    d3 = col // r1
    arranged = M.reshape(d2, d3, r1)
    core = jnp.transpose(arranged, (2, 0, 1))
    return jnp.einsum("la,akg->lkg", Us, core)


def _spectral_average(index_l, index_k, index_g, y, Us, d2, d3, d_star):
    """(d*/n) sum_i y_i Yi, the matrix whose rank-r SVD is Algorithm 2."""
    Us = _f64(Us)
    y = _f64(y)
    n = y.shape[0]
    r1 = Us.shape[1]
    contrib = y[:, None] * Us[index_l]
    acc = jnp.zeros((d2, d3, r1), dtype=jnp.float64)
    acc = acc.at[index_k, index_g].add(contrib)
    return (d_star / n) * acc.reshape(d2, d3 * r1)


def spectral_init(index_l, index_k, index_g, y, Us, d2, d3, rank, d_star, mu):
    """Algorithm 2: rank-r SVD of (d*/n) sum y_i Yi, then factor trimming.

    Both singular-vector factors are row-clipped, then
    M0 = L0 L0^T M_tilde R0 R0^T as written in the paper.
    """
    average = _spectral_average(
        index_l, index_k, index_g, y, Us, d2, d3, d_star
    )
    approx, left, _, right = trunc_rank(average, rank)
    left0 = trim_factor(left, mu)
    right0 = trim_factor(right, mu)
    return left0 @ (left0.T @ approx @ right0) @ right0.T


def operator_norm(index_l, index_k, index_g, Us, d2, d3, r1, d_star, n_iter=12, key=None):
    """Power iteration for the operator norm of Z |-> (d*/n) sum <Z, Yi> Yi."""
    if key is None:
        key = jax.random.PRNGKey(0)
    Z = jax.random.normal(key, (d2, d3 * r1), dtype=jnp.float64)
    Z = Z / jnp.linalg.norm(Z)
    zeros = jnp.zeros((index_l.shape[0],), dtype=jnp.float64)
    nrm = jnp.asarray(1.0, dtype=jnp.float64)
    for _ in range(n_iter):
        AZ = scaled_gradient(Z, index_l, index_k, index_g, Us, zeros, d_star)
        nrm = jnp.linalg.norm(AZ)
        Z = AZ / jnp.maximum(nrm, 1e-30)
    return nrm


def rgrad(
    M0,
    index_l,
    index_k,
    index_g,
    y,
    Us,
    rank,
    d_star,
    mu,
    lmax=40,
    c_step=1.0,
    key=None,
):
    """Algorithm 1: M <- Trim(Trunc_r(M - eta * PT(G))).

    G is the scaled gradient (d*/n) sum (<M, Yi> - y_i) Yi.
    eta starts at c_step / op_norm(scaled sensing Gram) and is halved until
    the least-squares loss does not increase. Stop on a loss plateau or
    after lmax iterations.
    """
    M = _f64(M0)
    Us = _f64(Us)
    d2 = M.shape[0]
    r1 = Us.shape[1]
    d3 = M.shape[1] // r1
    lip = operator_norm(
        index_l,
        index_k,
        index_g,
        Us,
        d2,
        d3,
        r1,
        d_star,
        key=key,
    )
    eta0 = c_step / jnp.maximum(lip, 1e-8)
    prev = least_squares(M, index_l, index_k, index_g, Us, y)
    for _ in range(lmax):
        _, left, _, right = trunc_rank(M, rank)
        grad = scaled_gradient(M, index_l, index_k, index_g, Us, y, d_star)
        direction = tangent_project(grad, left, right)
        eta = eta0
        accepted = None
        accepted_loss = prev
        for _bt in range(24):
            stepped = M - eta * direction
            trial = trim_matrix(stepped, rank, mu, trim_right=False)
            loss = least_squares(trial, index_l, index_k, index_g, Us, y)
            if loss <= prev:
                accepted = trial
                accepted_loss = loss
                break
            eta = eta * 0.5
        if accepted is None:
            break
        decrease = prev - accepted_loss
        M = accepted
        if decrease <= 1e-12 * jnp.maximum(prev, 1.0) or accepted_loss <= 1e-18:
            break
        prev = accepted_loss
    return M


def recover(
    indices,
    y,
    Us,
    shape,
    matrix_rank,
    mu=10.0,
    lmax=40,
    c_step=1.0,
    key=None,
):
    """Run spectral initialization and Riemannian GD, then rebuild the tensor.

    `indices` is an (n, 3) array of (l, k, g). `matrix_rank` is
    min(r2, r1 * r3) for a Tucker model, or r2 for plain mode-2
    matrix completion (pass Us = I_{d1} in that case).
    """
    indices = jnp.asarray(indices)
    index_l = indices[:, 0]
    index_k = indices[:, 1]
    index_g = indices[:, 2]
    Us = _f64(Us)
    y = _f64(y)
    d1, d2, d3 = shape
    d_star = int(d1) * int(d2) * int(d3)
    M0 = spectral_init(
        index_l, index_k, index_g, y, Us, d2, d3, matrix_rank, d_star, mu
    )
    M_hat = rgrad(
        M0,
        index_l,
        index_k,
        index_g,
        y,
        Us,
        matrix_rank,
        d_star,
        mu,
        lmax=lmax,
        c_step=c_step,
        key=key,
    )
    return reconstruct(M_hat, Us), M_hat


def relative_error(A, B):
    """||A - B||_F / ||B||_F."""
    A = _f64(A)
    B = _f64(B)
    return jnp.linalg.norm(A - B) / jnp.linalg.norm(B)


def matrix_rank_of(tucker_rank):
    """Reduced matrix rank min(r2, r1 * r3)."""
    r1, r2, r3 = tucker_rank
    return min(r2, r1 * r3)
