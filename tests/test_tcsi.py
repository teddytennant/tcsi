"""Tests for the TCSI observation identity and Algorithm 1 / Algorithm 2."""

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp

from tcsi.tcsi import (
    build_reduced_matrix,
    incoherence,
    least_squares,
    matrix_rank_of,
    observe,
    population_average,
    predict,
    project_rows,
    procrustes,
    random_tucker,
    reconstruct,
    recover,
    relative_error,
    scaled_gradient,
    sensing_matrix,
    side_subspace,
    spectral_init,
    subspace_delta,
    tangent_project,
    trim_factor,
    trunc_rank,
)


def _ball_radius(n_rows, r, mu, multiple=3.0):
    return (multiple * mu * r / n_rows) ** 0.5


def _problem(d=6, rank=(2, 2, 2), seed=0):
    key = jax.random.PRNGKey(seed)
    shape = (d, d, d)
    T, C, U, V, W = random_tucker(key, shape, rank)
    return T, C, U, V, W


def test_operator_identity_and_population_scale():
    d = 6
    rank = (2, 2, 2)
    T, C, U, V, W = _problem(d=d, rank=rank, seed=1)
    d1, d2, d3 = T.shape
    d_star = d1 * d2 * d3
    R = procrustes(U, U)
    assert jnp.linalg.norm(R - jnp.eye(rank[0])) < 1e-8

    M_star = build_reduced_matrix(C, V, W, R)
    # Every observed entry matches the sensing inner product.
    key = jax.random.PRNGKey(2)
    index_l, index_k, index_g, y = observe(key, T, n=40, sigma=0.0, replace=False)
    for i in range(y.shape[0]):
        Y = sensing_matrix(
            int(index_l[i]), int(index_k[i]), int(index_g[i]), U, d2, d3
        )
        inner = jnp.vdot(Y, M_star)
        assert abs(float(inner - y[i])) < 1e-8
        assert abs(float(predict(M_star, index_l[i : i + 1], index_k[i : i + 1], index_g[i : i + 1], U)[0] - y[i])) < 1e-8

    # Population identity: the scale that recovers M* is d* = d1 d2 d3.
    average = population_average(T, U)
    # Least-squares scale c * average ~= M*.
    c_hat = jnp.vdot(M_star, average) / jnp.vdot(average, average)
    assert abs(float(c_hat) - d_star) / d_star < 1e-8
    assert jnp.linalg.norm(d_star * average - M_star) / jnp.linalg.norm(M_star) < 1e-8

    # Same identity after an orthogonal change of the subspace basis.
    Q, _ = jnp.linalg.qr(jax.random.normal(jax.random.PRNGKey(3), (rank[0], rank[0]), dtype=jnp.float64))
    Us = U @ Q
    assert float(subspace_delta(Us, U)) < 1e-8
    R_rot = procrustes(Us, U)
    M_rot = build_reduced_matrix(C, V, W, R_rot)
    y_rot = predict(M_rot, index_l, index_k, index_g, Us)
    assert jnp.max(jnp.abs(y_rot - y)) < 1e-8
    assert jnp.linalg.norm(reconstruct(M_rot, Us) - T) / jnp.linalg.norm(T) < 1e-8
    assert jnp.linalg.norm(reconstruct(M_star, U) - T) / jnp.linalg.norm(T) < 1e-8


def test_incoherence_trim_trunc_tangent():
    N, r, mu = 12, 2, 1.0
    flat = jnp.stack(
        [
            jnp.ones((N,), dtype=jnp.float64),
            jnp.array([1.0 if i % 2 == 0 else -1.0 for i in range(N)], dtype=jnp.float64),
        ],
        axis=1,
    )
    flat = flat / jnp.linalg.norm(flat, axis=0, keepdims=True)
    # Equal row norms, column orthonormal, incoherence 1, inside the ball.
    assert abs(float(incoherence(flat)) - 1.0) < 1e-8
    assert float(jnp.linalg.norm(flat.T @ flat - jnp.eye(r))) < 1e-8
    clipped_flat, radius = project_rows(flat, mu)
    assert abs(float(radius) - _ball_radius(N, r, mu)) < 1e-12
    assert jnp.linalg.norm(clipped_flat - flat) < 1e-8
    assert jnp.linalg.norm(trim_factor(flat, mu) - flat) < 1e-8

    # One row far outside the ball, another already inside. The long row is
    # projected onto the sphere; the short row is unchanged.
    spiky = jnp.zeros((N, r), dtype=jnp.float64)
    spiky = spiky.at[0, 0].set(8.0)
    spiky = spiky.at[1, 1].set(0.1)
    assert float(incoherence(spiky)) > 3.0 * mu
    clipped, radius = project_rows(spiky, mu)
    assert abs(float(jnp.linalg.norm(clipped[0])) - float(radius)) < 1e-8
    assert float(jnp.linalg.norm(clipped[1] - spiky[1])) < 1e-8
    assert float(jnp.max(jnp.linalg.norm(clipped, axis=1))) <= float(radius) + 1e-8
    assert jnp.linalg.norm(clipped - spiky) > 1.0

    p = 7
    U = jax.random.orthogonal(jax.random.PRNGKey(4), N, dtype=jnp.float64)[:, :r]
    V = jax.random.orthogonal(jax.random.PRNGKey(5), p, dtype=jnp.float64)[:, :r]
    S = jnp.array([3.0, 1.0], dtype=jnp.float64)
    M = (U * S) @ V.T
    M1, _, s1, _ = trunc_rank(M, 1)
    M2, _, s2, _ = trunc_rank(M, 2)
    assert abs(float(s2[0]) - 3.0) < 1e-8
    assert abs(float(s2[1]) - 1.0) < 1e-8
    assert jnp.linalg.norm(M2 - M) / jnp.linalg.norm(M) < 1e-8
    assert jnp.linalg.norm(M1 - (U[:, :1] * S[0]) @ V[:, :1].T) / float(S[0]) < 1e-8
    assert s1.shape[0] == 1

    # Tangent projection is the identity on the tangent space and kills
    # the normal component.
    G_u = jax.random.normal(jax.random.PRNGKey(6), (N, r), dtype=jnp.float64)
    G_v = jax.random.normal(jax.random.PRNGKey(7), (p, r), dtype=jnp.float64)
    Z_tan = U @ G_v.T + G_u @ V.T
    assert jnp.linalg.norm(tangent_project(Z_tan, U, V) - Z_tan) < 1e-8
    G = jax.random.normal(jax.random.PRNGKey(8), (N, p), dtype=jnp.float64)
    Z_nor = (jnp.eye(N) - U @ U.T) @ G @ (jnp.eye(p) - V @ V.T)
    assert jnp.linalg.norm(tangent_project(Z_nor, U, V)) < 1e-8
    # And it is a projection.
    Z = jax.random.normal(jax.random.PRNGKey(9), (N, p), dtype=jnp.float64)
    PZ = tangent_project(Z, U, V)
    assert jnp.linalg.norm(tangent_project(PZ, U, V) - PZ) < 1e-8


def test_gradient_vanishes_at_truth():
    d = 8
    rank = (2, 2, 2)
    T, C, U, V, W = _problem(d=d, rank=rank, seed=11)
    d1, d2, d3 = T.shape
    d_star = d1 * d2 * d3
    M_star = build_reduced_matrix(C, V, W, jnp.eye(rank[0]))
    index_l, index_k, index_g, y = observe(
        jax.random.PRNGKey(12), T, n=400, sigma=0.0, replace=False
    )
    G = scaled_gradient(M_star, index_l, index_k, index_g, U, y, d_star)
    assert float(jnp.linalg.norm(G)) < 1e-8
    assert float(least_squares(M_star, index_l, index_k, index_g, U, y)) < 1e-16


def _init_error(seed, n, replace):
    d = 12
    rank = (2, 2, 2)
    T, C, U, V, W = _problem(d=d, rank=rank, seed=seed)
    d_star = d * d * d
    M_star = build_reduced_matrix(C, V, W, jnp.eye(rank[0]))
    index_l, index_k, index_g, y = observe(
        jax.random.PRNGKey(1000 + seed), T, n=n, sigma=0.0, replace=replace
    )
    M0 = spectral_init(
        index_l, index_k, index_g, y, U, d, d, matrix_rank_of(rank), d_star, mu=10.0
    )
    return float(relative_error(M0, M_star))


def test_spectral_init_close():
    # A complete census makes Algorithm 2 exact: (d*/n) sum y_i Y_i = M*.
    d = 12
    rank = (2, 2, 2)
    T, C, U, V, W = _problem(d=d, rank=rank, seed=20)
    d_star = d * d * d
    M_star = build_reduced_matrix(C, V, W, jnp.eye(rank[0]))
    flat = jnp.arange(d_star)
    index_l = flat // (d * d)
    rem = flat % (d * d)
    index_k = rem // d
    index_g = rem % d
    y = T[index_l, index_k, index_g]
    M0 = spectral_init(
        index_l, index_k, index_g, y, U, d, d, matrix_rank_of(rank), d_star, mu=10.0
    )
    assert float(relative_error(M0, M_star)) < 1e-8

    # n=2000 is the suggested CPU check. Each sensing matrix is one row, so
    # the relative RMSE of the spectral average scales as sqrt(d2*d3*r1/n)
    # which is about 0.38 here. The rank-r projection lands near 0.3, not
    # near zero. A missing d* factor would leave an error near 1.
    rel_2000 = _init_error(seed=0, n=2000, replace=True)
    assert rel_2000 < 0.45, rel_2000

    # n large enough that the same estimator is actually small.
    rel_large = _init_error(seed=0, n=200000, replace=True)
    assert rel_large < 0.05, rel_large
    assert rel_large < rel_2000


def test_recovery_exact_subspace():
    d = 12
    rank = (2, 2, 2)
    shape = (d, d, d)
    T, _C, U, _V, _W = _problem(d=d, rank=rank, seed=30)
    n = 1500
    index_l, index_k, index_g, y = observe(
        jax.random.PRNGKey(31), T, n=n, sigma=0.0, replace=False
    )
    indices = jnp.stack([index_l, index_k, index_g], axis=1)
    T_hat, _M = recover(
        indices,
        y,
        U,
        shape,
        matrix_rank_of(rank),
        mu=10.0,
        lmax=40,
        key=jax.random.PRNGKey(32),
    )
    rel = float(relative_error(T_hat, T))
    assert rel < 1e-2, rel


def test_side_information_helps():
    """Accurate Us beats mode-2 matrix completion below its sample threshold.

    The baseline is the same Riemannian loop on the full mode-2 unfolding
    (Us = I), which does not use the subspace. At n=800 the reduced problem
    is recovered and the unfolding is not.
    """
    d = 12
    rank = (2, 2, 2)
    shape = (d, d, d)
    n = 800
    T, _C, U, _V, _W = _problem(d=d, rank=rank, seed=0)
    index_l, index_k, index_g, y = observe(
        jax.random.PRNGKey(500), T, n=n, sigma=0.0, replace=False
    )
    indices = jnp.stack([index_l, index_k, index_g], axis=1)
    T_side, _ = recover(
        indices,
        y,
        U,
        shape,
        matrix_rank_of(rank),
        mu=10.0,
        lmax=40,
        key=jax.random.PRNGKey(1),
    )
    T_base, _ = recover(
        indices,
        y,
        jnp.eye(d, dtype=jnp.float64),
        shape,
        rank[1],
        mu=10.0,
        lmax=40,
        key=jax.random.PRNGKey(2),
    )
    err_side = float(relative_error(T_side, T))
    err_base = float(relative_error(T_base, T))
    assert err_side < 1e-2, err_side
    assert err_base > 0.15, err_base
    assert err_side < err_base


def test_error_grows_with_side_noise():
    """Section 5.1: with small observation noise, error grows with sigma_s."""
    d = 10
    rank = (2, 2, 2)
    shape = (d, d, d)
    n = 700
    n_s = 80
    T, _C, U, _V, _W = _problem(d=d, rank=rank, seed=50)
    index_l, index_k, index_g, y = observe(
        jax.random.PRNGKey(51), T, n=n, sigma=0.0, replace=False
    )
    indices = jnp.stack([index_l, index_k, index_g], axis=1)
    errors = []
    deltas = []
    for i, sigma_s in enumerate((0.0, 0.35, 0.9)):
        Us = side_subspace(jax.random.PRNGKey(100 + i), U, n_s, sigma_s)
        deltas.append(float(subspace_delta(Us, U)))
        T_hat, _ = recover(
            indices,
            y,
            Us,
            shape,
            matrix_rank_of(rank),
            mu=10.0,
            lmax=30,
            key=jax.random.PRNGKey(60 + i),
        )
        errors.append(float(relative_error(T_hat, T)))
    assert deltas[0] < 1e-6
    assert deltas[0] < deltas[1] < deltas[2]
    assert errors[0] < errors[1] < errors[2]
    assert errors[0] < 1e-2
