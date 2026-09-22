"""One noiseless TCSI recovery on a synthetic Tucker tensor."""

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp

from tcsi.tcsi import (
    matrix_rank_of,
    observe,
    random_tucker,
    recover,
    relative_error,
)


def main():
    shape = (12, 12, 12)
    rank = (2, 2, 2)
    n = 1500
    key = jax.random.PRNGKey(0)
    k_tensor, k_obs, k_alg = jax.random.split(key, 3)
    T, _C, U, _V, _W = random_tucker(k_tensor, shape, rank)
    index_l, index_k, index_g, y = observe(k_obs, T, n, sigma=0.0, replace=False)
    indices = jnp.stack([index_l, index_k, index_g], axis=1)
    T_hat, _M_hat = recover(
        indices,
        y,
        U,
        shape,
        matrix_rank_of(rank),
        mu=10.0,
        lmax=40,
        key=k_alg,
    )
    err = relative_error(T_hat, T)
    print("relative error: {:.6e}".format(float(err)))
    return float(err)


if __name__ == "__main__":
    main()
