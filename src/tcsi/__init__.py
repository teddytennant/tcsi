"""Tensor Completion using Subspace Information (TCSI)."""

import jax

jax.config.update("jax_enable_x64", True)

from tcsi.tcsi import (
    build_reduced_matrix,
    hosvd,
    incoherence,
    observe,
    procrustes,
    reconstruct,
    recover,
    relative_error,
    rgrad,
    scaled_gradient,
    sensing_matrix,
    spectral_init,
    subspace_delta,
    tangent_project,
    trim_factor,
    trunc_rank,
)

__all__ = [
    "build_reduced_matrix",
    "hosvd",
    "incoherence",
    "observe",
    "procrustes",
    "reconstruct",
    "recover",
    "relative_error",
    "rgrad",
    "scaled_gradient",
    "sensing_matrix",
    "spectral_init",
    "subspace_delta",
    "tangent_project",
    "trim_factor",
    "trunc_rank",
]
