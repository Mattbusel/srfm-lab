"""Extension for kernel_fusion.py — appended programmatically."""


# ---------------------------------------------------------------------------
# Section: Additional fused TT operations
# ---------------------------------------------------------------------------

import numpy as np
import warnings


def fused_tt_outer_product(
    cores_a: list,
    cores_b: list,
) -> list:
    """
    Compute the TT outer product of two TT-tensors.

    The result represents the elementwise outer product, with bond
    dimensions equal to the products of the input bond dimensions.

    Parameters
    ----------
    cores_a : list of np.ndarray
    cores_b : list of np.ndarray

    Returns
    -------
    cores_out : list of np.ndarray
    """
    assert len(cores_a) == len(cores_b), "Both TTs must have same depth."
    cores_out = []
    for ca, cb in zip(cores_a, cores_b):
        # ca: (r_al, n_a, r_ar)  cb: (r_bl, n_b, r_br)
        # outer product core: (r_al * r_bl, n_a * n_b, r_ar * r_br)
        r_al, n_a, r_ar = ca.shape
        r_bl, n_b, r_br = cb.shape
        # Use kron product structure
        core = np.einsum("inj,kmn->ikmjn", ca, cb)
        # Reshape: (r_al * r_bl, n_a * n_b, r_ar * r_br)  -- note reorder
        core = core.reshape(r_al * r_bl, n_a * n_b, r_ar * r_br)
        cores_out.append(core.astype(np.float32))
    return cores_out


def fused_tt_hadamard(
    cores_a: list,
    cores_b: list,
) -> list:
    """
    Elementwise (Hadamard) product of two TT-tensors.

    Result has bond dimensions equal to products of input bond dimensions.

    Parameters
    ----------
    cores_a, cores_b : list of np.ndarray (r, n, r')

    Returns
    -------
    cores_out : list of np.ndarray
    """
    assert len(cores_a) == len(cores_b)
    cores_out = []
    for ca, cb in zip(cores_a, cores_b):
        r_al, n_a, r_ar = ca.shape
        r_bl, n_b, r_br = cb.shape
        assert n_a == n_b, "Physical dimensions must match for Hadamard product."
        # Khatri-Rao product of bond dimensions
        core = np.einsum("inj,kmj->ikmj", ca, cb)
        core = core.reshape(r_al * r_bl, n_a, r_ar * r_br)
        cores_out.append(core.astype(np.float32))
    return cores_out


def fused_tt_scale(cores: list, scalar: float) -> list:
    """
    Scale a TT-tensor by a scalar (applied to first core).

    Parameters
    ----------
    cores : list of np.ndarray
    scalar : float

    Returns
    -------
    scaled_cores : list of np.ndarray
    """
    if not cores:
        return cores
    result = [c.copy() for c in cores]
    result[0] = (result[0] * scalar).astype(np.float32)
    return result


def fused_tt_add(
    cores_a: list,
    cores_b: list,
) -> list:
    """
    Add two TT-tensors (direct sum construction).

    Bond dimensions of result = sum of bond dimensions.

    Parameters
    ----------
    cores_a, cores_b : list of np.ndarray (r, n, r')

    Returns
    -------
    cores_sum : list of np.ndarray
    """
    assert len(cores_a) == len(cores_b)
    d = len(cores_a)
    cores_sum = []

    for k in range(d):
        ca = cores_a[k]   # (r_al, n, r_ar)
        cb = cores_b[k]   # (r_bl, n, r_br)
        r_al, n, r_ar = ca.shape
        r_bl, _, r_br = cb.shape

        if k == 0:
            # Left boundary: concatenate along right bond
            core = np.concatenate([ca, cb], axis=2)   # (1, n, r_ar + r_br)
        elif k == d - 1:
            # Right boundary: concatenate along left bond
            core = np.concatenate([ca, cb], axis=0)   # (r_al + r_bl, n, 1)
        else:
            # Middle: block diagonal structure
            # (r_al + r_bl, n, r_ar + r_br)
            core = np.zeros((r_al + r_bl, n, r_ar + r_br), dtype=np.float64)
            core[:r_al, :, :r_ar] = ca
            core[r_al:, :, r_ar:] = cb

        cores_sum.append(core.astype(np.float32))

    return cores_sum


def fused_tt_trace(cores: list) -> float:
    """
    Compute the trace of a TT-matrix.

    Only valid when input and output modes are equal (square matrix).

    Parameters
    ----------
    cores : list of np.ndarray (r_l, n, m, r_r) — TT-matrix cores

    Returns
    -------
    trace : float
    """
    # Contract each core's input-output trace
    result = np.ones((1, 1))  # left boundary
    for core in cores:
        if core.ndim == 3:
            # TT-vector: sum over physical index
            n = core.shape[1]
            traced = core.sum(axis=1)   # (r_l, r_r)
        elif core.ndim == 4:
            # TT-matrix: trace over input=output
            r_l, n, m, r_r = core.shape
            min_nm = min(n, m)
            traced = core[:, :min_nm, :min_nm, :].trace(axis1=1, axis2=2)  # (r_l, r_r)
        else:
            warnings.warn(f"Unexpected core ndim={core.ndim} in fused_tt_trace; skipping.")
            continue
        result = result @ traced
    return float(result.ravel()[0])


def fused_batch_tt_matvec(
    cores: list,
    x_batch: np.ndarray,
) -> np.ndarray:
    """
    Apply a TT-matrix to a batch of vectors.

    Parameters
    ----------
    cores : list of np.ndarray, each (r_l, n_k, m_k, r_r)
        TT-matrix cores.
    x_batch : np.ndarray, shape (B, M)
        Batch of input vectors.

    Returns
    -------
    y_batch : np.ndarray, shape (B, N)
    """
    mode_n = [c.shape[1] for c in cores]
    mode_m = [c.shape[2] for c in cores]
    N = int(np.prod(mode_n))
    M = int(np.prod(mode_m))
    B = x_batch.shape[0]

    y_batch = np.zeros((B, N), dtype=np.float32)
    for b in range(B):
        # Reconstruct W and apply
        result = cores[0].squeeze(0)   # (n_1, m_1, r_1)
        for core in cores[1:]:
            result = np.einsum("...r,rnmR->...nmR", result, core)
        result = result.squeeze(-1)
        perm_n = list(range(0, 2 * len(cores), 2))
        perm_m = list(range(1, 2 * len(cores), 2))
        W = result.transpose(perm_n + perm_m).reshape(N, M)
        y_batch[b] = W @ x_batch[b]

    return y_batch


def estimate_tt_memory_bytes(
    cores: list,
    dtype_bytes: int = 4,
) -> int:
    """
    Estimate memory footprint of a TT representation.

    Parameters
    ----------
    cores : list of np.ndarray
    dtype_bytes : int
        Bytes per element (4 for float32, 8 for float64).

    Returns
    -------
    n_bytes : int
    """
    return sum(int(np.prod(c.shape)) * dtype_bytes for c in cores)


def tt_compression_report(
    original_shape: tuple,
    cores: list,
    dtype_bytes: int = 4,
) -> dict:
    """
    Generate a compression report comparing TT storage to dense storage.

    Parameters
    ----------
    original_shape : tuple
        Shape of the original tensor.
    cores : list of np.ndarray
    dtype_bytes : int

    Returns
    -------
    dict with compression statistics.
    """
    n_original = int(np.prod(original_shape)) * dtype_bytes
    n_tt = estimate_tt_memory_bytes(cores, dtype_bytes)
    return {
        "original_elements": int(np.prod(original_shape)),
        "original_bytes": n_original,
        "tt_elements": sum(int(np.prod(c.shape)) for c in cores),
        "tt_bytes": n_tt,
        "compression_ratio": n_original / max(1, n_tt),
        "bits_per_element": (n_tt * 8) / max(1, int(np.prod(original_shape))),
        "n_cores": len(cores),
        "bond_dimensions": [c.shape[-1] for c in cores[:-1]],
    }


def fuse_consecutive_tt_layers(
    cores_layer1: list,
    cores_layer2: list,
) -> list:
    """
    Fuse two consecutive TT-linear layers into a single TT-linear layer.

    Computes the product layer1 @ layer2 in TT format.

    Parameters
    ----------
    cores_layer1 : list of np.ndarray, each (r_l, n, m, r_r) — first layer
    cores_layer2 : list of np.ndarray, each (r_l, m, k, r_r) — second layer

    Returns
    -------
    fused_cores : list of np.ndarray, each (r_l, n, k, r_r)
    """
    assert len(cores_layer1) == len(cores_layer2)
    fused = []
    for c1, c2 in zip(cores_layer1, cores_layer2):
        # c1: (r1l, n, m, r1r)  c2: (r2l, m, k, r2r)
        r1l, n, m, r1r = c1.shape
        r2l, m2, k, r2r = c2.shape
        assert m == m2, f"Inner dimensions must match: {m} != {m2}"
        # Contract over m
        # fused: (r1l, r2l, n, k, r1r, r2r)
        fused_core = np.einsum("rnmR,smpT->rsnpRT", c1, c2)
        # Merge bond indices: (r1l*r2l, n, k, r1r*r2r)
        fused_core = fused_core.reshape(r1l * r2l, n, k, r1r * r2r)
        fused.append(fused_core.astype(np.float32))
    return fused
