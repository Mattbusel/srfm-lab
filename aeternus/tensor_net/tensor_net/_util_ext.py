"""Utility extensions for TensorNet — appended to kernel_fusion.py."""


import numpy as np


def tt_frobenius_norm(cores: list) -> float:
    """
    Compute Frobenius norm of a TT-tensor.

    Parameters
    ----------
    cores : list of np.ndarray, each (r_l, n, r_r)

    Returns
    -------
    norm : float
    """
    transfer = np.ones((1, 1))
    for core in cores:
        r_l, n, r_r = core.shape
        c = core  # (r_l, n, r_r)
        G = np.einsum("inj,ij,ink->jk", c, transfer, c)
        transfer = G
    return float(np.sqrt(max(0.0, transfer[0, 0])))


def tt_dot_product(cores_a: list, cores_b: list) -> float:
    """
    Compute inner product of two TT-tensors.

    Parameters
    ----------
    cores_a, cores_b : list of np.ndarray, each (r_l, n, r_r)

    Returns
    -------
    dot : float
    """
    assert len(cores_a) == len(cores_b)
    transfer = np.ones((1, 1))
    for ca, cb in zip(cores_a, cores_b):
        r_al, n_a, r_ar = ca.shape
        r_bl, n_b, r_br = cb.shape
        assert n_a == n_b
        G = np.einsum("inj,ij,ink->jk", ca, transfer, cb)
        transfer = G
    return float(transfer[0, 0])


def normalise_tt_cores(cores: list) -> list:
    """
    Normalise a TT-tensor so its Frobenius norm equals 1.

    Parameters
    ----------
    cores : list of np.ndarray

    Returns
    -------
    normalised : list of np.ndarray
    """
    norm = tt_frobenius_norm(cores)
    if norm < 1e-12:
        return [c.copy() for c in cores]
    result = [c.copy() for c in cores]
    result[0] = (result[0] / norm).astype(np.float32)
    return result


def tt_mean_absolute_deviation(
    tensor: np.ndarray,
    reference: np.ndarray | None = None,
) -> float:
    """
    Compute mean absolute deviation of a tensor from its mean or reference.

    Parameters
    ----------
    tensor : np.ndarray
    reference : np.ndarray, optional

    Returns
    -------
    mad : float
    """
    if reference is None:
        reference = tensor.mean(axis=0, keepdims=True)
    return float(np.abs(tensor - reference).mean())


def cores_to_full_tensor(cores: list) -> np.ndarray:
    """
    Reconstruct a full dense tensor from TT cores.

    Parameters
    ----------
    cores : list of np.ndarray, each (r_l, n_k, r_r)

    Returns
    -------
    tensor : np.ndarray, shape (n_1, n_2, ..., n_d)
    """
    result = cores[0].squeeze(0)   # (n_1, r_1)
    for core in cores[1:]:
        # result: (..., r) and core: (r, n_k, r')
        result = np.einsum("...r,rnR->...nR", result, core)
    return result.squeeze(-1)


def full_tensor_to_cores(
    tensor: np.ndarray,
    rank: int,
) -> list:
    """
    Decompose a dense tensor into TT cores via left-to-right SVD.

    Parameters
    ----------
    tensor : np.ndarray, shape (n_1, n_2, ..., n_d)
    rank : int
        Maximum TT bond dimension.

    Returns
    -------
    cores : list of np.ndarray
    """
    shape = tensor.shape
    d = len(shape)
    cores = []
    C = tensor.copy().astype(np.float64)
    r_left = 1

    for k in range(d - 1):
        n_k = shape[k]
        C = C.reshape(r_left * n_k, -1)
        U, s, Vt = np.linalg.svd(C, full_matrices=False)
        r_right = min(rank, len(s))
        U = U[:, :r_right]
        s = s[:r_right]
        core = U.reshape(r_left, n_k, r_right)
        cores.append(core.astype(np.float32))
        C = np.diag(s) @ Vt[:r_right, :]
        r_left = r_right

    # Last core
    n_last = shape[-1]
    core = C.reshape(r_left, n_last, 1)
    cores.append(core.astype(np.float32))
    return cores


def tt_bond_dimensions(cores: list) -> list:
    """
    Return the list of TT bond dimensions.

    Parameters
    ----------
    cores : list of np.ndarray

    Returns
    -------
    bonds : list of int  (length = n_cores - 1)
    """
    return [c.shape[-1] for c in cores[:-1]]


def tt_physical_dimensions(cores: list) -> list:
    """
    Return the physical (mode) dimensions.

    Parameters
    ----------
    cores : list of np.ndarray

    Returns
    -------
    dims : list of int
    """
    return [c.shape[1] for c in cores]


def compress_tt_svd(cores: list, rank: int) -> list:
    """
    Recompress a TT-tensor to a lower rank via right-to-left SVD sweep.

    Parameters
    ----------
    cores : list of np.ndarray
    rank : int

    Returns
    -------
    compressed_cores : list of np.ndarray
    """
    tensor = cores_to_full_tensor(cores)
    return full_tensor_to_cores(tensor, rank)
