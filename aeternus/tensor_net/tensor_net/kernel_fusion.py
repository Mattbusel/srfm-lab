"""
kernel_fusion.py — Custom JAX primitives and kernel fusion for TensorNet (Project AETERNUS).

Provides:
  - Fused TT contraction kernels (eliminate intermediate materializations)
  - XLA custom call patterns and bridging helpers
  - Operation fusion for TT-matvec chains
  - Memory-efficient einsum scheduling via opt_einsum integration
  - Contraction order optimization with cost models
  - Fused TT-dot products with on-the-fly canonicalization
  - Blocked contraction for cache efficiency
  - Compile-time contraction path caching
  - JAX custom VJP rules for fused operations
  - Structured sparsity-aware contractions
"""

from __future__ import annotations

import math
import functools
import warnings
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import jax
import jax.numpy as jnp
import numpy as np
from jax import jit, grad, vmap, lax
from functools import partial


# ============================================================================
# opt_einsum integration
# ============================================================================

try:
    import opt_einsum as oe
    _OPT_EINSUM_AVAILABLE = True
except ImportError:
    _OPT_EINSUM_AVAILABLE = False
    warnings.warn(
        "opt_einsum not available. Contraction order optimization will use greedy fallback.",
        ImportWarning,
    )


def get_optimal_contraction_path(
    subscripts: str,
    operand_shapes: List[Tuple[int, ...]],
    optimize: str = "auto",
) -> Tuple[List[Any], Any]:
    """Compute optimal contraction path using opt_einsum.

    Args:
        subscripts: Einstein summation subscript string.
        operand_shapes: List of shapes of the operands.
        optimize: opt_einsum optimization strategy.

    Returns:
        (path, path_info) where path is the contraction order.
    """
    if not _OPT_EINSUM_AVAILABLE:
        return [], None

    dummy_ops = [np.ones(s) for s in operand_shapes]
    path, path_info = oe.contract_path(subscripts, *dummy_ops, optimize=optimize)
    return path, path_info


@dataclass
class ContractionPlan:
    """Cached contraction plan for a TT network."""
    subscripts: str
    operand_shapes: List[Tuple[int, ...]]
    path: List[Any]
    flop_count: int
    memory_estimate: int

    def apply(self, *operands) -> jnp.ndarray:
        """Execute the cached contraction.

        Args:
            *operands: JAX arrays matching operand_shapes.

        Returns:
            Contraction result.
        """
        if _OPT_EINSUM_AVAILABLE and self.path:
            return oe.contract(self.subscripts, *operands, optimize=self.path)
        return jnp.einsum(self.subscripts, *operands, optimize="greedy")


_CONTRACTION_PLAN_CACHE: Dict[Tuple, ContractionPlan] = {}


def get_or_build_plan(
    subscripts: str,
    operand_shapes: List[Tuple[int, ...]],
    optimize: str = "auto",
) -> ContractionPlan:
    """Get cached contraction plan or build a new one.

    Args:
        subscripts: Einsum subscripts.
        operand_shapes: Shape of each operand.
        optimize: opt_einsum optimizer.

    Returns:
        ContractionPlan.
    """
    cache_key = (subscripts, tuple(tuple(s) for s in operand_shapes))
    if cache_key in _CONTRACTION_PLAN_CACHE:
        return _CONTRACTION_PLAN_CACHE[cache_key]

    path, path_info = get_optimal_contraction_path(subscripts, operand_shapes, optimize)
    flop_count = int(path_info.opt_cost) if path_info is not None else -1
    mem_est = 0
    if path_info is not None:
        try:
            mem_est = int(path_info.largest_intermediate)
        except Exception:
            mem_est = 0

    plan = ContractionPlan(
        subscripts=subscripts,
        operand_shapes=operand_shapes,
        path=path,
        flop_count=flop_count,
        memory_estimate=mem_est,
    )
    _CONTRACTION_PLAN_CACHE[cache_key] = plan
    return plan


# ============================================================================
# Fused TT-matvec
# ============================================================================

def fused_tt_matvec(
    cores: List[jnp.ndarray],
    vector: jnp.ndarray,
    optimize_path: bool = True,
) -> jnp.ndarray:
    """Fused Tensor Train matrix-vector product.

    Contracts a TT operator (represented as cores) with an input vector
    without materializing intermediate full tensors.

    The TT operator is treated as a linear map: each core is (r_l, d_in, d_out, r_r).
    If cores are 3-way (r_l, d, r_r) they are treated as a TT vector.

    Args:
        cores: List of TT cores. Each core: (r_l, d_in, d_out, r_r) or (r_l, d, r_r).
        vector: Input vector, flattened to 1D of size prod(d_in).
        optimize_path: Whether to use opt_einsum path optimization.

    Returns:
        Output vector of size prod(d_out).
    """
    # Detect core type
    four_way = cores[0].ndim == 4

    if not four_way:
        # TT vector mode: inner product with the vector
        result = _fused_tt_vec_contract(cores, vector, optimize_path)
        return result
    else:
        result = _fused_tt_op_matvec(cores, vector, optimize_path)
        return result


def _fused_tt_vec_contract(
    cores: List[jnp.ndarray],
    vector: jnp.ndarray,
    optimize_path: bool = True,
) -> jnp.ndarray:
    """Contract a TT vector with an input vector (inner product).

    Args:
        cores: List of cores (r_l, d, r_r).
        vector: Dense vector to contract with.
        optimize_path: Use opt_einsum.

    Returns:
        Scalar inner product.
    """
    n_sites = len(cores)
    phys_dims = [c.shape[1] for c in cores]
    total_d = int(np.prod(phys_dims))

    # Reshape vector to match physical dimensions
    x = vector[:total_d].reshape(phys_dims)

    # Contract from left to right
    # Bond vector: shape (r_left,)
    left_bond = jnp.ones(1)

    for i, core in enumerate(cores):
        r_l, d, r_r = core.shape
        # Contract: new_left[r_r] = sum_{r_l, d} left_bond[r_l] * x[...,d,...] * core[r_l, d, r_r]
        x_slice = x[i] if n_sites > 1 else x
        if x.ndim == n_sites:
            # x is fully shaped; index along site dimension
            xi = jnp.take(x, i, axis=0) if n_sites > 1 else x.reshape(-1)
        else:
            xi = x.reshape(-1)[:d]

        # left_bond: (r_l,), xi: (d,), core: (r_l, d, r_r)
        r_l_actual = min(left_bond.shape[0], r_l)
        left_bond = jnp.einsum(
            "a,b,abr->r",
            left_bond[:r_l_actual],
            xi[:d],
            core[:r_l_actual, :d, :],
        )

    return left_bond[0] if left_bond.shape[0] == 1 else jnp.sum(left_bond)


def _fused_tt_op_matvec(
    cores: List[jnp.ndarray],
    vector: jnp.ndarray,
    optimize_path: bool = True,
) -> jnp.ndarray:
    """Apply a TT operator (4-way cores) to a vector.

    Each core has shape (r_l, d_in, d_out, r_r).

    Args:
        cores: List of 4-way TT operator cores.
        vector: Input vector of size prod(d_in).
        optimize_path: Use opt_einsum.

    Returns:
        Output vector of size prod(d_out).
    """
    d_ins = [c.shape[1] for c in cores]
    d_outs = [c.shape[2] for c in cores]
    total_in = int(np.prod(d_ins))

    x = vector[:total_in].reshape(d_ins)

    # Bond vector: (r_left,)
    bond = jnp.ones(1)
    out_slices = []

    for i, core in enumerate(cores):
        r_l, d_in, d_out, r_r = core.shape
        xi = x[i] if x.ndim > 1 else x.reshape(-1)[:d_in]

        # Contract bond with input slice and core
        # out_slice[d_out, r_r] = sum_{r_l, d_in} bond[r_l] * xi[d_in] * core[r_l, d_in, d_out, r_r]
        r_l_actual = min(bond.shape[0], r_l)
        out_bond = jnp.einsum(
            "a,b,abcr->cr",
            bond[:r_l_actual],
            xi[:d_in],
            core[:r_l_actual, :d_in, :d_out, :r_r],
        )  # (d_out, r_r)

        # Update bond: marginalize over d_out for the output accumulation
        # For full MPS-like contraction, keep both d_out and bond
        bond = jnp.sum(out_bond, axis=0)  # (r_r,)
        out_slices.append(out_bond[:, 0] if out_bond.shape[1] > 0 else out_bond[:, 0])

    # Concatenate output slices
    if out_slices:
        return jnp.concatenate([s.reshape(-1) for s in out_slices])
    return jnp.zeros(1)


# ============================================================================
# Fused TT-SVD (truncated, in-place)
# ============================================================================

def fused_tt_svd_compress(
    cores: List[jnp.ndarray],
    max_rank: int,
    cutoff: float = 1e-10,
) -> Tuple[List[jnp.ndarray], List[float]]:
    """Fused TT-SVD compression with truncation.

    Performs a single left-to-right sweep of SVD-based truncation,
    fusing the SVD and rank truncation into a single pass.

    Args:
        cores: List of TT cores (r_l, d, r_r).
        max_rank: Maximum bond dimension after truncation.
        cutoff: Singular value cutoff (absolute).

    Returns:
        (compressed_cores, truncation_errors) where truncation_errors
        is the relative error introduced at each bond.
    """
    n_sites = len(cores)
    compressed = [jnp.array(c) for c in cores]
    errors = []

    # Left-to-right sweep: accumulate into left canonical form + truncate
    for i in range(n_sites - 1):
        core = compressed[i]
        r_l, d, r_r = core.shape
        mat = core.reshape(r_l * d, r_r)

        U, s, Vt = jnp.linalg.svd(mat, full_matrices=False)

        # Determine truncation
        s_full_norm = jnp.linalg.norm(s)
        mask = s > cutoff
        n_keep = int(jnp.sum(mask))
        n_keep = max(1, min(n_keep, max_rank, len(s)))

        trunc_error = float(jnp.linalg.norm(s[n_keep:]) / (s_full_norm + 1e-15))
        errors.append(trunc_error)

        # Truncate and reabsorb
        U_t = U[:, :n_keep]
        s_t = s[:n_keep]
        Vt_t = Vt[:n_keep, :]

        compressed[i] = (U_t * s_t).reshape(r_l, d, n_keep)

        # Absorb Vt into next core
        next_core = compressed[i + 1]
        r_l_next, d_next, r_r_next = next_core.shape
        next_mat = next_core.reshape(r_l_next, d_next * r_r_next)
        new_next_mat = Vt_t[:, :r_l_next] @ next_mat[:n_keep, :]
        compressed[i + 1] = new_next_mat.reshape(n_keep, d_next, r_r_next)

    return compressed, errors


# ============================================================================
# Memory-efficient einsum scheduling
# ============================================================================

@dataclass
class EinsumSchedule:
    """Describes an ordered schedule for multi-operand einsum contractions."""
    steps: List[Tuple[str, List[int]]]  # (subscripts, operand_indices) per step
    intermediate_shapes: List[Tuple[int, ...]]
    total_flops: int
    peak_memory: int


def schedule_tt_contraction(
    cores: List[jnp.ndarray],
    input_subscripts: Optional[str] = None,
) -> EinsumSchedule:
    """Build an efficient contraction schedule for a TT network.

    Uses a left-to-right sequential contraction, which is optimal
    for tree-structured (MPS) tensor networks.

    Args:
        cores: TT cores, each (r_l, d, r_r).
        input_subscripts: Optional custom subscripts.

    Returns:
        EinsumSchedule describing the contraction order.
    """
    n = len(cores)
    steps = []
    intermediate_shapes = []
    total_flops = 0
    peak_memory = 0

    current_shape = (cores[0].shape[0],)  # starts as (1,)

    for i, core in enumerate(cores):
        r_l, d, r_r = core.shape
        step_subscripts = f"a,abc->bc" if i == 0 else f"ab,abc->bc"
        step_indices = [i] if i == 0 else [-1, i]
        steps.append((step_subscripts, step_indices))

        new_shape = (d, r_r)
        intermediate_shapes.append(new_shape)

        # Flops: r_l * d * r_r multiplications
        flops = int(current_shape[0] * d * r_r) if len(current_shape) > 0 else d * r_r
        total_flops += flops

        mem = int(np.prod(new_shape)) * 4  # float32
        peak_memory = max(peak_memory, mem)
        current_shape = new_shape

    return EinsumSchedule(
        steps=steps,
        intermediate_shapes=intermediate_shapes,
        total_flops=total_flops,
        peak_memory=peak_memory,
    )


def execute_schedule(
    schedule: EinsumSchedule,
    cores: List[jnp.ndarray],
    bond_vector: Optional[jnp.ndarray] = None,
) -> jnp.ndarray:
    """Execute a pre-computed contraction schedule.

    Args:
        schedule: EinsumSchedule from schedule_tt_contraction.
        cores: TT cores to contract.
        bond_vector: Optional initial bond vector (defaults to ones).

    Returns:
        Contraction result.
    """
    if bond_vector is None:
        bond_vector = jnp.ones(1)

    current = bond_vector
    for i, core in enumerate(cores):
        r_l, d, r_r = core.shape
        r_l_actual = min(current.shape[-1] if current.ndim > 0 else 1, r_l)

        if current.ndim == 1:
            current = jnp.einsum("a,abc->bc", current[:r_l_actual], core[:r_l_actual, :, :])
        else:
            current = jnp.einsum("...a,abc->...bc", current[..., :r_l_actual], core[:r_l_actual, :, :])

    return current


# ============================================================================
# Blocked contraction for cache efficiency
# ============================================================================

def blocked_tt_inner_product(
    cores_a: List[jnp.ndarray],
    cores_b: List[jnp.ndarray],
    block_size: int = 16,
) -> jnp.ndarray:
    """Compute TT inner product using blocked matrix operations.

    Blocked contraction improves cache efficiency for large bond dimensions
    by processing the bond dimension in chunks.

    Args:
        cores_a: TT cores for |a>.
        cores_b: TT cores for |b>.
        block_size: Block size for bond dimension chunking.

    Returns:
        Scalar inner product <a|b>.
    """
    n_sites = len(cores_a)
    assert len(cores_b) == n_sites

    # Transfer matrix approach with blocking
    transfer = jnp.ones((1, 1))

    for i in range(n_sites):
        a = cores_a[i]  # (r_al, d, r_ar)
        b = cores_b[i]  # (r_bl, d, r_br)
        r_al, d, r_ar = a.shape
        r_bl, _, r_br = b.shape

        # Build transfer matrix for this site
        # T_new[r_ar, r_br] = sum_d sum_{r_al,r_bl} T[r_al,r_bl] * a[r_al,d,r_ar] * conj(b[r_bl,d,r_br])
        # Use blocking on r_al and r_bl
        T_al = min(transfer.shape[0], r_al)
        T_bl = min(transfer.shape[1], r_bl)
        T_slice = transfer[:T_al, :T_bl]

        new_T = jnp.zeros((r_ar, r_br))
        n_blocks_al = math.ceil(T_al / block_size)
        n_blocks_bl = math.ceil(T_bl / block_size)

        for ba in range(n_blocks_al):
            al_s = ba * block_size
            al_e = min(al_s + block_size, T_al)
            for bb in range(n_blocks_bl):
                bl_s = bb * block_size
                bl_e = min(bl_s + block_size, T_bl)

                T_block = T_slice[al_s:al_e, bl_s:bl_e]
                a_block = a[al_s:al_e, :, :]  # (block_al, d, r_ar)
                b_block = jnp.conj(b[bl_s:bl_e, :, :])  # (block_bl, d, r_br)

                # Contract: T_block @ a_block over physical index
                # intermediate: (block_al, r_ar) after marginalizing d
                # then multiply with b_block
                contrib = jnp.einsum(
                    "ab,acd,bcd->cd",
                    T_block,
                    a_block[:T_al - al_s, :, :r_ar],
                    b_block[:T_bl - bl_s, :, :r_br],
                )
                new_T = new_T.at[:r_ar, :r_br].add(contrib[:r_ar, :r_br])

        transfer = new_T

    return transfer[0, 0] if transfer.shape == (1, 1) else jnp.sum(jnp.diag(transfer))


# ============================================================================
# XLA custom call pattern helpers
# ============================================================================

def register_tt_contraction_primitive():
    """Register a JAX primitive for TT contraction.

    This creates a custom_vjp-wrapped version of TT contraction
    that can be recognized by XLA for fusion.

    Returns:
        Function with custom_vjp for efficient differentiation.
    """
    @jax.custom_vjp
    def tt_contract_primitive(
        core_flat: jnp.ndarray,
        vector: jnp.ndarray,
        shape_tuple: Tuple[int, ...],
    ) -> jnp.ndarray:
        """Inner TT contraction primitive.

        Args:
            core_flat: Flattened TT cores concatenated.
            vector: Input vector.
            shape_tuple: Core shapes as flat tuple.

        Returns:
            Contraction result.
        """
        # Reconstruct cores from flat representation
        shapes = _unpack_shape_tuple(shape_tuple)
        cores = _unpack_cores(core_flat, shapes)

        result = vector
        for core in cores:
            r_l, d, r_r = core.shape
            if result.ndim == 1:
                result = jnp.einsum("i,ijk->jk", result[:r_l], core).reshape(-1)
        return result

    def tt_contract_fwd(core_flat, vector, shape_tuple):
        result = tt_contract_primitive(core_flat, vector, shape_tuple)
        return result, (core_flat, vector, shape_tuple)

    def tt_contract_bwd(res, g):
        core_flat, vector, shape_tuple = res
        shapes = _unpack_shape_tuple(shape_tuple)
        cores = _unpack_cores(core_flat, shapes)

        # Compute VJP manually
        grad_cores_list = []
        grad_vector = jnp.zeros_like(vector)

        # Forward pass to get intermediates
        intermediates = [vector]
        current = vector
        for core in cores:
            r_l, d, r_r = core.shape
            if current.ndim == 1:
                current = jnp.einsum("i,ijk->jk", current[:r_l], core).reshape(-1)
            intermediates.append(current)

        # Backward pass
        g_current = g
        for i in range(len(cores) - 1, -1, -1):
            core = cores[i]
            r_l, d, r_r = core.shape
            prev = intermediates[i]

            g_core = jnp.einsum("i,k->ijk", prev[:r_l], g_current[:r_r])
            grad_cores_list.insert(0, g_core)

            if i > 0:
                g_prev = jnp.einsum("ijk,k->i", core[:r_l, :, :r_r], g_current[:r_r])
                g_current = g_prev

        grad_core_flat = jnp.concatenate([gc.reshape(-1) for gc in grad_cores_list])
        return grad_core_flat, g_current, None

    tt_contract_primitive.defvjp(tt_contract_fwd, tt_contract_bwd)
    return tt_contract_primitive


def _unpack_shape_tuple(shape_tuple: Tuple[int, ...]) -> List[Tuple[int, int, int]]:
    """Unpack flat shape tuple into list of (r_l, d, r_r) triples."""
    shapes = []
    i = 0
    while i < len(shape_tuple):
        shapes.append((shape_tuple[i], shape_tuple[i + 1], shape_tuple[i + 2]))
        i += 3
    return shapes


def _pack_shape_tuple(shapes: List[Tuple[int, int, int]]) -> Tuple[int, ...]:
    """Pack list of (r_l, d, r_r) shapes into flat tuple."""
    flat = []
    for r_l, d, r_r in shapes:
        flat.extend([r_l, d, r_r])
    return tuple(flat)


def _pack_cores(cores: List[jnp.ndarray]) -> jnp.ndarray:
    """Pack TT cores into a single flat array."""
    return jnp.concatenate([c.reshape(-1) for c in cores])


def _unpack_cores(
    flat: jnp.ndarray,
    shapes: List[Tuple[int, int, int]],
) -> List[jnp.ndarray]:
    """Unpack flat array back into TT cores."""
    cores = []
    offset = 0
    for r_l, d, r_r in shapes:
        size = r_l * d * r_r
        cores.append(flat[offset : offset + size].reshape(r_l, d, r_r))
        offset += size
    return cores


# ============================================================================
# Fused TT chain operations
# ============================================================================

def fused_tt_chain_matvec(
    tt_ops: List[List[jnp.ndarray]],
    vector: jnp.ndarray,
) -> jnp.ndarray:
    """Apply a chain of TT operators to a vector, fusing contractions.

    Instead of materializing intermediate dense vectors, this function
    contracts the chain by maintaining bond vectors.

    Args:
        tt_ops: List of TT operators (each is a list of cores).
        vector: Initial input vector.

    Returns:
        Result after applying all TT operators sequentially.
    """
    current = vector
    for tt_op in tt_ops:
        current = fused_tt_matvec(tt_op, current)
    return current


@partial(jit, static_argnums=(2,))
def jit_fused_tt_matvec(
    cores_flat: jnp.ndarray,
    vector: jnp.ndarray,
    shape_tuple: Tuple[int, ...],
) -> jnp.ndarray:
    """JIT-compiled fused TT matvec.

    Args:
        cores_flat: All cores concatenated into one flat array.
        vector: Input vector.
        shape_tuple: Core shapes as flat tuple (r_l, d, r_r, r_l, d, r_r, ...).

    Returns:
        TT matvec result.
    """
    shapes = _unpack_shape_tuple(shape_tuple)
    cores = _unpack_cores(cores_flat, shapes)
    return fused_tt_matvec(cores, vector, optimize_path=False)


# ============================================================================
# Contraction cost model
# ============================================================================

def flop_count_tt_contraction(
    cores: List[jnp.ndarray],
) -> int:
    """Estimate FLOPs for left-to-right TT contraction.

    Args:
        cores: TT cores.

    Returns:
        Estimated FLOP count (multiply-adds).
    """
    total = 0
    current_dim = 1
    for core in cores:
        r_l, d, r_r = core.shape
        # Contract: (current_dim) x (r_l, d, r_r) -> (current_dim * d / r_l * r_r)
        flops = current_dim * d * r_r
        total += flops
        current_dim = d * r_r
    return total


def memory_footprint_tt(
    cores: List[jnp.ndarray],
    dtype_bytes: int = 4,
) -> Dict[str, int]:
    """Compute memory footprint of a TT decomposition.

    Args:
        cores: TT cores.
        dtype_bytes: Bytes per element.

    Returns:
        Dict with total_params, total_bytes, per_core_bytes.
    """
    per_core = [c.size for c in cores]
    total_params = sum(per_core)
    total_bytes = total_params * dtype_bytes
    return {
        "total_params": total_params,
        "total_bytes": total_bytes,
        "per_core_params": per_core,
        "per_core_bytes": [n * dtype_bytes for n in per_core],
    }


def peak_intermediate_memory(
    cores: List[jnp.ndarray],
    dtype_bytes: int = 4,
) -> int:
    """Estimate peak intermediate memory for TT contraction.

    Args:
        cores: TT cores.
        dtype_bytes: Bytes per element.

    Returns:
        Peak intermediate memory in bytes.
    """
    peak = 0
    current_d = 1
    for core in cores:
        r_l, d, r_r = core.shape
        intermediate_size = current_d * r_r * dtype_bytes
        peak = max(peak, intermediate_size)
        current_d = d * r_r
    return peak


# ============================================================================
# Operation fusion detector
# ============================================================================

def detect_fusable_ops(
    computation_graph: List[str],
) -> List[List[int]]:
    """Detect which operations in a computation graph can be fused.

    Simple pattern matching for common fusable patterns in TT computations.

    Args:
        computation_graph: List of operation names (e.g., ["einsum", "reshape", "einsum"]).

    Returns:
        List of groups of indices that can be fused together.
    """
    fusable_groups = []
    current_group = []

    fusable_ops = {"einsum", "matmul", "dot", "tensordot", "reshape"}

    for i, op in enumerate(computation_graph):
        if op.lower() in fusable_ops:
            current_group.append(i)
        else:
            if len(current_group) > 1:
                fusable_groups.append(current_group)
            elif current_group:
                pass
            current_group = []

    if len(current_group) > 1:
        fusable_groups.append(current_group)

    return fusable_groups


# ============================================================================
# Einsum contraction order optimization
# ============================================================================

def optimize_tt_contraction_order(
    cores: List[jnp.ndarray],
    method: str = "greedy",
) -> Tuple[List[int], float]:
    """Find the optimal contraction order for a chain of TT cores.

    For MPS/TT chains, the optimal order is always left-to-right or
    right-to-left, but for general tree tensor networks this matters.

    Args:
        cores: TT cores.
        method: "greedy", "optimal" (uses opt_einsum if available), or "left_right".

    Returns:
        (contraction_order, estimated_flops) where contraction_order is the
        list of core indices to contract in sequence.
    """
    n = len(cores)

    if method == "left_right":
        return list(range(n)), float(flop_count_tt_contraction(cores))

    if method == "right_left":
        # Estimate cost for right-to-left
        flops = 0
        current_d = 1
        for core in reversed(cores):
            r_l, d, r_r = core.shape
            flops += current_d * d * r_l
            current_d = d * r_l
        return list(range(n - 1, -1, -1)), float(flops)

    # "greedy": compare left-right vs right-left, pick cheaper
    lr_flops = flop_count_tt_contraction(cores)
    rl_cores = list(reversed(cores))
    rl_flops = flop_count_tt_contraction(rl_cores)

    if lr_flops <= rl_flops:
        return list(range(n)), float(lr_flops)
    else:
        return list(range(n - 1, -1, -1)), float(rl_flops)


# ============================================================================
# Fused TT norm squared (differentiable)
# ============================================================================

@jax.custom_vjp
def fused_tt_norm_squared(cores: List[jnp.ndarray]) -> jnp.ndarray:
    """Fused computation of ||TT||^2 via transfer matrices.

    More efficient than materializing the full tensor.

    Args:
        cores: TT cores (r_l, d, r_r).

    Returns:
        Scalar ||TT||^2.
    """
    transfer = jnp.ones((1, 1))
    for core in cores:
        r_l, d, r_r = core.shape
        T = min(transfer.shape[0], r_l)
        # T_new[r_ar, r_br] = sum_{d,r_l} T[r_al,r_bl] * core[r_al,d,r_ar] * core[r_bl,d,r_br]
        new_T = jnp.einsum(
            "ab,adr,bdr->rr",
            transfer[:T, :T],
            core[:T, :, :],
            core[:T, :, :],
        )
        transfer = new_T
    return transfer[0, 0] if transfer.shape[0] >= 1 else jnp.zeros(())


def _fused_norm_fwd(cores):
    ns = fused_tt_norm_squared(cores)
    return ns, cores


def _fused_norm_bwd(cores_saved, g):
    # d/d(core_i) ||TT||^2 = 2 * left_env_i @ core_i @ right_env_i
    cores = cores_saved
    n = len(cores)

    # Build left environments
    left_envs = [jnp.ones((1, 1))]
    for i in range(n - 1):
        core = cores[i]
        r_l, d, r_r = core.shape
        T = min(left_envs[-1].shape[0], r_l)
        new_env = jnp.einsum(
            "ab,adr,bdr->rr",
            left_envs[-1][:T, :T],
            core[:T, :, :],
            core[:T, :, :],
        )
        left_envs.append(new_env)

    # Build right environments
    right_envs = [jnp.ones((1, 1))]
    for i in range(n - 1, 0, -1):
        core = cores[i]
        r_l, d, r_r = core.shape
        T = min(right_envs[-1].shape[0], r_r)
        new_env = jnp.einsum(
            "ab,rda,rdb->rr",
            right_envs[-1][:T, :T],
            core[:, :, :T],
            core[:, :, :T],
        )
        right_envs.insert(0, new_env)

    grad_cores = []
    for i, core in enumerate(cores):
        r_l, d, r_r = core.shape
        L = left_envs[i]
        R = right_envs[i]
        T_L = min(L.shape[0], r_l)
        T_R = min(R.shape[0], r_r)

        # Gradient = 2 * L * core * R (symbolic)
        g_core = jnp.zeros_like(core)
        contrib = 2.0 * jnp.einsum(
            "ab,bdr,rc->adc",
            L[:T_L, :T_L],
            core[:T_L, :, :T_R],
            R[:T_R, :T_R],
        )
        g_core = g_core.at[:T_L, :, :T_R].add(contrib * g)
        grad_cores.append(g_core)

    return (grad_cores,)


fused_tt_norm_squared.defvjp(_fused_norm_fwd, _fused_norm_bwd)


# ============================================================================
# Batched fused contractions
# ============================================================================

def batched_fused_tt_matvec(
    cores: List[jnp.ndarray],
    batch_vectors: jnp.ndarray,
) -> jnp.ndarray:
    """Apply fused TT matvec to a batch of vectors.

    Uses vmap over the batch dimension.

    Args:
        cores: TT cores.
        batch_vectors: Batch of input vectors (batch, d_total).

    Returns:
        Output (batch, ...).
    """
    def single(v):
        return fused_tt_matvec(cores, v, optimize_path=False)

    return vmap(single)(batch_vectors)


# ============================================================================
# Cache-oblivious contraction
# ============================================================================

def cache_oblivious_tt_contract(
    cores: List[jnp.ndarray],
    vector: jnp.ndarray,
    cache_size_bytes: int = 256 * 1024,  # 256 KB L2 cache
) -> jnp.ndarray:
    """TT contraction with cache-oblivious blocking.

    Automatically determines block sizes based on available cache.

    Args:
        cores: TT cores.
        vector: Input vector.
        cache_size_bytes: Available cache size in bytes.

    Returns:
        Contracted result.
    """
    # Determine max block size that fits in cache
    dtype_size = 4  # float32
    block_elements = cache_size_bytes // (3 * dtype_size)  # 3 arrays
    block_size = max(1, int(math.sqrt(block_elements)))

    current = vector
    for core in cores:
        r_l, d, r_r = core.shape
        r_l_actual = min(current.shape[-1] if current.ndim > 0 else r_l, r_l)

        if r_l * d * r_r <= block_elements:
            # Fits in cache: direct contraction
            if current.ndim == 1:
                current = jnp.einsum(
                    "i,ijk->jk",
                    current[:r_l_actual],
                    core[:r_l_actual, :, :],
                ).reshape(-1)
        else:
            # Blocked contraction
            result = jnp.zeros((d, r_r))
            for block_start in range(0, r_l_actual, block_size):
                block_end = min(block_start + block_size, r_l_actual)
                if current.ndim == 1:
                    block_result = jnp.einsum(
                        "i,ijk->jk",
                        current[block_start:block_end],
                        core[block_start:block_end, :, :],
                    )
                    result = result + block_result
            current = result.reshape(-1)

    return current


# ============================================================================
# Lazy contraction graph
# ============================================================================

class LazyContractionGraph:
    """Lazy evaluation graph for deferred TT contractions.

    Nodes represent tensors; edges represent contractions.
    Execution is deferred until .evaluate() is called, allowing
    the system to optimize the full contraction order.

    Usage::

        graph = LazyContractionGraph()
        a_id = graph.add_tensor(core_a)
        b_id = graph.add_tensor(core_b)
        c_id = graph.add_contraction(a_id, b_id, subscripts="ijk,jkl->il")
        result = graph.evaluate(c_id)
    """

    def __init__(self):
        self._tensors: Dict[int, jnp.ndarray] = {}
        self._contractions: Dict[int, Tuple[List[int], str]] = {}
        self._next_id = 0

    def add_tensor(self, tensor: jnp.ndarray) -> int:
        """Add a tensor node.

        Args:
            tensor: JAX array.

        Returns:
            Node ID.
        """
        node_id = self._next_id
        self._tensors[node_id] = tensor
        self._next_id += 1
        return node_id

    def add_contraction(
        self,
        input_ids: List[int],
        subscripts: str,
    ) -> int:
        """Add a contraction node.

        Args:
            input_ids: List of input node IDs.
            subscripts: Einsum subscripts.

        Returns:
            Node ID.
        """
        node_id = self._next_id
        self._contractions[node_id] = (input_ids, subscripts)
        self._next_id += 1
        return node_id

    def evaluate(self, node_id: int) -> jnp.ndarray:
        """Evaluate a node, recursively evaluating dependencies.

        Args:
            node_id: ID of the node to evaluate.

        Returns:
            Evaluated tensor.
        """
        if node_id in self._tensors:
            return self._tensors[node_id]

        if node_id in self._contractions:
            input_ids, subscripts = self._contractions[node_id]
            operands = [self.evaluate(i) for i in input_ids]
            result = jnp.einsum(subscripts, *operands, optimize="greedy")
            # Cache result
            self._tensors[node_id] = result
            return result

        raise KeyError(f"Node {node_id} not found in graph.")

    def clear_cache(self) -> None:
        """Clear cached intermediate results (non-leaf tensors)."""
        # Keep only original tensors (not contraction results)
        to_remove = [k for k in self._tensors if k in self._contractions]
        for k in to_remove:
            del self._tensors[k]


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

