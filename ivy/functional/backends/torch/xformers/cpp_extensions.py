import xformers
import torch
from typing import Optional, Tuple


def matmul_with_mask(
    a: torch.Tensor, b: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    return xformers.torch.ops.xformers.matmul_with_mask(a, b, mask)


def sddmm_sputnik(
    a: torch.Tensor,
    b: torch.Tensor,
    row_indices: torch.Tensor,
    column_indices: torch.Tensor,
) -> torch.Tensor:
    return xformers.torch.ops.xformers.sddmm_sputnik(a, b, row_indices, column_indices)


def efficient_attention_forward_small_k(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    compute_logsumexp: bool,
    attn_bias_: Optional[torch.Tensor],
    p: float,
) -> Tuple[torch.Tensor, torch.Tensor, int, int]:
    return xformers.torch.ops.xformers.efficient_attention_forward_small_k(
        query, key, value, compute_logsumexp, attn_bias_, p
    )


def efficient_attention_backward_small_k(
    grad_out: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    logsumexp: torch.Tensor,
    output: torch.Tensor,
    attn_bias_: Optional[torch.Tensor],
    p: float,
    rng_seed: int,
    rng_offset: int,
) -> Tuple[torch.Tensor, torch.Tensor, int, int]:
    return xformers.torch.ops.xformers.efficient_attention_backward_small_k(
        grad_out,
        query,
        key,
        value,
        logsumexp,
        output,
        attn_bias_,
        p,
        rng_seed,
        rng_offset,
    )


def sparse_softmax_sputnik(
    m: int,
    n: int,
    row_indices: torch.Tensor,
    values: torch.Tensor,
    row_offsets: torch.Tensor,
    column_indices: torch.Tensor,
) -> torch.Tensor:
    return xformers.torch.ops.xformers.sparse_softmax_sputnik(
        m,
        n,
        row_indices,
        values,
        row_offsets,
        column_indices,
    )


def sparse_softmax_backward_sputnik(
    m: int,
    n: int,
    row_indices: torch.Tensor,
    values: torch.Tensor,
    grad: torch.Tensor,
    row_offsets: torch.Tensor,
    column_indices: torch.Tensor,
) -> torch.Tensor:
    return xformers.torch.ops.xformers.sparse_softmax_sputnik(
        m,
        n,
        row_indices,
        values,
        grad,
        row_offsets,
        column_indices,
    )


def spmm_sputnik(
    b: torch.Tensor,
    row_indices: torch.Tensor,
    values: torch.Tensor,
    row_offsets: torch.Tensor,
    column_indices: torch.Tensor,
    m: int,
) -> torch.Tensor:
    return xformers.torch.ops.xformers.spmm_sputnik(
        b,
        row_indices,
        values,
        row_offsets,
        column_indices,
    )


def efficient_attention_backward_cutlass(
    grad_out_: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    bias: Optional[torch.Tensor],
    cu_seqlens_q: Optional[torch.Tensor],
    cu_seqlens_k: Optional[torch.Tensor],
    max_seqlen_q: int,
    max_seqlen_k: int,
    logsumexp: torch.Tensor,
    out: torch.Tensor,
    dropout_p: float,
    rng_seed: int,
    rng_offset: int,
    custom_mask_type: int,
    scale: Optional[float],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return xformers.torch.ops.xformers.efficient_attention_backward_cutlass(
        grad_out_,
        query,
        key,
        value,
        bias,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        logsumexp,
        out,
        dropout_p,
        rng_seed,
        rng_offset,
        custom_mask_type,
        scale,
    )


def _cutlass_rand_uniform(*args, **kwargs):
    return xformers.torch.ops.xformers._cutlass_rand_uniform(*args, **kwargs)


def efficient_attention_forward_cutlass(*args, **kwargs):
    return xformers.torch.ops.xformers.efficient_attention_forward_cutlass(
        *args, **kwargs
    )


def _temp_dropout(*args, **kwargs):
    return xformers.torch.ops.xformers._temp_dropout(*args, **kwargs)


def csr_sddmm(*args, **kwargs):
    return xformers.torch.ops.xformers.csr_sddmm(*args, **kwargs)


def coo_sddmm(*args, **kwargs):
    return xformers.torch.ops.xformers.coo_sddmm(*args, **kwargs)


def dual_gemm_silu_identity_mul(*args, **kwargs):
    return xformers.torch.ops.xformers.dual_gemm_silu_identity_mul(*args, **kwargs)


def gemm_fused_operand_sum(*args, **kwargs):
    return xformers.torch.ops.xformers.gemm_fused_operand_sum(*args, **kwargs)


def silu_bw_fused(*args, **kwargs):
    return xformers.torch.ops.xformers.silu_bw_fused(*args, **kwargs)


def swiglu_packedw(*args, **kwargs):
    return xformers.torch.ops.xformers.swiglu_packedw(*args, **kwargs)
