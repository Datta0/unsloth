# SPDX-License-Identifier: GNU Affero General Public License v3.0
# Copyright 2023-present the Unsloth team. All rights reserved.

import torch
import triton
import triton.language as tl

from .autotuning import (
    get_dW_kernel_configs,
    get_dX_kernel_configs,
    prune_dX_configs,
    prune_kernel_configs_backward_dW,
)

"""
dX backward kernel

- Shapes
    - the forward pass input X shape is [NUM_TOKENS, K] if permute_x else [NUM_TOKENS * TOPK, K]; output y is [NUM_TOKENS * TOPK, N]
    - the backward pass input dy shape is [NUM_TOKENS * TOPK, N], reduce across N, output dX is [NUM_TOKENS * TOPK, K]
- Note that in the backward pass, the output size is still [NUM_TOKENS * TOPK, K] since we still need to accumulate gradients for each expert chosen by the token in a post-processing step.

`permute_x` notes:
- In the forward pass, if we permute X on load, we need to permute on store in the backward pass to restore to original token order
- the output dX with have shape [NUM_TOKENS * TOPK, K] and we need to perform an additional reduction across topk to accumulate gradients
- This is done as a post-processing step in autograd.Function.
- If not `permute_x`, this postprocessing step should take place outside autograd.Function such that the gradient shape matches the input X shape.

`permute_y` notes:
- In the forward pass, if we permuted output on store (e.g., in the second grouped GEMM in fused MoE MLP), we need to permute on load to get from token order to expert grouped order
- We still store in contiguous order since we are writing out dX which will be the input to the backwards pass of the first grouped GEMM

`fused_mul` notes:
- In the forward pass, if we used the multiplication of topk weights (e.g., in the second grouped GEMM in fused MoE MLP), we need to make a few additional changes:
    1) We load topk_weights in natural (token) order.  Since we only enable `fuse_mul` when permuting on store (`permute_y`), we multiply grad_output by topk_weights before backpropagating
    2) We need to calculate the gradient of the topk_weights.  This gets messy since we need do an additional elementwise multiplication in the GEMM main loop and then write out in unpermuted order.  For now, we do not fuse this step but calculate as a simple

Invalid combinations:
- permute_y and use_tma_load: permuting y on store in forward -> load in permuted order in backward, therefore can't use TMA load (unless Blackwell which supports gather / scatter TMA)
- permute_x and use_tma_store: permuting x on load in forward -> store in permuted order in backward, therefore can't use TMA store (unless Blackwell which supports gather / scatter TMA)

TODO:
- We define indices for all conditions and expect that unused indices will be DCE'd during compilation.  Check that this is the case otherwise will result in unnecessary register usage.
"""


@triton.jit
def _grouped_gemm_dX_kernel(
    dY_ptr,  # [M_total, N]
    w_ptr,  # [E, N, K]
    dX_ptr,  # [M_total, K]
    gather_indices_ptr,
    m_sizes_ptr,
    # LoRA weight pointers (only used when FUSE_LORA=True)
    lora_a_ptr,  # (E, R, K) per-expert LoRA A weights
    lora_b_ptr,  # (E, N, R) per-expert LoRA B weights
    lora_scaling,  # float scalar
    # problem sizes
    NUM_EXPERTS: tl.constexpr,
    NUM_TOKENS,
    TOPK: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    NUM_SMS,
    # Tuning parameters
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    PERMUTE_X: tl.constexpr = False,
    PERMUTE_Y: tl.constexpr = False,
    USE_TMA_LOAD_W: tl.constexpr = False,
    USE_TMA_LOAD_dY: tl.constexpr = False,
    USE_TMA_STORE: tl.constexpr = False,
    FLATTEN: tl.constexpr = True,
    # LoRA fusion params
    FUSE_LORA: tl.constexpr = False,
    LORA_R: tl.constexpr = 0,
    BLOCK_R: tl.constexpr = 16,
) -> None:
    TOTAL_TOKENS = NUM_TOKENS * TOPK
    output_dtype = dX_ptr.dtype.element_ty

    tidx = tl.program_id(0)
    # This removes the need for predication along N in the GEMM main loop
    tl.static_assert(N % BLOCK_SIZE_N == 0, "N must be divisible by BLOCK_SIZE_N")
    tl.static_assert(K % BLOCK_SIZE_K == 0, "K must be divisible by BLOCK_SIZE_K")

    # Create TMA descriptors for loading sorted tokens
    # When using TMA load, we don't permute_x, so shape should be [TOTAL_TOKENS, K]
    # Also, we are defining a single global descriptor with single block shape
    # Need to check that this does not result in errors when crossing expert boundaries
    if USE_TMA_LOAD_dY:
        dY_desc = tl.make_tensor_descriptor(
            dY_ptr,
            shape = [TOTAL_TOKENS, N],
            strides = [N, 1],
            block_shape = [BLOCK_SIZE_M, BLOCK_SIZE_N],
        )

    if USE_TMA_LOAD_W:
        expert_stride = N * K
        w_desc = tl.make_tensor_descriptor(
            w_ptr,
            shape = [NUM_EXPERTS, N, K],
            strides = [expert_stride, K, 1],
            block_shape = [1, BLOCK_SIZE_N, BLOCK_SIZE_K],
        )

    m_end = 0
    processed_tiles = 0
    m_block_range = tl.arange(0, BLOCK_SIZE_M)
    n_block_range = tl.arange(0, BLOCK_SIZE_N)
    k_block_range = tl.arange(0, BLOCK_SIZE_K)

    for expert_idx in range(NUM_EXPERTS, flatten = FLATTEN):
        m_start = m_end
        m_size = tl.load(m_sizes_ptr + expert_idx).to(tl.int32)
        m_end = m_start + m_size

        if m_size > 0:
            # Advance n offset to the weights for that respective expert
            n_start = expert_idx * N
            # N_start_offset = g.to(tl.int64) * N
            # tiles for this group's GEMM
            num_m_tiles = tl.cdiv(m_size, BLOCK_SIZE_M)
            num_k_tiles = tl.cdiv(K, BLOCK_SIZE_K)
            num_tiles_per_expert = num_m_tiles * num_k_tiles

            if USE_TMA_STORE:
                # Need to define descript within loop to predicate store along M
                tl.static_assert(
                    K % BLOCK_SIZE_K == 0, "K must be divisible by BLOCK_SIZE_K"
                )
                dX_desc = tl.make_tensor_descriptor(
                    dX_ptr,
                    shape = [m_end, K],
                    strides = [K, 1],
                    block_shape = [BLOCK_SIZE_M, BLOCK_SIZE_K],
                )

            # Lower bound and upper bound are defined relative to the total tiles processed so far
            # This ensures that we are only processing tiles for the current expert group AND
            # we never exceed the total number of tiles for all expert groups
            while tidx >= processed_tiles and tidx < (
                processed_tiles + num_tiles_per_expert
            ):
                group_index = tidx - processed_tiles

                # Output tile for this thread block for this expert group
                tile_m_idx = group_index % num_m_tiles
                tile_k_idx = group_index // num_m_tiles

                if PERMUTE_X or PERMUTE_Y:
                    # These will be used for loading and storing in permuted order
                    gather_offsets = tile_m_idx * BLOCK_SIZE_M + m_block_range
                    # indices_to_gather = m_start + gather_offsets
                    indices_to_gather = m_start + tl.max_contiguous(
                        tl.multiple_of(gather_offsets % m_size, BLOCK_SIZE_M),
                        BLOCK_SIZE_M,
                    )
                    expert_token_idx = tl.load(
                        gather_indices_ptr + indices_to_gather,
                        mask = indices_to_gather < TOTAL_TOKENS,
                    )
                    expert_token_offsets = expert_token_idx[:, None]

                    # Masks for permuted load and store
                    row_mask = gather_offsets < m_size
                    row_mask = row_mask[:, None]

                    # We only take into account the following two cases: (PERMUTE_X and NOT PERMUTE_Y) and (NOT PERMUTE_X and PERMUTE_Y)
                    # Hence, we can make the following simplifying assumptions when loading and storing
                    # Note the different strides between the two cases: the offsets for loading and storing are flipped and the strides must also be adjusted

                    if PERMUTE_X:
                        # Case where we permuted on load in the forward pass (typically first grouped GEMM in MoE MLP)
                        load_a_idx = (
                            indices_to_gather[:, None] * N
                        )  # Load in contiguous (expert grouped) order
                        store_idx = (
                            expert_token_offsets * K
                        )  # Permute on store from expert -> token order
                    else:
                        # Case where we permuted on store in the forward pass (typically second grouped GEMM in MoE MLP)
                        load_a_idx = (
                            expert_token_offsets * N
                        )  # Permute on load from token -> expert order
                        store_idx = (
                            indices_to_gather[:, None] * K
                        )  # Store in contiguous order
                else:
                    # # Position in full matrix - needed for TMA
                    # m_offset = (M_start + (tile_m_idx * BLOCK_SIZE_M)).to(tl.int32)
                    # k_offset = (tile_k_idx * BLOCK_SIZE_K).to(tl.int32)
                    # Offsets *relative* to the *current* expert -- m_start will then advance to this expert's start token
                    offs_am = tile_m_idx * BLOCK_SIZE_M + m_block_range

                    # [M, N] @ [N, K] -> [M, K] => Stride for A is N, stride for B is K
                    # We need two additional offsets:
                    # 1. For A, m_start to advance to this expert's start token
                    # 2. For B, n_start to advance to this expert's weights since we are passing in an [E, N, K] weight matrix
                    row_offsets_a = m_start + offs_am[:, None]
                    load_a_idx = row_offsets_a * N
                    store_idx = row_offsets_a * K
                    row_mask = offs_am[:, None] < m_size

                if not USE_TMA_LOAD_dY:
                    dY_ptrs = dY_ptr + load_a_idx + n_block_range[None, :]

                offs_bk = tile_k_idx * BLOCK_SIZE_K + k_block_range
                if not USE_TMA_LOAD_W:
                    row_offsets_b = n_start + n_block_range
                    # offs_bn = n_start + n_block_range
                    # row_offsets_b = tl.max_contiguous(tl.multiple_of(offs_bn, BLOCK_SIZE_N), BLOCK_SIZE_N)
                    w_ptrs = w_ptr + row_offsets_b[:, None] * K + offs_bk[None, :]

                # TODO: check whether predication along K is needed since we checked that K is divisible by BLOCK_SIZE_K in the forward kernel
                # col_mask = offs_bk[None, :] < K
                store_mask = row_mask  # & col_mask

                accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_K), dtype = tl.float32)

                # LoRA accumulator for dY @ B, stays in registers
                if FUSE_LORA:
                    dy_b_acc = tl.zeros((BLOCK_SIZE_M, BLOCK_R), dtype = tl.float32)
                    r_range = tl.arange(0, BLOCK_R)
                    r_mask = r_range < LORA_R
                    # B pointers: B[expert_idx] is [N, R], load [BLOCK_N, BLOCK_R]
                    b_expert_base = expert_idx * N * LORA_R
                    b_ptrs = (
                        lora_b_ptr
                        + b_expert_base
                        + n_block_range[:, None] * LORA_R
                        + r_range[None, :]
                    )

                # GEMM main loop
                for n_offset in range(0, N, BLOCK_SIZE_N):
                    # dY block [M, N]
                    if not USE_TMA_LOAD_dY:
                        dY = tl.load(dY_ptrs, mask = row_mask)
                    else:
                        dY = dY_desc.load(
                            [m_start + tile_m_idx * BLOCK_SIZE_M, n_offset]
                        )

                    if not USE_TMA_LOAD_W:
                        w = tl.load(w_ptrs)  # , mask=col_mask)
                    else:
                        w = w_desc.load(
                            [expert_idx, n_offset, tile_k_idx * BLOCK_SIZE_K]
                        )
                        w = tl.reshape(w, (BLOCK_SIZE_N, BLOCK_SIZE_K))

                    # [M, N] @ [N, K] -> [M, K]
                    dY = dY.to(w.dtype)
                    accumulator += tl.dot(dY, w)  # NOTE: no transpose of b

                    # Fused LoRA: accumulate dY @ B in the same N-loop
                    if FUSE_LORA:
                        b = tl.load(b_ptrs, mask = r_mask[None, :])
                        b = b.to(dY.dtype)
                        dy_b_acc += tl.dot(dY, b)  # [M,N] @ [N,R] -> [M,R]
                        b_ptrs += BLOCK_SIZE_N * LORA_R

                    # Advance A along contiguous dimension
                    if not USE_TMA_LOAD_dY:
                        dY_ptrs += BLOCK_SIZE_N
                    if not USE_TMA_LOAD_W:
                        w_ptrs += BLOCK_SIZE_N * K

                # LoRA epilogue: load A[expert] and compute (dY@B) @ A
                # A layout: (E, R, K), so A[e,r,k] = ptr + e*R*K + r*K + k
                if FUSE_LORA:
                    a_expert_base = expert_idx * LORA_R * K
                    a_ptrs = (
                        lora_a_ptr
                        + a_expert_base
                        + r_range[:, None] * K
                        + (tile_k_idx * BLOCK_SIZE_K + k_block_range)[None, :]
                    )
                    a = tl.load(a_ptrs, mask = r_mask[:, None])
                    a = a.to(dy_b_acc.dtype)
                    dy_b_cast = dy_b_acc.to(a.dtype)
                    lora_dx = tl.dot(dy_b_cast, a)  # [M,R] @ [R,K] -> [M,K]
                    accumulator += lora_scaling * lora_dx

                dX = accumulator.to(output_dtype)

                # Writing out a BLOCK_M x BLOCK_K tile, so we need to stride by K
                if USE_TMA_STORE:
                    offset_m = tile_m_idx * BLOCK_SIZE_M  # .to(tl.int32)
                    offset_k = tile_k_idx * BLOCK_SIZE_K  # .to(tl.int32)
                    dX_desc.store([m_start + offset_m, offset_k], dX)
                else:
                    tl.store(
                        dX_ptr + store_idx + offs_bk[None, :],
                        dX,
                        mask = store_mask,
                    )

                # Move to the next tile within this expert group
                tidx += NUM_SMS

            # Update the total tiles count for the next expert group
            processed_tiles += num_tiles_per_expert


_autotuned_grouped_gemm_dX_kernel = triton.autotune(
    configs = get_dX_kernel_configs(),
    prune_configs_by = {"early_config_prune": prune_dX_configs},
    # NOTE: NUM_TOKENS removed from key to avoid recompilation for every sequence length
    key = ["NUM_EXPERTS", "N", "K", "PERMUTE_X", "PERMUTE_Y", "FUSE_LORA", "LORA_R"],
)(_grouped_gemm_dX_kernel)

"""
notes on permute_x:
- for the first grouped GEMM, we permuted on load -> X was [num_tokens, K] and stored y in expert grouped order [num_tokens * topk, K]
- in the backwards pass, we need to permute on load of X while loading dy in contiguous (expert grouped) order
- since we are writing out dW, there is no need to permute on store

notes on permute_y:
- for the second grouped GEMM, we permuted on store -> y was permuted from expert grouped order to token order, x was loaded in expert grouped order since it was the output of the first grouped GEMM
- in the backwards pass, we need to permute on load of dy to get from token order to expert grouped order to match the order of X
- since we are writing out dW, there is no need to permute on store

notes on TMA loading:
- if we're TMA loading both X and dY, then we need to mask along the M dimension
to account for expert boundaries
- we can either
    - define TMA descriptors within the outer for loop to predicate loads
    or
    - mask along M after loading
"""


@triton.jit
def _grouped_gemm_dW_kernel(
    x_ptr,
    dY_ptr,
    dW_ptr,
    m_sizes_ptr,
    gather_indices_ptr,
    # problem sizes
    NUM_TOKENS,
    TOPK: tl.constexpr,
    NUM_EXPERTS: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    NUM_SMS,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    PERMUTE_X: tl.constexpr = False,
    PERMUTE_Y: tl.constexpr = False,
    USE_TMA_LOAD_dY: tl.constexpr = False,
    USE_TMA_LOAD_X: tl.constexpr = False,
    USE_TMA_STORE: tl.constexpr = False,
    FLATTEN: tl.constexpr = True,
    acc_dtype: tl.constexpr = tl.float32,
) -> None:
    TOTAL_TOKENS = NUM_TOKENS * TOPK
    TMA_LOAD_BOTH: tl.constexpr = USE_TMA_LOAD_X and USE_TMA_LOAD_dY

    tidx = tl.program_id(0)
    output_dtype = dW_ptr.dtype.element_ty

    if USE_TMA_LOAD_dY and not TMA_LOAD_BOTH:
        dY_desc = tl.make_tensor_descriptor(
            dY_ptr,
            shape = [TOTAL_TOKENS, N],
            strides = [N, 1],
            block_shape = [BLOCK_SIZE_M, BLOCK_SIZE_N],
        )

    if USE_TMA_LOAD_X and not TMA_LOAD_BOTH:
        x_desc = tl.make_tensor_descriptor(
            x_ptr,
            shape = [TOTAL_TOKENS, K],
            strides = [K, 1],
            block_shape = [BLOCK_SIZE_M, BLOCK_SIZE_K],
        )
    # Output tiles per expert, since each expert weight matrix is [N, K]
    num_n_tiles = tl.cdiv(N, BLOCK_SIZE_N)
    num_k_tiles = tl.cdiv(K, BLOCK_SIZE_K)
    output_tiles_per_expert = num_n_tiles * num_k_tiles

    block_range_m = tl.arange(0, BLOCK_SIZE_M)
    block_range_n = tl.arange(0, BLOCK_SIZE_N)
    block_range_k = tl.arange(0, BLOCK_SIZE_K)

    # NOTE: Important that N % BLOCK_SIZE_N == 0 and K % BLOCK_SIZE_K == 0 when using TMA store
    if USE_TMA_STORE:
        tl.static_assert(N % BLOCK_SIZE_N == 0, "N must be divisible by BLOCK_SIZE_N")
        tl.static_assert(K % BLOCK_SIZE_K == 0, "K must be divisible by BLOCK_SIZE_K")
        dW_desc = tl.make_tensor_descriptor(
            dW_ptr,
            shape = [NUM_EXPERTS, N, K],
            strides = [N * K, K, 1],
            block_shape = [1, BLOCK_SIZE_N, BLOCK_SIZE_K],
        )

    for tile_idx in range(
        tidx, output_tiles_per_expert, NUM_SMS
    ):  # , flatten=FLATTEN):
        # Output tile index
        tile_n_idx = tile_idx % num_n_tiles
        tile_k_idx = tile_idx // num_n_tiles

        # Output tile offsets
        n_offset = tile_n_idx * BLOCK_SIZE_N
        k_offset = tile_k_idx * BLOCK_SIZE_K

        # For storing
        # TODO: Check whether the k mask is needed since we statically check that K is divisible by BLOCK_SIZE_K in the forward kernel
        # ditto for n_mask
        n_mask = block_range_n + n_offset < N
        k_mask = block_range_k + k_offset < K
        nk_mask = n_mask[:, None] & k_mask[None, :]

        m_end = 0
        for expert_idx in range(NUM_EXPERTS):
            # We need to instantiate a fresh accumulator for each expert
            accumulator = tl.zeros((BLOCK_SIZE_N, BLOCK_SIZE_K), dtype = acc_dtype)

            m_start = m_end
            # Need to figure out why this cast is needed, otherwise compiler complains about mismatching types
            m_size = tl.load(m_sizes_ptr + expert_idx).to(tl.int32)
            m_end = m_start + m_size

            # NOTE: when storing the result, we need to offset by n_start since we are storing the result for this expert to the global [E, N, K] weight matrix
            n_start = expert_idx * N
            store_row_offs = n_start + n_offset + block_range_n

            if m_size > 0:
                if TMA_LOAD_BOTH:
                    dY_desc = tl.make_tensor_descriptor(
                        dY_ptr,
                        shape = [m_end, N],
                        strides = [N, 1],
                        block_shape = [BLOCK_SIZE_M, BLOCK_SIZE_N],
                    )

                    x_desc = tl.make_tensor_descriptor(
                        x_ptr,
                        shape = [m_end, K],
                        strides = [K, 1],
                        block_shape = [BLOCK_SIZE_M, BLOCK_SIZE_K],
                    )

                for tile_m_idx in range(0, m_size, BLOCK_SIZE_M):
                    m_block_size = tl.minimum(BLOCK_SIZE_M, m_size - tile_m_idx)

                    if m_block_size > 0:
                        # Global offset for this chunk
                        m_global_offset = m_start + tile_m_idx
                        m_offsets = m_global_offset + block_range_m

                        if PERMUTE_X or PERMUTE_Y:
                            # These will be used for loading and storing in permuted order
                            gather_offsets = (
                                tile_m_idx + block_range_m
                            )  # NOTE: tile_m_idx is already strided by BLOCK_SIZE_M

                            indices_to_gather = m_start + tl.max_contiguous(
                                tl.multiple_of(gather_offsets % m_size, BLOCK_SIZE_M),
                                BLOCK_SIZE_M,
                            )
                            # indices_to_gather = m_start + gather_offsets
                            expert_token_idx = tl.load(
                                gather_indices_ptr + indices_to_gather,
                                mask = indices_to_gather < TOTAL_TOKENS,
                            )
                            expert_token_offsets = expert_token_idx[:, None]

                            # Masks for permuted load and store
                            row_load_mask = gather_offsets < m_size

                            # We only take into account the following two cases: (PERMUTE_X and NOT PERMUTE_Y) and (NOT PERMUTE_X and PERMUTE_Y)
                            # Hence, we can make the following simplifying assumptions when loading and storing
                            # Note the different strides between the two cases: the offsets for loading and storing are flipped and the strides must also be adjusted
                            if PERMUTE_X:
                                x_row_load_idx = (
                                    (expert_token_offsets // TOPK) * K
                                )  # Permute on load from token -> expert order, divide by TOPK to index from original number of tokens
                                dY_row_load_idx = m_offsets[:, None] * N
                            else:
                                x_row_load_idx = (
                                    indices_to_gather[:, None] * K
                                )  # Load in contiguous order (no permutation on load)
                                dY_row_load_idx = expert_token_offsets * N

                        else:
                            x_row_load_idx = m_offsets[:, None] * K
                            dY_row_load_idx = m_offsets[:, None] * N
                            row_load_mask = block_range_m < m_block_size

                        mk_mask = row_load_mask[:, None] & k_mask[None, :]
                        mn_mask = row_load_mask[:, None] & n_mask[None, :]

                        if USE_TMA_LOAD_X:
                            x = x_desc.load([m_global_offset, k_offset])
                        else:
                            x = tl.load(
                                x_ptr
                                + x_row_load_idx
                                + (k_offset + block_range_k)[None, :],
                                mask = mk_mask,
                            )

                        if USE_TMA_LOAD_dY:
                            dY = dY_desc.load([m_global_offset, n_offset])
                        else:
                            dY = tl.load(
                                dY_ptr
                                + dY_row_load_idx
                                + (n_offset + block_range_n)[None, :],
                                mask = mn_mask,
                            )

                        accumulator += tl.dot(
                            dY.T.to(x.dtype),  # [BLOCK_N, BLOCK_M]
                            x,  # [BLOCK_M, BLOCK_K]
                        )

                y = accumulator.to(output_dtype)
                if USE_TMA_STORE:
                    # Need to expand dims to match [E, N, K] shape
                    y = tl.expand_dims(y, 0)
                    dW_desc.store([expert_idx, n_offset, k_offset], y)
                else:
                    tl.store(
                        dW_ptr
                        # + (n_offset + offs_n)[:, None] * K
                        + store_row_offs[:, None] * K
                        + (k_offset + block_range_k)[None, :],
                        y,
                        mask = nk_mask,
                    )


_autotuned_grouped_gemm_dW_kernel = triton.autotune(
    configs = get_dW_kernel_configs(),
    prune_configs_by = {"early_config_prune": prune_kernel_configs_backward_dW},
    # NOTE: NUM_TOKENS removed from key to avoid recompilation for every sequence length
    key = ["NUM_EXPERTS", "N", "K", "PERMUTE_X", "PERMUTE_Y"],
)(_grouped_gemm_dW_kernel)


def compute_lora_grads(
    dY: torch.Tensor,
    X: torch.Tensor,
    lora_A: torch.Tensor,
    lora_B: torch.Tensor,
    expert_offsets: torch.Tensor,
    num_experts: int,
    scaling: float,
) -> tuple:
    """Compute LoRA A and B gradients per-expert using PyTorch ops.

    dA_e = scaling * (dY_e @ B_e)^T @ X_e
    dB_e = scaling * dY_e^T @ (X_e @ A_e^T)

    Args:
        dY: (M_total, N) gradients, expert-grouped
        X: (M_total, K) inputs, expert-grouped
        lora_A: (E, R, K)
        lora_B: (E, N, R)
        expert_offsets: (E,) cumulative token counts
        num_experts: number of experts
        scaling: LoRA scaling factor

    Returns:
        dA: (E, R, K)
        dB: (E, N, R)
    """
    E = num_experts
    R = lora_A.shape[1]
    K = lora_A.shape[2]
    N = lora_B.shape[1]
    compute_dtype = dY.dtype

    dA = torch.zeros_like(lora_A)
    dB = torch.zeros_like(lora_B)

    prev_offset = 0
    for e in range(E):
        curr_offset = expert_offsets[e].item()
        if curr_offset > prev_offset:
            dy_e = dY[prev_offset:curr_offset]  # (M_e, N)
            x_e = X[prev_offset:curr_offset]    # (M_e, K)
            a_e = lora_A[e].to(compute_dtype)   # (R, K)
            b_e = lora_B[e].to(compute_dtype)   # (N, R)

            # dA_e = scaling * (dY_e @ B_e)^T @ X_e
            # (dY_e @ B_e): (M_e, N) @ (N, R) -> (M_e, R)
            dy_b = dy_e @ b_e  # (M_e, R)
            # (dy_b)^T @ X_e: (R, M_e) @ (M_e, K) -> (R, K)
            dA[e] = (scaling * (dy_b.T @ x_e)).to(lora_A.dtype)

            # dB_e = scaling * dY_e^T @ (X_e @ A_e^T)
            # (X_e @ A_e^T): (M_e, K) @ (K, R) -> (M_e, R)
            x_a = x_e @ a_e.T  # (M_e, R)
            # dY_e^T @ x_a: (N, M_e) @ (M_e, R) -> (N, R)
            dB[e] = (scaling * (dy_e.T @ x_a)).to(lora_B.dtype)

        prev_offset = curr_offset

    return dA, dB
