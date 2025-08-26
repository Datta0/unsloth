# Copyright 2023-present Daniel Han-Chen & the Unsloth team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
import math
from typing import Optional, Tuple
from math import sqrt as math_sqrt

# Import flash attention and xformers directly to avoid circular imports
try:
    from flash_attn import flash_attn_func
    HAS_FLASH_ATTENTION = True
except:
    flash_attn_func = None
    HAS_FLASH_ATTENTION = False
pass

try:
    from xformers.ops import memory_efficient_attention as xformers_attention
    HAS_XFORMERS = True
except:
    xformers_attention = None
    HAS_XFORMERS = False
pass

# Check if SDPA has GQA support
from torch.nn.functional import scaled_dot_product_attention
SDPA_HAS_GQA = "enable_gqa" in scaled_dot_product_attention.__doc__

torch_matmul = torch.matmul
torch_nn_functional_softmax = torch.nn.functional.softmax


def compute_attention_matrix(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    is_causal: bool = False,
    n_groups: int = 1,
    n_kv_heads: int = 0,
    n_heads: int = 0,
    kv_seq_len: int = 0,
    sliding_window: Optional[int] = None,
    softmax_scale: Optional[float] = None,
    use_flash_attention: Optional[bool] = None,
    use_xformers: Optional[bool] = None,
) -> torch.Tensor:
    """
    Compute the attention matrix (Q @ K.T) @ V with various optimization options.
    
    Args:
        Q: Query tensor of shape [bsz, n_heads, q_len, head_dim]
        K: Key tensor of shape [bsz, n_kv_heads, kv_seq_len, head_dim]
        V: Value tensor of shape [bsz, n_kv_heads, kv_seq_len, head_dim]
        attention_mask: Optional attention mask
        is_causal: Whether to apply causal masking
        n_groups: Number of groups for grouped-query attention
        n_kv_heads: Number of key/value heads
        n_heads: Number of query heads
        kv_seq_len: Length of key/value sequences
        sliding_window: Optional sliding window size
        softmax_scale: Scaling factor for attention scores
        use_flash_attention: Whether to use Flash Attention (defaults to HAS_FLASH_ATTENTION)
        use_xformers: Whether to use xFormers (defaults to HAS_XFORMERS)
    
    Returns:
        Attention output tensor
    """
    # Set defaults for optional parameters
    if use_flash_attention is None:
        use_flash_attention = HAS_FLASH_ATTENTION
    if use_xformers is None:
        use_xformers = HAS_XFORMERS
    
    bsz, _, q_len, _ = Q.shape
    
    # Handle sliding windows
    if sliding_window is not None and kv_seq_len > sliding_window:
        slicing_tokens = kv_seq_len - sliding_window
        K = K[:, :, slicing_tokens:, :]
        V = V[:, :, slicing_tokens:, :]
        kv_seq_len = sliding_window
    pass
    
    # XFormers memory efficient attention
    if use_xformers and xformers_attention is not None and attention_mask is None:
        Q = Q.transpose(1, 2)
        K = K.transpose(1, 2)
        V = V.transpose(1, 2)
        
        # Grouped query attention
        if n_groups != 1:
            K = K[:, :, None, :, :].expand(bsz, n_kv_heads, n_groups, kv_seq_len, Q.shape[-1])
            V = V[:, :, None, :, :].expand(bsz, n_kv_heads, n_groups, kv_seq_len, Q.shape[-1])
            K = K.reshape(bsz, n_heads, kv_seq_len, Q.shape[-1])
            V = V.reshape(bsz, n_heads, kv_seq_len, Q.shape[-1])
        pass
        
        A = xformers_attention(Q, K, V, scale=softmax_scale)
        A = A.transpose(1, 2)
        return A
    pass
    
    # Flash Attention
    if use_flash_attention and flash_attn_func is not None and attention_mask is None:
        Q = Q.transpose(1, 2)
        K = K.transpose(1, 2)
        V = V.transpose(1, 2)
        
        window = (-1, -1) if (sliding_window is None or kv_seq_len <= sliding_window) else (sliding_window, sliding_window)
        A = flash_attn_func(Q, K, V, causal=is_causal, window_size=window, softmax_scale=softmax_scale)
        return A.transpose(1, 2)
    pass
    
    # Standard scaled dot-product attention with grouped-query attention
    if n_groups != 1:
        K = K[:, :, None, :, :].expand(bsz, n_kv_heads, n_groups, kv_seq_len, Q.shape[-1])
        V = V[:, :, None, :, :].expand(bsz, n_kv_heads, n_groups, kv_seq_len, Q.shape[-1])
        K = K.reshape(bsz, n_heads, kv_seq_len, Q.shape[-1])
        V = V.reshape(bsz, n_heads, kv_seq_len, Q.shape[-1])
    pass
    
    # Must be contiguous or else results are False!
    Q, K, V = Q.contiguous(), K.contiguous(), V.contiguous()
    
    # Needs (batch_size, n_heads, seq_len, head_dim)
    # is_causal and attention_mask must not be both set!
    if SDPA_HAS_GQA and n_groups != 1:
        A = scaled_dot_product_attention(
            Q, K, V, 
            attn_mask=attention_mask, 
            is_causal=is_causal, 
            scale=softmax_scale,
            enable_gqa=True
        )
    else:
        A = scaled_dot_product_attention(
            Q, K, V, 
            attn_mask=attention_mask, 
            is_causal=is_causal, 
            scale=softmax_scale
        )
    pass
    
    return A


def compute_attention_matrix_inference(
    Qn: torch.Tensor,
    Kn: torch.Tensor,
    Vn: torch.Tensor,
    K1: torch.Tensor,
    V1: torch.Tensor,
    attention: torch.Tensor,
    scalar: float,
    n_groups: int = 1,
    n_kv_heads: int = 0,
    n_heads: int = 0,
    cached_len: int = 0,
    sliding_window: Optional[int] = None,
    attention_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Compute the attention matrix (Q @ K.T) @ V for inference with KV cache.
    
    Args:
        Qn: Current query tensor of shape [bsz, n_heads, 1, head_dim]
        Kn: Current key tensor of shape [bsz, n_kv_heads, 1, head_dim]
        Vn: Current value tensor of shape [bsz, n_kv_heads, 1, head_dim]
        K1: Cached keys of shape [bsz, n_kv_heads, seq_len, head_dim]
        V1: Cached values of shape [bsz, n_kv_heads, seq_len, head_dim]
        attention: Pre-allocated attention buffer
        scalar: Scaling factor for attention scores (1/sqrt(head_dim))
        n_groups: Number of groups for grouped-query attention
        n_kv_heads: Number of key/value heads
        n_heads: Number of query heads
        cached_len: Length of cached keys/values
        sliding_window: Optional sliding window size
        attention_mask: Optional attention mask
    
    Returns:
        Attention output tensor
    """
    bsz = Qn.shape[0]
    
    # Handle sliding windows
    if sliding_window is not None and cached_len > sliding_window:
        slicing_tokens = cached_len - sliding_window
        Knn = Kn[:, :, slicing_tokens:, :]
        Vnn = Vn[:, :, slicing_tokens:, :]
        cached_len = sliding_window
    else:
        Knn, Vnn = Kn, Vn
    pass
    
    # Grouped query attention
    if n_groups != 1:
        Knn = Knn[:, :, None, :, :].expand(bsz, n_kv_heads, n_groups, cached_len, Qn.shape[-1])
        Vnn = Vnn[:, :, None, :, :].expand(bsz, n_kv_heads, n_groups, cached_len, Qn.shape[-1])
        Knn = Knn.reshape(bsz, n_heads, cached_len, Qn.shape[-1])
        Vnn = Vnn.reshape(bsz, n_heads, cached_len, Qn.shape[-1])
    pass
    
    # Single batch optimized path
    if bsz == 1:
        Qn *= scalar  # See https://github.com/ggerganov/llama.cpp/issues/7805#issuecomment-2153349963
        # It seems like doing (Q * scalar) @ K is better than (Q @ K) * scalar to stop overflows
        A = torch_matmul(Qn, Knn.transpose(2, 3), out=attention[:, :, :, :cached_len])
        # if attention_mask is not None: A += attention_mask # Must add attention_mask for batched
        A[:] = torch_nn_functional_softmax(A, dim=-1, dtype=torch.float32)  # .to(A.dtype)
        A = torch_matmul(A, Vnn, out=Qn)
    else:
        if SDPA_HAS_GQA and n_groups != 1:
            A = scaled_dot_product_attention(
                Qn, Knn, Vnn, 
                attn_mask=attention_mask, 
                is_causal=False, 
                enable_gqa=True
            )
        else:
            A = scaled_dot_product_attention(
                Qn, Knn, Vnn, 
                attn_mask=attention_mask, 
                is_causal=False
            )
        pass
    pass
    
    return A


def reshape_for_grouped_query_attention(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    n_groups: int,
    n_kv_heads: int,
    n_heads: int,
    kv_seq_len: int,
    requires_grad: bool = False
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Reshape tensors for grouped-query attention.
    
    Args:
        Q: Query tensor
        K: Key tensor
        V: Value tensor
        n_groups: Number of groups
        n_kv_heads: Number of key/value heads
        n_heads: Number of query heads
        kv_seq_len: Key/value sequence length
        requires_grad: Whether gradients are required
    
    Returns:
        Reshaped Q, K, V tensors
    """
    bsz = K.shape[0]
    
    if n_groups != 1:
        K = K[:, :, None, :, :].expand(bsz, n_kv_heads, n_groups, kv_seq_len, K.shape[-1])
        V = V[:, :, None, :, :].expand(bsz, n_kv_heads, n_groups, kv_seq_len, V.shape[-1])
        K = K.reshape(bsz, n_heads, kv_seq_len, K.shape[-1])
        V = V.reshape(bsz, n_heads, kv_seq_len, V.shape[-1])
    pass
    
    # Must be contiguous or else results are False!
    # https://github.com/pytorch/pytorch/issues/112577
    Q, K, V = Q.contiguous(), K.contiguous(), V.contiguous()
    
    return Q, K, V


def scaled_dot_product_attention_softcap(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    attn_mask: Optional[torch.Tensor] = None,
    is_causal: bool = False,
    scale: Optional[float] = None,
    softcap: Optional[float] = None,
) -> torch.Tensor:
    """
    Computes scaled dot product attention with optional softcapping (used in Gemma2)
    
    Args:
        Q: Query tensor
        K: Key tensor
        V: Value tensor
        attn_mask: Optional attention mask
        is_causal: Whether to apply causal mask
        scale: Scaling factor (default: 1/sqrt(head_dim))
        softcap: Softcap value for attention scores
    
    Returns:
        Attention output tensor
    """
    from torch.nn.functional import softmax
    
    # Compute scale if not provided
    if scale is None:
        scale = 1.0 / (Q.size(-1) ** 0.5)
    
    # Compute attention scores
    attn_scores = torch.matmul(Q, K.transpose(-2, -1)) * scale
    
    # Apply softcapping if specified
    if softcap is not None:
        attn_scores = attn_scores / softcap
        attn_scores = torch.tanh(attn_scores)
        attn_scores = attn_scores * softcap
    
    # Apply mask
    if attn_mask is not None:
        attn_scores += attn_mask
    
    # Apply causal mask if needed
    if is_causal:
        causal_mask = torch.triu(torch.ones_like(attn_scores[0]), diagonal=1).bool()
        attn_scores.masked_fill_(causal_mask, float('-inf'))
    
    # Apply softmax
    attn_weights = softmax(attn_scores, dim=-1)
    
    # Apply attention weights to values
    return torch.matmul(attn_weights, V)