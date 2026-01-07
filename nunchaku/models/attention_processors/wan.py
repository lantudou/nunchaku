"""
Attention processors for Wan2.2 I2V quantized transformer.
"""

from typing import Optional, Tuple

import torch
from torch.nn import functional as F


class NunchakuWanAttnProcessor:
    """
    Attention processor for Wan2.2 self-attention (attn1).

    Uses fused QKV projection and RoPE embeddings.
    """

    def __call__(
        self,
        attn,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> torch.Tensor:
        """
        Forward pass for self-attention.

        Parameters
        ----------
        attn : NunchakuWanSelfAttention
            Self-attention module with fused QKV.
        hidden_states : torch.Tensor, shape (B, S, C)
            Input hidden states.
        encoder_hidden_states : Optional[torch.Tensor]
            Not used for self-attention. Must be None.
        attention_mask : Optional[torch.Tensor]
            Attention mask.
        rotary_emb : Optional[Tuple[torch.Tensor, torch.Tensor]]
            Rotary embeddings (freqs_cos, freqs_sin).

        Returns
        -------
        torch.Tensor, shape (B, S, C)
            Output hidden states.
        """
        batch_size, seq_len, _ = hidden_states.shape

        # Fused QKV projection
        qkv = attn.to_qkv(hidden_states)
        query, key, value = qkv.chunk(3, dim=-1)

        # Apply QK normalization
        query = attn.norm_q(query)
        key = attn.norm_k(key)

        # Reshape for multi-head attention: (B, S, H, D)
        query = query.unflatten(2, (attn.heads, -1))
        key = key.unflatten(2, (attn.heads, -1))
        value = value.unflatten(2, (attn.heads, -1))

        # Apply rotary embeddings
        if rotary_emb is not None:
            query = self._apply_rotary_emb(query, *rotary_emb)
            key = self._apply_rotary_emb(key, *rotary_emb)

        # Transpose for attention: (B, H, S, D)
        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)

        # Compute attention
        hidden_states = F.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        )

        # Reshape back: (B, S, C)
        hidden_states = hidden_states.transpose(1, 2).flatten(2, 3)
        hidden_states = hidden_states.to(query.dtype)

        # Output projection
        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)

        return hidden_states

    @staticmethod
    def _apply_rotary_emb(
        hidden_states: torch.Tensor,
        freqs_cos: torch.Tensor,
        freqs_sin: torch.Tensor,
    ) -> torch.Tensor:
        """Apply rotary positional embeddings."""
        x1, x2 = hidden_states.unflatten(-1, (-1, 2)).unbind(-1)
        cos = freqs_cos[..., 0::2]
        sin = freqs_sin[..., 1::2]
        out = torch.empty_like(hidden_states)
        out[..., 0::2] = x1 * cos - x2 * sin
        out[..., 1::2] = x1 * sin + x2 * cos
        return out.type_as(hidden_states)


class NunchakuWanCrossAttnProcessor:
    """
    Attention processor for Wan2.2 cross-attention (attn2).

    Handles both text-only attention and I2V image+text attention.
    For I2V, encoder_hidden_states contains [image_embeddings, text_embeddings],
    where the last 512 tokens are text embeddings.
    """

    def __call__(
        self,
        attn,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> torch.Tensor:
        """
        Forward pass for cross-attention.

        Parameters
        ----------
        attn : NunchakuWanCrossAttention
            Cross-attention module with separate Q and fused KV.
        hidden_states : torch.Tensor, shape (B, S, C)
            Input hidden states (query source).
        encoder_hidden_states : torch.Tensor, shape (B, S_enc, C)
            Encoder hidden states (key/value source).
            For I2V: [image_embeddings, text_embeddings] concatenated.
        attention_mask : Optional[torch.Tensor]
            Attention mask.
        rotary_emb : Optional[Tuple[torch.Tensor, torch.Tensor]]
            Not used for cross-attention.

        Returns
        -------
        torch.Tensor, shape (B, S, C)
            Output hidden states.
        """
        batch_size, seq_len, _ = hidden_states.shape

        # Handle I2V: split image and text embeddings
        encoder_hidden_states_img = None
        if attn.to_added_kv is not None:
            # 512 is the text encoder context length (hardcoded for now)
            image_context_length = encoder_hidden_states.shape[1] - 512
            encoder_hidden_states_img = encoder_hidden_states[:, :image_context_length]
            encoder_hidden_states = encoder_hidden_states[:, image_context_length:]

        # Q projection (from hidden_states)
        query = attn.to_q(hidden_states)

        # Fused KV projection (from encoder_hidden_states)
        kv = attn.to_kv(encoder_hidden_states)
        key, value = kv.chunk(2, dim=-1)

        # Apply QK normalization
        query = attn.norm_q(query)
        key = attn.norm_k(key)

        # Reshape for multi-head attention: (B, S, H, D)
        query = query.unflatten(2, (attn.heads, -1))
        key = key.unflatten(2, (attn.heads, -1))
        value = value.unflatten(2, (attn.heads, -1))

        # I2V: compute image attention
        hidden_states_img = None
        if encoder_hidden_states_img is not None:
            # Fused added KV projection for image embeddings
            added_kv = attn.to_added_kv(encoder_hidden_states_img)
            key_img, value_img = added_kv.chunk(2, dim=-1)
            key_img = attn.norm_added_k(key_img)

            key_img = key_img.unflatten(2, (attn.heads, -1))
            value_img = value_img.unflatten(2, (attn.heads, -1))

            # Transpose for attention: (B, H, S, D)
            query_img = query.transpose(1, 2)
            key_img = key_img.transpose(1, 2)
            value_img = value_img.transpose(1, 2)

            hidden_states_img = F.scaled_dot_product_attention(
                query_img, key_img, value_img, attn_mask=None, dropout_p=0.0, is_causal=False
            )
            hidden_states_img = hidden_states_img.transpose(1, 2).flatten(2, 3)
            hidden_states_img = hidden_states_img.type_as(query)

        # Transpose for attention: (B, H, S, D)
        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)

        # Compute text attention
        hidden_states = F.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        )
        hidden_states = hidden_states.transpose(1, 2).flatten(2, 3)
        hidden_states = hidden_states.to(query.dtype)

        # Add image attention output (I2V)
        if hidden_states_img is not None:
            hidden_states = hidden_states + hidden_states_img

        # Output projection
        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)

        return hidden_states
