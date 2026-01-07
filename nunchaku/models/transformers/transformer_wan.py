"""
Nunchaku quantized Wan2.2 I2V Transformer model.

This module provides SVD W4A4 quantized implementations of the Wan2.2 video generation
transformer for image-to-video (I2V) tasks.
"""

import gc
import json
import math
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
from warnings import warn

import torch
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from diffusers.models.normalization import FP32LayerNorm
from diffusers.models.transformers.transformer_wan import (
    WanRotaryPosEmbed,
    WanTimeTextImageEmbedding,
    WanTransformer3DModel,
    WanTransformerBlock,
)
from huggingface_hub import utils
from torch import nn

from ...utils import get_precision, load_state_dict_in_safetensors
from ..attention import NunchakuBaseAttention, NunchakuFeedForward
from ..attention_processors.wan import NunchakuWanAttnProcessor, NunchakuWanCrossAttnProcessor
from ..linear import SVDQW4A4Linear
from ..utils import CPUOffloadManager, fuse_linears
from .utils import NunchakuModelLoaderMixin


class NunchakuWanSelfAttention(NunchakuBaseAttention):
    """
    Nunchaku-optimized quantized self-attention module for Wan2.2.

    Uses fused QKV projection for efficient W4A4 quantized inference.

    Parameters
    ----------
    dim : int
        Input/output feature dimension.
    heads : int
        Number of attention heads.
    dim_head : int
        Dimension per attention head.
    eps : float, optional
        Epsilon for RMSNorm. Default is 1e-5.
    dropout : float, optional
        Dropout probability. Default is 0.0.
    processor : str, optional
        Attention processor name. Default is "flashattn2".
    **kwargs
        Additional arguments for quantization.
    """

    def __init__(
        self,
        dim: int,
        heads: int,
        dim_head: int,
        eps: float = 1e-5,
        dropout: float = 0.0,
        processor: str = "flashattn2",
        **kwargs,
    ):
        super().__init__(processor)
        self.inner_dim = dim_head * heads
        self.heads = heads
        self.head_dim = dim_head

        # Fused QKV projection
        with torch.device("meta"):
            to_qkv = nn.Linear(dim, self.inner_dim * 3, bias=True)
        self.to_qkv = SVDQW4A4Linear.from_linear(to_qkv, **kwargs)

        # QK normalization
        self.norm_q = nn.RMSNorm(dim_head * heads, eps=eps, elementwise_affine=True)
        self.norm_k = nn.RMSNorm(dim_head * heads, eps=eps, elementwise_affine=True)

        # Output projection
        with torch.device("meta"):
            to_out_linear = nn.Linear(self.inner_dim, dim, bias=True)
        self.to_out = nn.ModuleList([
            SVDQW4A4Linear.from_linear(to_out_linear, **kwargs),
            nn.Dropout(dropout),
        ])

    def set_processor(self, processor: str):
        """Set the attention processor."""
        if processor == "flashattn2":
            self.processor = NunchakuWanAttnProcessor()
        else:
            raise ValueError(f"Processor {processor} is not supported")

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> torch.Tensor:
        """Forward pass for self-attention."""
        return self.processor(self, hidden_states, encoder_hidden_states, attention_mask, rotary_emb)


class NunchakuWanCrossAttention(NunchakuBaseAttention):
    """
    Nunchaku-optimized quantized cross-attention module for Wan2.2.

    Uses separate Q projection and fused KV projection.
    For I2V models, includes additional fused KV projection for image embeddings.

    Parameters
    ----------
    dim : int
        Input/output feature dimension.
    heads : int
        Number of attention heads.
    dim_head : int
        Dimension per attention head.
    eps : float, optional
        Epsilon for RMSNorm. Default is 1e-5.
    dropout : float, optional
        Dropout probability. Default is 0.0.
    added_kv_proj_dim : Optional[int], optional
        Dimension for additional KV projection (I2V). Default is None.
    processor : str, optional
        Attention processor name. Default is "flashattn2".
    **kwargs
        Additional arguments for quantization.
    """

    def __init__(
        self,
        dim: int,
        heads: int,
        dim_head: int,
        eps: float = 1e-5,
        dropout: float = 0.0,
        added_kv_proj_dim: Optional[int] = None,
        processor: str = "flashattn2",
        **kwargs,
    ):
        super().__init__(processor)
        self.inner_dim = dim_head * heads
        self.heads = heads
        self.head_dim = dim_head
        self.added_kv_proj_dim = added_kv_proj_dim

        # Q projection (separate, from hidden_states)
        with torch.device("meta"):
            to_q = nn.Linear(dim, self.inner_dim, bias=True)
        self.to_q = SVDQW4A4Linear.from_linear(to_q, **kwargs)

        # Fused KV projection (from encoder_hidden_states)
        with torch.device("meta"):
            to_kv = nn.Linear(dim, self.inner_dim * 2, bias=True)
        self.to_kv = SVDQW4A4Linear.from_linear(to_kv, **kwargs)

        # QK normalization
        self.norm_q = nn.RMSNorm(dim_head * heads, eps=eps, elementwise_affine=True)
        self.norm_k = nn.RMSNorm(dim_head * heads, eps=eps, elementwise_affine=True)

        # I2V: additional fused KV projection for image embeddings
        self.to_added_kv = None
        self.norm_added_k = None
        if added_kv_proj_dim is not None:
            with torch.device("meta"):
                to_added_kv = nn.Linear(added_kv_proj_dim, self.inner_dim * 2, bias=True)
            self.to_added_kv = SVDQW4A4Linear.from_linear(to_added_kv, **kwargs)
            self.norm_added_k = nn.RMSNorm(dim_head * heads, eps=eps)

        # Output projection
        with torch.device("meta"):
            to_out_linear = nn.Linear(self.inner_dim, dim, bias=True)
        self.to_out = nn.ModuleList([
            SVDQW4A4Linear.from_linear(to_out_linear, **kwargs),
            nn.Dropout(dropout),
        ])

    def set_processor(self, processor: str):
        """Set the attention processor."""
        if processor == "flashattn2":
            self.processor = NunchakuWanCrossAttnProcessor()
        else:
            raise ValueError(f"Processor {processor} is not supported")

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> torch.Tensor:
        """Forward pass for cross-attention."""
        return self.processor(self, hidden_states, encoder_hidden_states, attention_mask, rotary_emb)


class NunchakuWanTransformerBlock(nn.Module):
    """
    Quantized Wan2.2 Transformer Block.

    Contains quantized self-attention, cross-attention, and feed-forward layers.

    Parameters
    ----------
    block : WanTransformerBlock
        The original block to wrap and quantize.
    **kwargs
        Additional arguments for quantization.
    """

    def __init__(
        self,
        block: WanTransformerBlock,
        **kwargs,
    ):
        super().__init__()

        dim = block.attn1.to_q.in_features
        num_heads = block.attn1.heads
        dim_head = dim // num_heads
        eps = block.norm1.eps
        added_kv_proj_dim = getattr(block.attn2, 'add_k_proj', None)
        if added_kv_proj_dim is not None:
            added_kv_proj_dim = added_kv_proj_dim.in_features

        # 1. Self-attention - reuse norm from original block
        self.norm1 = block.norm1
        self.attn1 = NunchakuWanSelfAttention(
            dim=dim,
            heads=num_heads,
            dim_head=dim_head,
            eps=eps,
            **kwargs,
        )
        # Copy attention norms from original block
        self.attn1.norm_q = block.attn1.norm_q
        self.attn1.norm_k = block.attn1.norm_k

        # 2. Cross-attention - reuse norm from original block
        self.attn2 = NunchakuWanCrossAttention(
            dim=dim,
            heads=num_heads,
            dim_head=dim_head,
            eps=eps,
            added_kv_proj_dim=added_kv_proj_dim,
            **kwargs,
        )
        self.norm2 = block.norm2
        # Copy attention norms from original block
        self.attn2.norm_q = block.attn2.norm_q
        self.attn2.norm_k = block.attn2.norm_k
        if self.attn2.norm_added_k is not None and hasattr(block.attn2, 'norm_added_k'):
            self.attn2.norm_added_k = block.attn2.norm_added_k

        # 3. Feed-forward - reuse norm and wrap ffn
        self.norm3 = block.norm3
        self.ffn = NunchakuFeedForward(block.ffn, **kwargs)

        # Reuse scale_shift_table from original block
        self.scale_shift_table = block.scale_shift_table

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        temb: torch.Tensor,
        rotary_emb: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass for the transformer block.

        Parameters
        ----------
        hidden_states : torch.Tensor
            Input hidden states.
        encoder_hidden_states : torch.Tensor
            Encoder hidden states for cross-attention.
        temb : torch.Tensor
            Timestep embedding for modulation.
        rotary_emb : torch.Tensor
            Rotary position embeddings.

        Returns
        -------
        torch.Tensor
            Output hidden states.
        """
        if temb.ndim == 4:
            # temb: batch_size, seq_len, 6, inner_dim (wan2.2 ti2v)
            shift_msa, scale_msa, gate_msa, c_shift_msa, c_scale_msa, c_gate_msa = (
                self.scale_shift_table.unsqueeze(0) + temb.float()
            ).chunk(6, dim=2)
            shift_msa = shift_msa.squeeze(2)
            scale_msa = scale_msa.squeeze(2)
            gate_msa = gate_msa.squeeze(2)
            c_shift_msa = c_shift_msa.squeeze(2)
            c_scale_msa = c_scale_msa.squeeze(2)
            c_gate_msa = c_gate_msa.squeeze(2)
        else:
            # temb: batch_size, 6, inner_dim (wan2.1/wan2.2 14B)
            shift_msa, scale_msa, gate_msa, c_shift_msa, c_scale_msa, c_gate_msa = (
                self.scale_shift_table + temb.float()
            ).chunk(6, dim=1)

        # 1. Self-attention
        norm_hidden_states = (self.norm1(hidden_states.float()) * (1 + scale_msa) + shift_msa).type_as(hidden_states)
        attn_output = self.attn1(norm_hidden_states, None, None, rotary_emb)
        hidden_states = (hidden_states.float() + attn_output * gate_msa).type_as(hidden_states)

        # 2. Cross-attention
        norm_hidden_states = self.norm2(hidden_states.float()).type_as(hidden_states)
        attn_output = self.attn2(norm_hidden_states, encoder_hidden_states, None, None)
        hidden_states = hidden_states + attn_output

        # 3. Feed-forward
        norm_hidden_states = (self.norm3(hidden_states.float()) * (1 + c_scale_msa) + c_shift_msa).type_as(
            hidden_states
        )
        ff_output = self.ffn(norm_hidden_states)
        hidden_states = (hidden_states.float() + ff_output.float() * c_gate_msa).type_as(hidden_states)

        return hidden_states


class NunchakuWanTransformer3DModel(WanTransformer3DModel, NunchakuModelLoaderMixin):
    """
    Quantized Wan2.2 I2V Transformer Model.

    Supports SVD W4A4 quantized inference and optional CPU offloading.

    Parameters
    ----------
    *args
        Positional arguments for WanTransformer3DModel.
    **kwargs
        Keyword arguments for WanTransformer3DModel and quantization.

    Attributes
    ----------
    offload : bool
        Whether CPU offloading is enabled.
    offload_manager : CPUOffloadManager or None
        Manager for offloading transformer blocks.
    _is_initialized : bool
        Whether the model has been patched for quantization.
    """

    def __init__(self, *args, **kwargs):
        self.offload = kwargs.pop("offload", False)
        self.offload_manager = None
        self._is_initialized = False
        super().__init__(*args, **kwargs)

    @classmethod
    @utils.validate_hf_hub_args
    def from_pretrained(cls, pretrained_model_name_or_path: str | os.PathLike[str], **kwargs):
        """
        Load a quantized model from a pretrained checkpoint.

        Parameters
        ----------
        pretrained_model_name_or_path : str or os.PathLike
            Path to the pretrained model checkpoint (safetensors file).
        **kwargs
            Additional arguments for loading and quantization.

        Returns
        -------
        NunchakuWanTransformer3DModel
            The loaded and quantized model.
        """
        device = kwargs.get("device", "cpu")
        offload = kwargs.get("offload", False)
        torch_dtype = kwargs.get("torch_dtype", torch.bfloat16)

        if isinstance(pretrained_model_name_or_path, str):
            pretrained_model_name_or_path = Path(pretrained_model_name_or_path)

        assert pretrained_model_name_or_path.is_file() and pretrained_model_name_or_path.name.endswith(
            (".safetensors", ".sft")
        ), "Only safetensors are supported"

        state_dict, metadata = load_state_dict_in_safetensors(pretrained_model_name_or_path, return_metadata=True)
        config = json.loads(metadata.get("config", "{}"))
        quantization_config = json.loads(metadata.get("quantization_config", "{}"))
        rank = quantization_config.get("rank", 32)

        # Remap keys: ffn.net.2.linear.* -> ffn.net.2.*
        remapped_state_dict = {}
        for k, v in state_dict.items():
            new_key = k.replace(".ffn.net.2.linear.", ".ffn.net.2.")
            remapped_state_dict[new_key] = v
        state_dict = remapped_state_dict

        # Build model from config - use direct instantiation to avoid config issues
        init_kwargs = {
            "in_channels": config.get("in_channels", 16),
            "out_channels": config.get("out_channels", 16),
            "num_attention_heads": config.get("num_attention_heads", 40),
            "attention_head_dim": config.get("attention_head_dim", 128),
            "num_layers": config.get("num_layers", 40),
            "ffn_dim": config.get("ffn_dim", 13824),
            "text_dim": config.get("text_dim", 4096),
            "freq_dim": config.get("freq_dim", 256),
            "cross_attn_norm": config.get("cross_attn_norm", True),
            "qk_norm": config.get("qk_norm", "rms_norm_across_heads"),
            "eps": config.get("eps", 1e-6),
            "patch_size": tuple(config.get("patch_size", (1, 2, 2))),
            "rope_max_seq_len": config.get("rope_max_seq_len", 1024),
            "image_dim": config.get("image_dim", None),
            "added_kv_proj_dim": config.get("added_kv_proj_dim", None),
            "pos_embed_seq_len": config.get("pos_embed_seq_len", None),
        }
        with torch.device("meta"):
            transformer = cls(**init_kwargs).to(torch_dtype)

        precision = get_precision()
        if precision == "fp4":
            precision = "nvfp4"

        transformer._patch_model(precision=precision, rank=rank)
        transformer = transformer.to_empty(device=device)

        # Re-initialize rope as to_empty does not work on it
        transformer.rope = WanRotaryPosEmbed(
            attention_head_dim=config.get("attention_head_dim", 128),
            patch_size=tuple(config.get("patch_size", (1, 2, 2))),
            max_seq_len=config.get("rope_max_seq_len", 1024),
        )
        transformer.rope = transformer.rope.to(device)

        # Handle missing keys
        model_state_dict = transformer.state_dict()
        for k in model_state_dict.keys():
            if k not in state_dict:
                if ".wcscales" in k:
                    state_dict[k] = torch.ones_like(model_state_dict[k])
                elif ".scale_shift_table" in k:
                    # scale_shift_table may not be saved in quantized checkpoint
                    # Initialize with zeros (no shift/scale effect initially)
                    state_dict[k] = torch.zeros_like(model_state_dict[k])
                    warn(f"Missing {k} in checkpoint, using zero initialization")
                else:
                    raise KeyError(f"Missing key in checkpoint: {k}")

        # Load wtscale from state dict (float on CPU)
        for n, m in transformer.named_modules():
            if isinstance(m, SVDQW4A4Linear):
                if m.wtscale is not None:
                    m.wtscale = state_dict.pop(f"{n}.wtscale", 1.0)

        transformer.load_state_dict(state_dict)
        transformer.set_offload(offload)

        return transformer

    def _patch_model(self, **kwargs):
        """
        Patch the model with quantized transformer blocks.

        Parameters
        ----------
        **kwargs
            Additional arguments for quantization (precision, rank, etc.).

        Returns
        -------
        self
        """
        # Replace blocks with quantized versions
        new_blocks = nn.ModuleList()
        for i, block in enumerate(self.blocks):
            new_block = NunchakuWanTransformerBlock(block, **kwargs)
            new_blocks.append(new_block)

        self.blocks = new_blocks
        self._is_initialized = True
        return self

    def set_offload(self, offload: bool, **kwargs):
        """
        Enable or disable CPU offloading for transformer blocks.

        Parameters
        ----------
        offload : bool
            Whether to enable offloading.
        **kwargs
            Additional arguments for CPUOffloadManager.
        """
        if offload == self.offload:
            return
        self.offload = offload
        if offload:
            self.offload_manager = CPUOffloadManager(
                self.blocks,
                use_pin_memory=kwargs.get("use_pin_memory", True),
                on_gpu_modules=[
                    self.rope,
                    self.patch_embedding,
                    self.condition_embedder,
                    self.norm_out,
                    self.proj_out,
                ],
                num_blocks_on_gpu=kwargs.get("num_blocks_on_gpu", 1),
            )
        else:
            self.offload_manager = None
            gc.collect()
            torch.cuda.empty_cache()

    def forward(
        self,
        hidden_states: torch.Tensor,
        timestep: torch.LongTensor,
        encoder_hidden_states: torch.Tensor,
        encoder_hidden_states_image: Optional[torch.Tensor] = None,
        return_dict: bool = True,
        attention_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Union[torch.Tensor, Transformer2DModelOutput]:
        """
        Forward pass for the quantized Wan2.2 transformer.

        Parameters
        ----------
        hidden_states : torch.Tensor
            Input latent tensor of shape (B, C, T, H, W).
        timestep : torch.LongTensor
            Timestep tensor.
        encoder_hidden_states : torch.Tensor
            Text encoder hidden states.
        encoder_hidden_states_image : Optional[torch.Tensor], optional
            Image encoder hidden states for I2V. Default is None.
        return_dict : bool, optional
            Whether to return a dict. Default is True.
        attention_kwargs : Optional[Dict[str, Any]], optional
            Additional attention arguments. Default is None.

        Returns
        -------
        torch.Tensor or Transformer2DModelOutput
            Output tensor or model output.
        """
        device = hidden_states.device
        if self.offload:
            self.offload_manager.set_device(device)

        batch_size, num_channels, num_frames, height, width = hidden_states.shape
        p_t, p_h, p_w = self.config.patch_size
        post_patch_num_frames = num_frames // p_t
        post_patch_height = height // p_h
        post_patch_width = width // p_w

        rotary_emb = self.rope(hidden_states)

        hidden_states = self.patch_embedding(hidden_states)
        hidden_states = hidden_states.flatten(2).transpose(1, 2)

        # Handle timestep shape
        if timestep.ndim == 2:
            ts_seq_len = timestep.shape[1]
            timestep = timestep.flatten()
        else:
            ts_seq_len = None

        temb, timestep_proj, encoder_hidden_states, encoder_hidden_states_image = self.condition_embedder(
            timestep, encoder_hidden_states, encoder_hidden_states_image, timestep_seq_len=ts_seq_len
        )

        if ts_seq_len is not None:
            timestep_proj = timestep_proj.unflatten(2, (6, -1))
        else:
            timestep_proj = timestep_proj.unflatten(1, (6, -1))

        # Concatenate image and text embeddings for I2V
        if encoder_hidden_states_image is not None:
            encoder_hidden_states = torch.concat([encoder_hidden_states_image, encoder_hidden_states], dim=1)

        # Transformer blocks
        compute_stream = torch.cuda.current_stream()
        if self.offload:
            self.offload_manager.initialize(compute_stream)

        for block_idx, block in enumerate(self.blocks):
            with torch.cuda.stream(compute_stream):
                if self.offload:
                    block = self.offload_manager.get_block(block_idx)

                if torch.is_grad_enabled() and self.gradient_checkpointing:
                    hidden_states = self._gradient_checkpointing_func(
                        block, hidden_states, encoder_hidden_states, timestep_proj, rotary_emb
                    )
                else:
                    hidden_states = block(hidden_states, encoder_hidden_states, timestep_proj, rotary_emb)

            if self.offload:
                self.offload_manager.step(compute_stream)

        # Output norm, projection & unpatchify
        if temb.ndim == 3:
            shift, scale = (self.scale_shift_table.unsqueeze(0).to(temb.device) + temb.unsqueeze(2)).chunk(2, dim=2)
            shift = shift.squeeze(2)
            scale = scale.squeeze(2)
        else:
            shift, scale = (self.scale_shift_table.to(temb.device) + temb.unsqueeze(1)).chunk(2, dim=1)

        shift = shift.to(hidden_states.device)
        scale = scale.to(hidden_states.device)

        hidden_states = (self.norm_out(hidden_states.float()) * (1 + scale) + shift).type_as(hidden_states)
        hidden_states = self.proj_out(hidden_states)

        hidden_states = hidden_states.reshape(
            batch_size, post_patch_num_frames, post_patch_height, post_patch_width, p_t, p_h, p_w, -1
        )
        hidden_states = hidden_states.permute(0, 7, 1, 4, 2, 5, 3, 6)
        output = hidden_states.flatten(6, 7).flatten(4, 5).flatten(2, 3)

        if not return_dict:
            return (output,)

        return Transformer2DModelOutput(sample=output)

    def to(self, *args, **kwargs):
        """
        Override .to() to handle offload mode and prevent dtype changes after quantization.
        """
        device_arg_or_kwarg_present = any(isinstance(arg, torch.device) for arg in args) or "device" in kwargs
        dtype_present_in_args = "dtype" in kwargs

        for arg in args:
            if not isinstance(arg, str):
                continue
            try:
                torch.device(arg)
                device_arg_or_kwarg_present = True
            except RuntimeError:
                pass

        if not dtype_present_in_args:
            for arg in args:
                if isinstance(arg, torch.dtype):
                    dtype_present_in_args = True
                    break

        if dtype_present_in_args and self._is_initialized:
            raise ValueError(
                "Casting a quantized model to a new `dtype` is unsupported. To set the dtype of unquantized layers, "
                "use the `torch_dtype` argument when loading the model using `from_pretrained`."
            )
        if self.offload:
            if device_arg_or_kwarg_present:
                warn("Skipping moving the model to GPU as offload is enabled", UserWarning)
                return self
        return super(type(self), self).to(*args, **kwargs)


def get_wan_state_dict_mapping_config() -> Dict[str, Any]:
    """
    Get the state dict mapping configuration for Wan2.2 I2V model.

    This configuration is used during quantization to map original layer names
    to nunchaku's fused layer structure.

    Returns
    -------
    dict
        Configuration dictionary containing:
        - local_name_map: Maps nunchaku layer names to original layer names
        - smooth_name_map: Maps layers to their smoothing reference layers
        - branch_name_map: Maps layers to their SVD branch reference layers
        - convert_map: Maps layers to their quantization type
    """
    local_name_map = {
        # Self-attention (attn1): Q/K/V fused
        "attn1.to_qkv": ["attn1.to_q", "attn1.to_k", "attn1.to_v"],
        "attn1.norm_q": "attn1.norm_q",
        "attn1.norm_k": "attn1.norm_k",
        "attn1.to_out.0": "attn1.to_out.0",
        # Cross-attention (attn2): Q separate, K/V fused
        "attn2.to_q": "attn2.to_q",
        "attn2.to_kv": ["attn2.to_k", "attn2.to_v"],
        "attn2.norm_q": "attn2.norm_q",
        "attn2.norm_k": "attn2.norm_k",
        "attn2.to_out.0": "attn2.to_out.0",
        # I2V: additional KV for image embeddings
        "attn2.to_added_kv": ["attn2.add_k_proj", "attn2.add_v_proj"],
        "attn2.norm_added_k": "attn2.norm_added_k",
        # Feed-forward network
        "ffn.net.0.proj": "ffn.net.0.proj",
        "ffn.net.2": "ffn.net.2",  # Some models use ffn.net.2.linear
    }

    smooth_name_map = {
        # Self-attention smooth: Q is reference for QKV
        "attn1.to_qkv": "attn1.to_q",
        "attn1.to_out.0": "attn1.to_out.0",
        # Cross-attention smooth: Q separate, K is reference for KV
        "attn2.to_q": "attn2.to_q",
        "attn2.to_kv": "attn2.to_k",
        "attn2.to_out.0": "attn2.to_out.0",
        # I2V: K is reference for added KV
        "attn2.to_added_kv": "attn2.add_k_proj",
        # FFN smooth
        "ffn.net.0.proj": "ffn.net.0.proj",
        "ffn.net.2": "ffn.net.2",
    }

    branch_name_map = {
        # Self-attention branches
        "attn1.to_qkv": "attn1.to_q",
        "attn1.to_out.0": "attn1.to_out.0",
        # Cross-attention branches
        "attn2.to_q": "attn2.to_q",
        "attn2.to_kv": "attn2.to_k",
        "attn2.to_out.0": "attn2.to_out.0",
        # I2V branches
        "attn2.to_added_kv": "attn2.add_k_proj",
        # FFN branches
        "ffn.net.0.proj": "ffn.net.0.proj",
        "ffn.net.2": "ffn.net.2",
    }

    convert_map = {
        "attn1.to_qkv": "linear",      # Self-attention QKV
        "attn1.to_out.0": "linear",    # Self-attention output
        "attn2.to_q": "linear",        # Cross-attention Q
        "attn2.to_kv": "linear",       # Cross-attention KV
        "attn2.to_out.0": "linear",    # Cross-attention output
        "attn2.to_added_kv": "linear", # I2V additional KV
        "ffn.net.0.proj": "linear",    # FFN up projection
        "ffn.net.2": "linear",         # FFN down projection
    }

    return {
        "local_name_map": local_name_map,
        "smooth_name_map": smooth_name_map,
        "branch_name_map": branch_name_map,
        "convert_map": convert_map,
    }


def convert_wan_state_dict_for_nunchaku(
    state_dict: Dict[str, torch.Tensor],
    num_blocks: int,
    has_added_kv: bool = True,
) -> Dict[str, torch.Tensor]:
    """
    Convert original Wan2.2 state dict to nunchaku's fused layer format.

    This function handles the QKV fusion for self-attention and KV fusion
    for cross-attention, preparing the state dict for quantization.

    Parameters
    ----------
    state_dict : dict
        Original Wan2.2 model state dict.
    num_blocks : int
        Number of transformer blocks.
    has_added_kv : bool, optional
        Whether the model has additional KV projections (I2V). Default is True.

    Returns
    -------
    dict
        Converted state dict with fused projections.

    Notes
    -----
    Non-quantized layers (norms, embeddings, etc.) are copied as-is.
    """
    new_state_dict = {}

    for key, value in state_dict.items():
        # Check if this is a block layer
        if key.startswith("blocks."):
            parts = key.split(".")
            block_idx = parts[1]
            layer_path = ".".join(parts[2:])

            # Handle self-attention QKV fusion
            if layer_path.startswith("attn1.to_q."):
                suffix = layer_path[len("attn1.to_q."):]
                q_key = f"blocks.{block_idx}.attn1.to_q.{suffix}"
                k_key = f"blocks.{block_idx}.attn1.to_k.{suffix}"
                v_key = f"blocks.{block_idx}.attn1.to_v.{suffix}"

                if q_key in state_dict and k_key in state_dict and v_key in state_dict:
                    # Concatenate Q, K, V weights/biases
                    new_key = f"blocks.{block_idx}.attn1.to_qkv.{suffix}"
                    if new_key not in new_state_dict:
                        new_state_dict[new_key] = torch.cat([
                            state_dict[q_key],
                            state_dict[k_key],
                            state_dict[v_key],
                        ], dim=0)
            elif layer_path.startswith("attn1.to_k.") or layer_path.startswith("attn1.to_v."):
                # Skip - handled by to_q fusion
                continue

            # Handle cross-attention KV fusion
            elif layer_path.startswith("attn2.to_k."):
                suffix = layer_path[len("attn2.to_k."):]
                k_key = f"blocks.{block_idx}.attn2.to_k.{suffix}"
                v_key = f"blocks.{block_idx}.attn2.to_v.{suffix}"

                if k_key in state_dict and v_key in state_dict:
                    new_key = f"blocks.{block_idx}.attn2.to_kv.{suffix}"
                    if new_key not in new_state_dict:
                        new_state_dict[new_key] = torch.cat([
                            state_dict[k_key],
                            state_dict[v_key],
                        ], dim=0)
            elif layer_path.startswith("attn2.to_v."):
                # Skip - handled by to_k fusion
                continue

            # Handle I2V additional KV fusion
            elif has_added_kv and layer_path.startswith("attn2.add_k_proj."):
                suffix = layer_path[len("attn2.add_k_proj."):]
                k_key = f"blocks.{block_idx}.attn2.add_k_proj.{suffix}"
                v_key = f"blocks.{block_idx}.attn2.add_v_proj.{suffix}"

                if k_key in state_dict and v_key in state_dict:
                    new_key = f"blocks.{block_idx}.attn2.to_added_kv.{suffix}"
                    if new_key not in new_state_dict:
                        new_state_dict[new_key] = torch.cat([
                            state_dict[k_key],
                            state_dict[v_key],
                        ], dim=0)
            elif has_added_kv and layer_path.startswith("attn2.add_v_proj."):
                # Skip - handled by add_k_proj fusion
                continue

            # Handle FFN net.2.linear vs net.2 naming
            elif layer_path.startswith("ffn.net.2.linear."):
                # Rename to ffn.net.2
                suffix = layer_path[len("ffn.net.2.linear."):]
                new_key = f"blocks.{block_idx}.ffn.net.2.{suffix}"
                new_state_dict[new_key] = value

            else:
                # Copy other layers as-is
                new_state_dict[key] = value
        else:
            # Copy non-block layers as-is
            new_state_dict[key] = value

    return new_state_dict
