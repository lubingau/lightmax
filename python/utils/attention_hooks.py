"""
attention_hooks.py

Shared utility for applying pre-softmax attention logit saturation by
monkey-patching the `eager_attention_forward` in HuggingFace transformers
or the `forward` of timm attention modules.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

from tqdm import tqdm as _tqdm


def progress_bar(iterable, **kwargs):
    """Unified tqdm wrapper for evaluation scripts.

    Override default ``desc`` via ``progress_bar(..., desc="...")``.
    """
    kwargs.setdefault("desc", "  Batches")
    return _tqdm(iterable, **kwargs)


def _make_eager_saturation_hook(thr: Optional[float]):
    """
    Return a replacement for HF's `eager_attention_forward` that clamps
    QK^T * scaling to [-thr, thr] before softmax.  If thr is None the
    result is None (baseline, no patching).
    """
    if thr is None:
        return None  # signal "keep original"

    def _hooked(
        module: nn.Module,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        scaling: float,
        dropout: float = 0.0,
        **kwargs,
    ):
        attn_weights = torch.matmul(query, key.transpose(2, 3)) * scaling
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask
        # --- SATURATION: clamp pre-softmax logits ---
        attn_weights = torch.clamp(attn_weights, -thr, thr)
        # --------------------------------------------
        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
        attn_weights = F.dropout(attn_weights, p=dropout, training=module.training)
        attn_output = torch.matmul(attn_weights, value)
        attn_output = attn_output.transpose(1, 2).contiguous()
        return attn_output, attn_weights

    return _hooked


def install_hf_saturation_hook(model: nn.Module, thr: Optional[float]):
    """
    Monkey-patch all `eager_attention_forward` in every attention module
    of a HuggingFace model.  thr=None is a no-op (baseline).
    """
    hook_fn = _make_eager_saturation_hook(thr)
    if hook_fn is None:
        return  # baseline -- no patching

    for module in model.modules():
        if hasattr(module, "eager_attention_forward"):
            module.eager_attention_forward = hook_fn.__get__(module, type(module))


# ────────────────────── timm attention hooks ──────────────────────


def _make_timm_attention_hook(thr: Optional[float]):
    """Saturation hook for timm's `Attention` (ViT). Returns None if thr is None."""
    if thr is None:
        return None

    def _hooked(self, x, attn_mask=None):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)
        q = q * self.scale
        attn = q @ k.transpose(-2, -1)
        if attn_mask is not None:
            attn = attn + attn_mask
        # --- SATURATION ---
        attn = torch.clamp(attn, -thr, thr)
        # -----------------
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = attn @ v
        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    return _hooked


def _make_timm_windowattention_hook(thr: Optional[float]):
    """Saturation hook for timm's `WindowAttention` (Swin)."""
    if thr is None:
        return None

    def _hooked(self, x, mask=None):
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q = q * self.scale
        attn = q @ k.transpose(-2, -1)
        attn = attn + self._get_rel_pos_bias()
        if mask is not None:
            num_win = mask.shape[0]
            attn = attn.view(-1, num_win, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
        # --- SATURATION ---
        attn = torch.clamp(attn, -thr, thr)
        # -----------------
        attn = self.softmax(attn)
        attn = self.attn_drop(attn)
        x = attn @ v
        x = x.transpose(1, 2).reshape(B_, N, -1)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    return _hooked


# ────────────────────── T5 attention hooks (MOMENT) ──────────────────────


def _make_t5_attention_hook(thr: Optional[float]):
    """Saturation hook for T5Attention used by MOMENT."""
    if thr is None:
        return None

    def _hooked(self, hidden_states, mask=None, key_value_states=None,
                 position_bias=None, past_key_value=None, layer_head_mask=None,
                 query_length=None, use_cache=False, output_attentions=False,
                 **kwargs):
        bs, seq_len = hidden_states.shape[:2]
        d_head = self.key_value_proj_dim if hasattr(self, "key_value_proj_dim") else self.d_model // self.n_heads

        query = self.q(hidden_states).view(bs, -1, self.n_heads, d_head).transpose(1, 2)
        if key_value_states is not None:
            kv = key_value_states
        else:
            kv = hidden_states
        key = self.k(kv).view(bs, -1, self.n_heads, d_head).transpose(1, 2)
        value = self.v(kv).view(bs, -1, self.n_heads, d_head).transpose(1, 2)

        scale = d_head ** -0.5
        attn = torch.matmul(query, key.transpose(-1, -2)) * scale
        if position_bias is not None:
            attn = attn + position_bias
        if mask is not None:
            attn = attn + mask
        # --- SATURATION ---
        attn = torch.clamp(attn, -thr, thr)
        # -----------------
        attn = F.softmax(attn, dim=-1)
        attn = F.dropout(attn, p=self.dropout, training=self.training)
        out = torch.matmul(attn, value).transpose(1, 2).reshape(bs, seq_len, -1)
        out = self.o(out)
        return (out, position_bias, attn)

    return _hooked


# ────────────────────── Swin & Wav2Vec2 hooks ──────────────────────


def _make_eager_forward_clamp_wrapper(thr: float):
    """Wrap an `eager_attention_forward` to clamp before softmax."""
    def _wrapped(
        module: nn.Module,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        scaling: float,
        dropout: float = 0.0,
        **kwargs,
    ):
        attn_weights = torch.matmul(query, key.transpose(2, 3)) * scaling
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask
        attn_weights = torch.clamp(attn_weights, -thr, thr)
        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
        attn_weights = F.dropout(attn_weights, p=dropout, training=module.training)
        attn_output = torch.matmul(attn_weights, value)
        attn_output = attn_output.transpose(1, 2).contiguous()
        return attn_output, attn_weights

    return _wrapped


def install_swin_saturation_hook(model: nn.Module, thr: Optional[float]):
    """Patch `eager_attention_forward` in transformers.models.swin.modeling_swin."""
    if thr is None:
        return
    import transformers.models.swin.modeling_swin as swin_module
    swin_module.eager_attention_forward = _make_eager_forward_clamp_wrapper(thr)


def install_wav2vec2_saturation_hook(model: nn.Module, thr: Optional[float]):
    """Patch `eager_attention_forward` in transformers.models.wav2vec2.modeling_wav2vec2."""
    if thr is None:
        return
    import transformers.models.wav2vec2.modeling_wav2vec2 as w2v_module
    w2v_module.eager_attention_forward = _make_eager_forward_clamp_wrapper(thr)


# ────────────────────── Mask2FormerAttention hook (inline bmm + softmax) ──────────────────────


def install_modernbert_saturation_hook(model: nn.Module, thr: Optional[float]):
    """Patch `eager_attention_forward` in transformers.models.modernbert.modeling_modernbert.

    When thr is None, the original function is restored (undo).
    """
    import transformers.models.modernbert.modeling_modernbert as modernbert_module
    # Save original on first invocation
    if not hasattr(install_modernbert_saturation_hook, "_original"):
        install_modernbert_saturation_hook._original = modernbert_module.eager_attention_forward

    if thr is None:
        modernbert_module.eager_attention_forward = install_modernbert_saturation_hook._original
        return

    modernbert_module.eager_attention_forward = _make_eager_forward_clamp_wrapper(thr)


# ────────────────────── Mask2FormerAttention hook (inline bmm + softmax) ──────────────────────


def install_mask2former_attention_saturation_hook(model: nn.Module, thr: Optional[float]):
    """Install saturation on Mask2FormerAttention modules (inline bmm)."""
    if thr is None:
        return
    from transformers.models.mask2former.modeling_mask2former import Mask2FormerAttention

    def _patched_forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_embeddings: torch.Tensor | None = None,
        key_value_states: torch.Tensor | None = None,
        key_value_position_embeddings: torch.Tensor | None = None,
        output_attentions: bool = False,
    ):
        hidden_states = hidden_states.permute(1, 0, 2) if hidden_states is not None else None
        position_embeddings = position_embeddings.permute(1, 0, 2) if position_embeddings is not None else None
        key_value_states = key_value_states.permute(1, 0, 2) if key_value_states is not None else None
        key_value_position_embeddings = (
            key_value_position_embeddings.permute(1, 0, 2) if key_value_position_embeddings is not None else None
        )
        is_cross_attention = key_value_states is not None
        batch_size, target_len, embed_dim = hidden_states.size()
        if position_embeddings is not None:
            hidden_states_original = hidden_states
            hidden_states = self.with_pos_embed(hidden_states, position_embeddings)
        if key_value_position_embeddings is not None:
            key_value_states_original = key_value_states
            key_value_states = self.with_pos_embed(key_value_states, key_value_position_embeddings)
        query_states = self.q_proj(hidden_states) * self.scaling
        if is_cross_attention:
            key_states = self._shape(self.k_proj(key_value_states), -1, batch_size)
            value_states = self._shape(self.v_proj(key_value_states_original), -1, batch_size)
        else:
            key_states = self._shape(self.k_proj(hidden_states), -1, batch_size)
            value_states = self._shape(self.v_proj(hidden_states_original), -1, batch_size)
        proj_shape = (batch_size * self.num_heads, -1, self.head_dim)
        query_states = self._shape(query_states, target_len, batch_size).view(*proj_shape)
        key_states = key_states.view(*proj_shape)
        value_states = value_states.view(*proj_shape)
        source_len = key_states.size(1)
        attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
        if attention_mask is not None:
            attn_weights += attention_mask
        attn_weights = torch.clamp(attn_weights, -thr, thr)
        attn_weights = nn.functional.softmax(attn_weights, dim=-1)
        if output_attentions:
            attn_weights_reshaped = attn_weights.view(batch_size, self.num_heads, target_len, source_len)
            attn_weights = attn_weights_reshaped.view(batch_size * self.num_heads, target_len, source_len)
        else:
            attn_weights_reshaped = None
        attn_probs = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
        attn_output = torch.bmm(attn_probs, value_states)
        attn_output = attn_output.view(batch_size, self.num_heads, target_len, self.head_dim)
        attn_output = attn_output.transpose(1, 2)
        attn_output = attn_output.reshape(batch_size, target_len, embed_dim)
        attn_output = self.out_proj(attn_output).permute(1, 0, 2)
        return attn_output, attn_weights_reshaped

    for module in model.modules():
        if isinstance(module, Mask2FormerAttention):
            module.forward = _patched_forward.__get__(module, Mask2FormerAttention)


# ────────────────────── public installers ──────────────────────


def install_timm_saturation_hook(model: nn.Module, thr: Optional[float]):
    """Install on timm `Attention` (ViT) and `WindowAttention` (Swin)."""
    from timm.layers.attention import Attention
    from timm.models.swin_transformer import WindowAttention

    hook = _make_timm_attention_hook(thr)
    if hook is not None:
        for module in model.modules():
            if isinstance(module, Attention):
                module.forward = hook.__get__(module, Attention)

    win_hook = _make_timm_windowattention_hook(thr)
    if win_hook is not None:
        for module in model.modules():
            if isinstance(module, WindowAttention):
                module.forward = win_hook.__get__(module, WindowAttention)


def install_t5_saturation_hook(model: nn.Module, thr: Optional[float]):
    """Install saturation on T5Attention (used by MOMENT)."""
    from transformers.models.t5.modeling_t5 import T5Attention

    hook = _make_t5_attention_hook(thr)
    if hook is not None:
        for module in model.modules():
            if isinstance(module, T5Attention):
                module.forward = hook.__get__(module, T5Attention)

def install_patchtst_saturation_hook(model: nn.Module, thr: Optional[float]):
    """Install saturation on PatchTSTAttention (used by PatchTSTForPrediction)."""
    if thr is None:
        return

    from transformers.models.patchtst.modeling_patchtst import PatchTSTAttention

    def _make_patched_forward(thr_value: float):
        def _patched_forward(
            self,
            hidden_states: torch.Tensor,
            key_value_states: torch.Tensor | None = None,
            attention_mask: torch.Tensor | None = None,
            output_attentions: bool | None = False,
            **kwargs,
        ):
            is_cross_attention = key_value_states is not None
            input_shape = hidden_states.shape[:-1]
            hidden_shape = (*input_shape, -1, self.head_dim)

            query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
            current_states = key_value_states if is_cross_attention else hidden_states
            kv_shape = (*current_states.shape[:-1], -1, self.head_dim)
            key_states = self.k_proj(current_states).view(kv_shape).transpose(1, 2)
            value_states = self.v_proj(current_states).view(kv_shape).transpose(1, 2)

            # --- Manual attention with saturation clamp ---
            attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) * self.scaling
            if attention_mask is not None:
                attn_weights = attn_weights + attention_mask
            attn_weights = torch.clamp(attn_weights, -thr_value, thr_value)
            attn_weights = torch.nn.functional.softmax(attn_weights, dim=-1)
            attn_weights = torch.nn.functional.dropout(
                attn_weights, p=0.0 if not self.training else self.dropout,
                training=self.training,
            )
            attn_output = torch.matmul(attn_weights, value_states)
            attn_output = attn_output.transpose(1, 2).contiguous()
            attn_output = attn_output.reshape(*input_shape, -1).contiguous()
            attn_output = self.out_proj(attn_output)
            return attn_output, attn_weights
        return _patched_forward

    patched_fn = _make_patched_forward(thr)
    for module in model.modules():
        if isinstance(module, PatchTSTAttention):
            module.forward = patched_fn.__get__(module, PatchTSTAttention)


# =====================================================================
# LightMax hooks — replace F.softmax with lightmax_module(attn_weights)
# The clamp is also applied (saturation is always used with LightMax).
# =====================================================================


def _make_eager_lightmax_hook(lightmax_module, thr):
    """Return a replacement for HF's `eager_attention_forward` using LightMax."""
    if lightmax_module is None:
        return None

    def _hooked(module, query, key, value, attention_mask, scaling, dropout=0.0, **kwargs):
        attn_weights = torch.matmul(query, key.transpose(2, 3)) * scaling
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask
        if thr is not None:
            attn_weights = torch.clamp(attn_weights, -thr, thr)
        attn_weights = lightmax_module(attn_weights)
        attn_weights = attn_weights.to(value.dtype)  # ensure matching dtypes for matmul
        attn_weights = F.dropout(attn_weights, p=dropout, training=module.training)
        attn_output = torch.matmul(attn_weights, value)
        attn_output = attn_output.transpose(1, 2).contiguous()
        return attn_output, attn_weights
    return _hooked


def install_hf_lightmax_hook(model, lightmax_module, thr=None):
    """Replace eager_attention_forward softmax with LightMax in HF attention modules."""
    hook_fn = _make_eager_lightmax_hook(lightmax_module, thr)
    if hook_fn is None:
        return
    for module in model.modules():
        if hasattr(module, "eager_attention_forward"):
            module.eager_attention_forward = hook_fn.__get__(module, type(module))

def _make_timm_lightmax_hook(lightmax_module, thr):
    """Return a replacement for timm's `Attention.forward` using LightMax."""
    if lightmax_module is None:
        return None

    def _hooked(self, x, attn_mask=None):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)
        q = q * self.scale
        attn = q @ k.transpose(-2, -1)
        if attn_mask is not None:
            attn = attn + attn_mask
        if thr is not None:
            attn = torch.clamp(attn, -thr, thr)
        attn = lightmax_module(attn)
        attn = attn.to(v.dtype)  # ensure matching dtypes for matmul
        attn = self.attn_drop(attn)
        x = attn @ v
        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x
    return _hooked


def _make_timm_windowattention_lightmax_hook(lightmax_module, thr):
    """Return a replacement for timm's `WindowAttention.forward` using LightMax."""
    if lightmax_module is None:
        return None

    def _hooked(self, x, mask=None):
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q = q * self.scale
        attn = q @ k.transpose(-2, -1)
        attn = attn + self._get_rel_pos_bias()
        if mask is not None:
            num_win = mask.shape[0]
            attn = attn.view(-1, num_win, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
        if thr is not None:
            attn = torch.clamp(attn, -thr, thr)
        attn = lightmax_module(attn)
        attn = attn.to(v.dtype)  # ensure matching dtypes for matmul
        attn = self.attn_drop(attn)
        x = attn @ v
        x = x.transpose(1, 2).reshape(B_, N, -1)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x
    return _hooked


def install_timm_lightmax_hook(model, lightmax_module, thr=None):
    """Install LightMax on timm `Attention` (ViT) and `WindowAttention` (Swin)."""
    from timm.layers.attention import Attention
    from timm.models.swin_transformer import WindowAttention

    hook = _make_timm_lightmax_hook(lightmax_module, thr)
    if hook is not None:
        for module in model.modules():
            if isinstance(module, Attention):
                module.forward = hook.__get__(module, Attention)

    win_hook = _make_timm_windowattention_lightmax_hook(lightmax_module, thr)
    if win_hook is not None:
        for module in model.modules():
            if isinstance(module, WindowAttention):
                module.forward = win_hook.__get__(module, WindowAttention)

def _make_t5_lightmax_hook(lightmax_module, thr):
    """Return a replacement for T5Attention.forward using LightMax."""
    if lightmax_module is None:
        return None

    def _hooked(self, hidden_states, mask=None, key_value_states=None,
                 position_bias=None, past_key_value=None, layer_head_mask=None,
                 query_length=None, use_cache=False, output_attentions=False,
                 **kwargs):
        bs, seq_len = hidden_states.shape[:2]
        d_head = self.key_value_proj_dim if hasattr(self, "key_value_proj_dim") else self.d_model // self.n_heads

        query = self.q(hidden_states).view(bs, -1, self.n_heads, d_head).transpose(1, 2)
        if key_value_states is not None:
            kv = key_value_states
        else:
            kv = hidden_states
        key = self.k(kv).view(bs, -1, self.n_heads, d_head).transpose(1, 2)
        value = self.v(kv).view(bs, -1, self.n_heads, d_head).transpose(1, 2)

        scale = d_head ** -0.5
        attn = torch.matmul(query, key.transpose(-1, -2)) * scale
        if position_bias is not None:
            attn = attn + position_bias
        if mask is not None:
            attn = attn + mask
        if thr is not None:
            attn = torch.clamp(attn, -thr, thr)
        attn = lightmax_module(attn)
        attn = attn.to(value.dtype)  # ensure matching dtypes for matmul
        attn = F.dropout(attn, p=self.dropout, training=self.training)
        out = torch.matmul(attn, value).transpose(1, 2).reshape(bs, seq_len, -1)
        out = self.o(out)
        return (out, position_bias, attn)
    return _hooked


def install_t5_lightmax_hook(model, lightmax_module, thr=None):
    """Install LightMax on T5Attention (used by MOMENT)."""
    from transformers.models.t5.modeling_t5 import T5Attention
    hook = _make_t5_lightmax_hook(lightmax_module, thr)
    if hook is not None:
        for module in model.modules():
            if isinstance(module, T5Attention):
                module.forward = hook.__get__(module, T5Attention)


def _make_patchtst_lightmax_hook(lightmax_module, thr):
    """Return a replacement for PatchTSTAttention.forward using LightMax."""
    if lightmax_module is None:
        return None

    def _patched_forward(self, hidden_states, key_value_states=None,
                         attention_mask=None, output_attentions=False, **kwargs):
        is_cross_attention = key_value_states is not None
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        current_states = key_value_states if is_cross_attention else hidden_states
        kv_shape = (*current_states.shape[:-1], -1, self.head_dim)
        key_states = self.k_proj(current_states).view(kv_shape).transpose(1, 2)
        value_states = self.v_proj(current_states).view(kv_shape).transpose(1, 2)

        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) * self.scaling
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask
        if thr is not None:
            attn_weights = torch.clamp(attn_weights, -thr, thr)
        attn_weights = lightmax_module(attn_weights)
        attn_weights = attn_weights.to(value_states.dtype)  # ensure matching dtypes for matmul
        attn_weights = torch.nn.functional.dropout(
            attn_weights, p=0.0 if not self.training else self.dropout, training=self.training,
        )
        attn_output = torch.matmul(attn_weights, value_states)
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.out_proj(attn_output)
        return attn_output, attn_weights
    return _patched_forward


def install_patchtst_lightmax_hook(model, lightmax_module, thr=None):
    """Install LightMax on PatchTSTAttention (used by PatchTSTForPrediction)."""
    if lightmax_module is None:
        return
    from transformers.models.patchtst.modeling_patchtst import PatchTSTAttention
    patched_fn = _make_patchtst_lightmax_hook(lightmax_module, thr)
    for module in model.modules():
        if isinstance(module, PatchTSTAttention):
            module.forward = patched_fn.__get__(module, PatchTSTAttention)


def install_modernbert_lightmax_hook(model, lightmax_module, thr=None):
    """Replace eager_attention_forward softmax with LightMax in ModernBERT (attention only)."""
    if lightmax_module is None:
        return
    import transformers.models.modernbert.modeling_modernbert as modernbert_module
    modernbert_module.eager_attention_forward = _make_eager_lightmax_hook(lightmax_module, thr)


def install_wav2vec2_lightmax_hook(model, lightmax_module, thr=None):
    """Replace eager_attention_forward softmax with LightMax in wav2vec2."""
    if lightmax_module is None:
        return
    import transformers.models.wav2vec2.modeling_wav2vec2 as w2v_module
    w2v_module.eager_attention_forward = _make_eager_lightmax_hook(lightmax_module, thr)


def install_swin_lightmax_hook(model, lightmax_module, thr=None):
    """Replace eager_attention_forward softmax with LightMax in HuggingFace Swin."""
    if lightmax_module is None:
        return
    import transformers.models.swin.modeling_swin as swin_module
    swin_module.eager_attention_forward = _make_eager_lightmax_hook(lightmax_module, thr)

