"""Adapter for the Llama-style decoder layout.

Covers every model whose decoder layers expose ``self_attn.{q,k,v,o}_proj``, a whole
``mlp``, and an ``input_layernorm`` -- Llama, Qwen2/Qwen3, Mistral and their kin. The
lookups stay duck-typed rather than isinstance-based so a model only has to have the
shape, not the class.
"""

from __future__ import annotations

from torch import nn

from . import ModelAdapter, UnsupportedModelError, register_adapter

_QKV_NAMES = ("q_proj", "k_proj", "v_proj")
_ATTENTION_ATTRS = ("self_attn", "attention")
_MLP_ATTRS = ("mlp", "feed_forward")
_FINAL_NORM_ATTRS = ("norm", "final_layernorm", "ln_f")


@register_adapter
class LlamaLayoutAdapter(ModelAdapter):
    """Resolves LFA's anchoring sites in a Llama-style decoder."""

    name = "llama_layout"

    @classmethod
    def matches(cls, model: nn.Module) -> bool:
        try:
            layer = cls().layer(model, 0)
        except Exception:
            return False
        attn = _first_attr(layer, _ATTENTION_ATTRS)
        if attn is None or not all(hasattr(attn, n) for n in (*_QKV_NAMES, "o_proj")):
            return False
        return _first_attr(layer, _MLP_ATTRS) is not None and hasattr(layer, "input_layernorm")

    def base_model(self, model: nn.Module) -> nn.Module:
        """Unwrap a PEFT wrapper, then descend ``.model`` until the decoder with ``.layers``."""
        obj = model
        if hasattr(obj, "peft_config") and hasattr(obj, "base_model"):
            # PeftModelForCausalLM -> LoraModel -> the wrapped causal LM.
            obj = getattr(obj.base_model, "model", obj.base_model)

        seen: set[int] = set()
        while not hasattr(obj, "layers"):
            seen.add(id(obj))
            inner = getattr(obj, "model", None)
            if inner is None or id(inner) in seen:
                raise UnsupportedModelError(
                    f"Cannot find a layers container in {type(model).__name__}"
                )
            obj = inner
        return obj

    def num_layers(self, model: nn.Module) -> int:
        config = getattr(model, "config", None)
        for attr in ("num_hidden_layers", "n_layer", "num_layers"):
            value = getattr(config, attr, None)
            if isinstance(value, int):
                return value
        return len(self.base_model(model).layers)

    def layer(self, model: nn.Module, i: int) -> nn.Module:
        layers = self.base_model(model).layers
        if i < 0 or i >= len(layers):
            raise IndexError(f"Layer index {i} out of range [0, {len(layers)})")
        return layers[i]

    def qkv_modules(self, model: nn.Module, i: int) -> dict[str, nn.Module]:
        attn = self._attention(model, i)
        modules = {}
        for name in _QKV_NAMES:
            if not hasattr(attn, name):
                raise UnsupportedModelError(f"Cannot find {name} in layer {i}'s attention module")
            modules[name] = getattr(attn, name)
        return modules

    def o_proj_module(self, model: nn.Module, i: int) -> nn.Module:
        attn = self._attention(model, i)
        if not hasattr(attn, "o_proj"):
            raise UnsupportedModelError(f"Cannot find o_proj in layer {i}'s attention module")
        return attn.o_proj

    def mlp_module(self, model: nn.Module, i: int) -> nn.Module:
        mlp = _first_attr(self.layer(model, i), _MLP_ATTRS)
        if mlp is None:
            raise UnsupportedModelError(f"Cannot find an MLP module in layer {i}")
        return mlp

    def embed_modules(self, model: nn.Module) -> tuple[nn.Module, nn.Module]:
        base = self.base_model(model)
        if not hasattr(base, "embed_tokens"):
            raise UnsupportedModelError(f"Cannot find embed_tokens in {type(model).__name__}")
        if len(base.layers) == 0:
            raise UnsupportedModelError(f"{type(model).__name__} has no transformer layers")
        return base.embed_tokens, base.layers[0].input_layernorm

    def final_norm_module(self, model: nn.Module) -> nn.Module:
        base = self.base_model(model)
        norm = _first_attr(base, _FINAL_NORM_ATTRS)
        if norm is None:
            raise UnsupportedModelError(f"Cannot find a final norm in {type(model).__name__}")
        return norm

    def lm_head_module(self, model: nn.Module) -> nn.Module:
        obj = model
        if hasattr(obj, "peft_config") and hasattr(obj, "base_model"):
            obj = getattr(obj.base_model, "model", obj.base_model)
        if not hasattr(obj, "lm_head"):
            raise UnsupportedModelError(f"Cannot find lm_head in {type(model).__name__}")
        return obj.lm_head

    def lora_target_modules(self) -> list[str]:
        return ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

    def _attention(self, model: nn.Module, i: int) -> nn.Module:
        attn = _first_attr(self.layer(model, i), _ATTENTION_ATTRS)
        if attn is None:
            raise UnsupportedModelError(f"Cannot find an attention module in layer {i}")
        return attn


def _first_attr(obj: object, names: tuple[str, ...]):
    """The first of ``names`` present on ``obj``, or ``None``."""
    for name in names:
        value = getattr(obj, name, None)
        if value is not None:
            return value
    return None
