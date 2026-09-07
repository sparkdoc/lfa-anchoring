"""Model adapters: the single place LFA is allowed to know a model's layout.

Every other part of the package reaches a sub-module through a :class:`ModelAdapter`
rather than by attribute path, so supporting a new architecture means writing one
adapter and registering it -- not editing the anchoring, training or analysis code.

An adapter is stateless: each method takes the model it should act on.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch
from torch import nn

__all__ = [
    "ModelAdapter",
    "UnsupportedModelError",
    "register_adapter",
    "get_adapter",
    "registered_adapters",
    "effective_weight",
    "LlamaLayoutAdapter",
]

#: The four hidden-state sites LFA anchors, mapped to the module whose *input* is
#: that site. Anchoring compares teacher and student outputs of these modules.
ANCHOR_SITES = ("pre_qkv", "pre_o", "pre_mlp", "pre_lm_head")


class UnsupportedModelError(ValueError):
    """Raised when no registered adapter recognises a model's layout."""


class ModelAdapter(ABC):
    """Resolves the sub-modules LFA anchors, for one family of model layouts."""

    #: Short identifier for this layout, used in logs and config records.
    name: str = "adapter"

    @classmethod
    @abstractmethod
    def matches(cls, model: nn.Module) -> bool:
        """Whether this adapter can resolve ``model``'s sub-modules.

        Must return ``False`` rather than raise for a model it does not recognise.
        """

    @abstractmethod
    def base_model(self, model: nn.Module) -> nn.Module:
        """Unwrap PEFT and causal-LM wrappers down to the decoder holding ``.layers``."""

    @abstractmethod
    def num_layers(self, model: nn.Module) -> int:
        """Number of transformer layers."""

    @abstractmethod
    def layer(self, model: nn.Module, i: int) -> nn.Module:
        """The transformer layer at index ``i``."""

    @abstractmethod
    def qkv_modules(self, model: nn.Module, i: int) -> dict[str, nn.Module]:
        """Layer ``i``'s query/key/value projections, keyed ``q_proj``/``k_proj``/``v_proj``."""

    @abstractmethod
    def o_proj_module(self, model: nn.Module, i: int) -> nn.Module:
        """Layer ``i``'s attention output projection."""

    @abstractmethod
    def mlp_module(self, model: nn.Module, i: int) -> nn.Module:
        """Layer ``i``'s MLP, as a whole callable (the nonlinearity is part of the anchored function)."""

    @abstractmethod
    def embed_modules(self, model: nn.Module) -> tuple[nn.Module, nn.Module]:
        """``(embed_tokens, layers[0].input_layernorm)`` -- the composed embedding function."""

    @abstractmethod
    def final_norm_module(self, model: nn.Module) -> nn.Module:
        """The decoder's final norm, applied before the LM head."""

    @abstractmethod
    def lm_head_module(self, model: nn.Module) -> nn.Module:
        """The language-modelling head."""

    @abstractmethod
    def lora_target_modules(self) -> list[str]:
        """Module names to attach LoRA adapters to for this layout."""

    def site_module(self, model: nn.Module, layer: int, site: str) -> nn.Module:
        """The module whose *input* is ``site``.

        ``site`` is one of ``pre_qkv``, ``pre_o``, ``pre_mlp`` (resolved within
        ``layer``) or ``pre_lm_head`` (model-level; ``layer`` is ignored).
        """
        if site == "pre_qkv":
            return self.qkv_modules(model, layer)["q_proj"]
        if site == "pre_o":
            return self.o_proj_module(model, layer)
        if site == "pre_mlp":
            return self.mlp_module(model, layer)
        if site == "pre_lm_head":
            return self.lm_head_module(model)
        raise ValueError(f"Unknown anchoring site {site!r}; expected one of {list(ANCHOR_SITES)}")


_ADAPTERS: list[type[ModelAdapter]] = []


def register_adapter(cls: type[ModelAdapter]) -> type[ModelAdapter]:
    """Register an adapter class for :func:`get_adapter` to consider. Usable as a decorator."""
    if cls not in _ADAPTERS:
        _ADAPTERS.append(cls)
    return cls


def registered_adapters() -> list[type[ModelAdapter]]:
    """The registered adapter classes, in the order :func:`get_adapter` tries them."""
    return list(_ADAPTERS)


def get_adapter(model: nn.Module) -> ModelAdapter:
    """The first registered adapter whose :meth:`~ModelAdapter.matches` accepts ``model``.

    Raises:
        UnsupportedModelError: naming the model's ``config.model_type``.
    """
    for cls in _ADAPTERS:
        if cls.matches(model):
            return cls()
    model_type = getattr(getattr(model, "config", None), "model_type", None)
    described = f"model_type={model_type!r}" if model_type is not None else "a model with no config.model_type"
    known = ", ".join(cls.name for cls in _ADAPTERS) or "none"
    raise UnsupportedModelError(
        f"No LFA model adapter matches {described} ({type(model).__name__}). Registered adapters: {known}."
    )


def effective_weight(module: nn.Module) -> torch.Tensor:
    """The weight ``module`` actually applies in its forward pass.

    For a plain ``nn.Linear`` this is ``module.weight``. For a PEFT LoRA ``Linear``
    it is ``W_base + scaling * B @ A`` summed over the adapters that are active and
    not already merged into the base weight -- a merged adapter's delta is in
    ``module.weight`` already, and adding it again would double-count it. Under
    ``disable_adapter()`` the base weight alone is returned, matching the forward pass
    (PEFT's ``active_adapters`` does not itself reflect that flag).

    The LoRA result stays connected to the autograd graph, so a caller may
    differentiate through it.
    """
    if hasattr(module, "lora_A") and hasattr(module, "lora_B") and hasattr(module, "scaling"):
        merged = set(getattr(module, "merged_adapters", None) or ())
        active = getattr(module, "active_adapters", None)
        if active is None:
            active = list(module.lora_A.keys())
        if getattr(module, "disable_adapters", False):
            active = []

        weight = module.weight.clone()
        for adapter_name in active:
            if adapter_name in merged or adapter_name not in module.lora_A:
                continue
            lora_a = module.lora_A[adapter_name].weight
            lora_b = module.lora_B[adapter_name].weight
            weight = weight + (lora_b @ lora_a) * module.scaling[adapter_name]
        return weight

    return module.weight


from .llama_layout import LlamaLayoutAdapter  # noqa: E402  (registers on import; needs the names above)
