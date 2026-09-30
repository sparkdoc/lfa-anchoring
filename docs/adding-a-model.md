# Adding a model

Three things are needed for a model this package has never seen: an **adapter** (so LFA can find
the sub-modules to anchor), an **artifact** (so it has a p(h) to sample), and a **λ calibration**
(so the anchor is set to something that means anything on that model). The first is usually free;
the second is a few GPU-hours, once; the third is the real work.

## 1. The adapter

`lfa/adapters/` is the only place this package is allowed to know a model's layout. Everything else
reaches a sub-module through a `ModelAdapter`, so a new architecture is one new class, not an edit
to the anchoring, training or analysis code.

**Try the existing one first.** `LlamaLayoutAdapter` covers any decoder whose layers expose
`self_attn.{q,k,v,o}_proj`, a whole `mlp`, and an `input_layernorm` — Llama, Qwen2/Qwen3, Mistral
and their kin. The lookups are duck-typed, not `isinstance`-based, so a model only has to have the
shape:

```python
from transformers import AutoModelForCausalLM
from lfa.adapters import get_adapter

model = AutoModelForCausalLM.from_pretrained("your/model")
print(get_adapter(model).name)          # 'llama_layout', or UnsupportedModelError
```

`UnsupportedModelError` names the `config.model_type` it could not place.

### The interface

A new adapter subclasses `ModelAdapter` and implements: `matches`, `base_model`, `num_layers`,
`layer`, `qkv_modules`, `o_proj_module`, `mlp_module`, `embed_modules`, `final_norm_module`,
`lm_head_module`, `lora_target_modules`. It is **stateless** — every method takes the model it acts
on — and registered with `@register_adapter`. `matches` must *return `False`* for a model it does
not recognise rather than raise, since `get_adapter` tries every registered class in turn.

Three conventions worth copying rather than re-deriving:

* **`base_model` unwraps PEFT first, then descends `.model` until it finds `.layers`.** Every other
  method goes through it, so a LoRA-wrapped model and a bare one resolve identically.
* **`num_layers` prefers the config** (`num_hidden_layers`, then `n_layer`, then `num_layers`) and
  only falls back to `len(base_model.layers)`. The artifact's layer count is validated against
  this, so a config that disagrees with the module list is worth failing on.
* **`final_norm_module` tries `norm`, then `final_layernorm`, then `ln_f`** — the three names the
  same module goes by across families.

`site_module(model, layer, site)` maps the four anchoring sites onto those accessors and is what
the artifact build hooks and what the extension collects through; it is implemented on the base
class, so a new adapter gets it for free.

### What needs a genuinely new adapter

* **Fused QKV.** A model with a single `qkv_proj` (or `Wqkv`) has one site where LFA expects three
  modules. The adapter has to expose the three, which means either splitting the fused weight into
  views or anchoring the fused projection as one module — a decision about what function is being
  preserved, not a naming fix.
* **Mixture-of-experts MLPs.** `mlp_module` returns *the whole MLP as one callable* because the
  nonlinearity is part of the anchored function. With a router and N experts, the function on a
  sampled `h` is whatever the router sends it to, and the sampled states have no routing history —
  so what p(h) means at that site needs deciding before an adapter is written.
* **Anything without a per-layer `input_layernorm`.** `embed_modules` returns
  `(embed_tokens, layers[0].input_layernorm)`, the composed function the layer-0 lookup table is
  built from.

Two natural next verified models are **Gemma 3 1B** and **Llama 3.2 1B**: both are Llama-layout, so
the existing adapter should place them, and both are small enough that the artifact build and a λ
sweep fit on one 24 GB card.

## 2. The artifact

p(h) is model-specific — it is that model's own hidden states — so a new model needs its own
artifact, built once.

```bash
lfa init runs/new_model --model your/model --artifact self-generated --recipe my_point.yaml
```

The model writes its own text and p(h) is fitted on it — at the recorded frame by default, with no
dataset downloaded — and the result is kept in the local store, so every later workspace over that
model reuses it ([the-artifact.md](the-artifact.md)). It starts each document from the model's
document-boundary token — its declared `generation_config.bos_token_id`, else the tokenizer's BOS,
else its EOS. The optional chat-format share (`--n-chat`, outside the recorded frame and off by
default) starts from the user-turn header of its chat template; without a chat template that share
is skipped and the log says so. No bundled recipe names a new model, so pass your copy of one with
`--recipe` (§3), or name it at every `train`.

What the route was worth on Qwen3-0.6B (an artifact fitted on the model's own text matched one
fitted on real text at every λ tried, and was at least as good at the recipe's λ; one model, one
seed, one domain) is in [the-artifact.md](the-artifact.md#the-frame). On any other model nothing
has been measured: it gives you a first artifact, and §3 calibrates λ against it.

The advanced alternative is an artifact fitted on real text, over a downloaded seed corpus:

```bash
lfa prepare-seed-corpus --out data/seed_corpus_10to1.jsonl
lfa build-artifact --model your/model --corpus data/seed_corpus_10to1.jsonl \
                   --out data/distributions/your-model/distribution_stats.pt \
                   --layer-group-size 7
```

Everything about that step — the corpus composition, the memory arithmetic, what is stored and what
is deliberately not — is in [the-artifact.md](the-artifact.md). The build
writes a `__meta__` block naming the model, its hidden size and its depth, and every training run
validates the artifact against the model it is about to anchor, so a mismatched pair fails at
startup rather than anchoring toward the wrong function in silence.

## 3. Calibrating λ

**Do not port λ.** It is coupled to the LoRA rank, to the artifact, and to the corpus, and none of
those survives a change of model. The bundled 100,000 belongs to Qwen3-0.6B at rank 32 on its
self-generated artifact at the recorded frame, and means nothing elsewhere.

A workable procedure:

1. Start from a copy of the bundled recipe with `model_id` and `calibrated_rank` set to your
   point ([recipes.md](recipes.md)). Set `calibrated_artifact` to a placeholder such as
   `uncalibrated` until the sweep below has been read: the copy carries `self-generated`, which
   would silence the warning for a self-generated artifact of your model although nothing was
   measured on it, and the placeholder makes every stage say that λ was calibrated against
   something else, which is true. Set it back to `self-generated` (with `self_generated_frame` at
   the frame you built at) once the sweep is read. Keep `supplement_fraction` at the value you
   will train at: `train` mixes the supplement in during the sweep as it will afterwards, and λ is
   read in that mix.
2. Train the **unanchored** control first (`lfa evaluate --compare-unanchored`, or a recipe with
   `lambda_qkv: 0`, `lambda_mlp: 0`, `mu: 0`). It sets both ends of the scale: how far the domain
   can move, and what that costs on the general axis when nothing is preserved.
3. Sweep λ over roughly a decade, a factor of 2–3 apart, at a fixed rank and epoch count. Read the
   **frontier** — held-out domain perplexity against general perplexity — not a single point.
4. Pick from the frontier, not from the general axis alone. Over-anchoring makes general perplexity
   look its best while domain quality collapses; that failure is invisible unless you are watching
   the domain number.
5. Then, and only then, change rank. Lower rank ⇒ lower λ, and the frontier has to be re-read.

μ = 0.05 is a reasonable starting backstop on a new model; it is a global shrinkage term rather
than a steering one, so it is far less sensitive than λ.
