# Model-integration cookbook

Written for an agent (or a person) integrating a new causal language model into this package. Each
step gives the command to run, what to check, and what it means if the check fails. Work through
the steps in order: the first three take minutes, most of it on a CPU, and a failure caught there
saves GPU-hours later.

A model this package has never seen needs three things: an **adapter** (so LFA can find the
sub-modules to anchor), an **artifact** (so it has a p(h) to sample), and a **λ calibration** (so
the anchor is set to something that means anything on that model). The first is usually free; the
second is a few GPU-hours, once; the third is the real work.

The steps:

1. [Decide fit](#1-decide-fit) — will the package's assumptions hold, and does the adapter place it?
2. [Tokenizer and chat checks](#2-tokenizer-and-chat-checks) — what the generators read off the tokenizer.
3. [Memory and cost arithmetic](#3-memory-and-cost-arithmetic) — before anything runs on the card.
4. [Build the self-generated artifact, then probe it](#4-build-the-self-generated-artifact-then-probe-it).
5. [Calibrate λ](#5-calibrate-λ).
6. [Write the recipe](#6-write-the-recipe).
7. [Tests to add](#7-tests-to-add).
8. [Acceptance checklist](#8-acceptance-checklist).
9. [Worked example: Qwen3-1.7B](#9-worked-example-qwen3-17b).
10. [Extending to multimodal models (image and audio) — general advice, untested](#10-extending-to-multimodal-models-image-and-audio--general-advice-untested).

Every GPU command below runs on one card. Pin it with `CUDA_VISIBLE_DEVICES=<n>` or `--device`; the
package refuses to shard a model across cards unless told to ([faq.md](faq.md#how-much-gpu-memory-does-a-run-need)).

## 1. Decide fit

**What the package assumes.**

* A decoder-only causal LM that loads with `AutoModelForCausalLM` and `AutoTokenizer`
  (`lfa/models.py`).
* Per layer: attention with separate `q_proj`, `k_proj`, `v_proj` and `o_proj`; an MLP with
  `gate_proj`, `up_proj` and `down_proj`; an `input_layernorm`. Model level: a final norm and an
  `lm_head`.
* Training fits one 24 GB card in bfloat16, with gradient checkpointing (§3 has the arithmetic).
* A tokenizer with a chat template, for the question-and-answer supplement. Without one the
  supplement prompts go out as plain text, outside the recorded frame, and the log says so.

**The adapter check.** `lfa/adapters/` is the only place this package is allowed to know a model's
layout. Everything else reaches a sub-module through a `ModelAdapter`, so a new architecture is one
new class, not an edit to the anchoring, training or analysis code.

Try the existing adapter first. `LlamaLayoutAdapter` covers any decoder whose layers expose
`self_attn.{q,k,v,o}_proj` (or `attention.…`), a whole `mlp` (or `feed_forward`), and an
`input_layernorm` — Llama, Qwen2/Qwen3, Mistral and their kin. The lookups are duck-typed, not
`isinstance`-based, so a model only has to have the shape. The check needs no weights: build the
model on the meta device from its config.

```python
import torch
from transformers import AutoConfig, AutoModelForCausalLM
from lfa.adapters import get_adapter

config = AutoConfig.from_pretrained("your/model")
with torch.device("meta"):
    model = AutoModelForCausalLM.from_config(config)
adapter = get_adapter(model)
print(adapter.name, adapter.num_layers(model))      # 'llama_layout' 28, or UnsupportedModelError
```

* **Passes:** it prints `llama_layout`. Go on to the site geometry test in §7, which proves the
  widths, before anything touches a GPU.
* **Fails:** `UnsupportedModelError` names the `config.model_type` it could not place. The model
  needs an adapter of its own (below).

**The interface.** A new adapter subclasses `ModelAdapter` (`lfa/adapters/__init__.py`) and
implements: `matches`, `base_model`, `num_layers`, `layer`, `qkv_modules`, `o_proj_module`,
`mlp_module`, `embed_modules`, `final_norm_module`, `lm_head_module`, `lora_target_modules`. It is
**stateless** — every method takes the model it acts on — and registered with `@register_adapter`.
`matches` must *return `False`* for a model it does not recognise rather than raise, since
`get_adapter` tries every registered class in turn and returns the first that matches.

Three conventions worth copying rather than re-deriving:

* **`base_model` unwraps PEFT first, then descends `.model` until it finds `.layers`.** Every other
  method goes through it, so a LoRA-wrapped model and a bare one resolve identically.
* **`num_layers` prefers the config** (`num_hidden_layers`, then `n_layer`, then `num_layers`) and
  only falls back to `len(base_model.layers)`. The artifact's layer count is validated against
  this, so a config that disagrees with the module list is worth failing on.
* **`final_norm_module` tries `norm`, then `final_layernorm`, then `ln_f`** — the three names the
  same module goes by across families.

`site_module(model, layer, site)` maps the four anchoring sites (`pre_qkv`, `pre_o`, `pre_mlp`,
`pre_lm_head`) onto those accessors and is what the artifact build hooks and what the extension
collects through; it is implemented on the base class, so a new adapter gets it for free.

**What needs a genuinely new adapter.**

* **Fused QKV.** A model with a single `qkv_proj` (Phi-3) or `query_key_value` (GPT-NeoX) has one
  module where LFA expects three. The adapter has to expose the three, which means either splitting
  the fused weight into views or anchoring the fused projection as one module — a decision about
  what function is being preserved, not a naming fix. A fused `gate_up_proj` raises the same
  question for the MLP's LoRA targets.
* **Mixture-of-experts MLPs.** `mlp_module` returns *the whole MLP as one callable* because the
  nonlinearity is part of the anchored function. With a router and N experts, the function on a
  sampled `h` is whatever the router sends it to, and the sampled states have no routing history —
  so what p(h) means at that site needs deciding before an adapter is written.
* **Anything without a per-layer `input_layernorm`.** `embed_modules` returns
  `(embed_tokens, layers[0].input_layernorm)`, the composed function the layer-0 lookup table is
  built from.
* **Names outside the duck-typing** — `transformer.h`, `c_attn`, `dense_h_to_4h` and the like. The
  adapter maps them; nothing else in the package should learn them.

**Where projection names are also written down.** The adapter is the only place that knows the
*layout*, but three places name the *projections*, and a model whose projections are named or
shaped differently needs them reviewed alongside the adapter:

* `lfa/losses.py` — `_QKV_NAMES`, `_ATTENTION_PROJECTIONS`, `_MLP_PROJECTIONS`, and
  `_target_modules`, which reads the MLP's projections off the module by name for μ's weight loss.
* `lfa/probe.py` — `SITE_PROJECTIONS`, and the `pre_mlp` price, which is written for a SwiGLU block
  `down(silu(gate(h)) * up(h))`. The probe refuses an MLP without those three projections or whose
  `act_fn` is not SiLU (a gated GELU, for one), rather than price the wrong function.
* `LlamaLayoutAdapter.lora_target_modules` — `q/k/v/o_proj` and `gate/up/down_proj`.

Two natural next models to verify are **Gemma 3 1B** and **Llama 3.2 1B**: both should be placed by
the existing adapter, and both are small enough that the artifact build and a λ sweep fit on one
24 GB card. Neither has been run here. Gemma's MLP is a gated GELU, so `lfa probe-artifact` would
refuse it as written.

## 2. Tokenizer and chat checks

Self-generation and the supplement read four things off the tokenizer and the model's
`generation_config`. Each is read, never assumed, but each can be read *wrong* on a template
nobody has tried, and a wrong one fails quietly: a corpus that starts every document mid-turn, or a
supplement whose answers run on past the turn. Check them on the real tokenizer — it downloads in
seconds, with no weights.

```python
from transformers import AutoTokenizer, GenerationConfig
import types
from lfa.selfgen.generate import (boundary_markers, chat_turn_end, chat_user_header,
                                  pick_seed_prefix)

tok = AutoTokenizer.from_pretrained("your/model")
model = types.SimpleNamespace(generation_config=GenerationConfig.from_pretrained("your/model"))
prefix = pick_seed_prefix(tok, model)
print(repr(prefix), tok(prefix, add_special_tokens=False)["input_ids"])
print(repr(chat_user_header(tok)), repr(chat_turn_end(tok)))
print(boundary_markers(tok, prefix))
```

* **Document start** (`pick_seed_prefix`). The token a free-running sample starts from: the
  model's declared `generation_config.bos_token_id`, then the tokenizer's BOS, then its EOS, then a
  bare newline. Qwen3 declares `<|endoftext|>` there while `tok.bos_token` is `None`, which is why
  the declared id comes first. **Check:** it is the model's document-boundary token and it encodes
  to **one** id. **If not:** a multi-token or chat-turn prefix starts every document inside some
  other context, and the artifact describes that context instead of free text. Fix the precedence
  for this model before building anything.
* **User-turn opener** (`chat_user_header`). Isolated by rendering two conversations and taking the
  difference, so a preamble the template adds is not mistaken for the opener. **Check:** it is
  exactly what opens a user turn (`"<|im_start|>user\n"` under ChatML). **If `None`:** the model
  has no usable chat template; the optional `--n-chat` share is skipped and the log says so.
* **Turn end** (`chat_turn_end`). The special token that closes an assistant turn, read off the
  template: the earliest special token in the assistant turn's close, where "special" includes
  added tokens marked `special=True` that are not in the special-tokens map. ChatML gives
  `<|im_end|>`, Llama-3-style templates `<|eot_id|>`. The supplement writer stops on it and cuts
  each answer at it. **If `None` or wrong:** the writer stops only at EOS and the parsed answers
  carry whatever the model wrote after its turn.
* **Boundary markers** (`boundary_markers`) — where a raw continuation is cut: the document
  boundary, EOS, and the opener's leading special token. **Check:** the boundary and the turn
  opener's special token are both in it.

**Template quirks to look for.**

* **Thinking flags.** The supplement prompt and the training rows are rendered with
  `enable_thinking=False`. Under Qwen3's template that renders an *empty* think block in the
  assistant turn; a template without the flag ignores the extra variable. Render one prompt with
  `tok.apply_chat_template([{"role": "user", "content": "x"}], tokenize=False,
  add_generation_prompt=True, enable_thinking=False)` and read it: a template whose thinking switch
  is spelled differently will think at length inside every supplement answer.
* **A default system prompt.** Some templates inject a system turn when the conversation has none —
  Qwen2.5-Instruct's is an example. `chat_user_header` handles it (the preamble is not the opener;
  a CPU test pins this on a synthetic template), but the supplement prompts then carry that system
  text, and so do the training rows rendered from it. Read a rendered prompt and decide whether that
  is what you want the model trained on.
* **Sampling defaults.** `generate` inherits every knob it is not given from the checkpoint's
  `generation_config.json`; Qwen3's ships `top_k: 20`, which silently truncates a 151,936-token
  vocabulary. The generator passes every knob explicitly — temperature, top-p, `top_k=0`,
  `min_p=0.0`, `repetition_penalty=1.0` — so a model's own defaults never reach the artifact
  corpus. **Check:** if you write a generator of your own, do the same.

**The test to copy:** `tests/test_model_tokenizers.py` (marked `slow`: it downloads tokenizer and
config files). Add the model id to its `MODELS` and run `pytest tests/test_model_tokenizers.py -m
slow -q` (§7).

## 3. Memory and cost arithmetic

Do this on paper first. Two models have been measured (Qwen3-0.6B and Qwen3-1.7B, below and in §9);
a new model's numbers are arithmetic until it has run.

**Training VRAM** (bf16, LoRA, gradient checkpointing on, one model: the frozen teacher is read out
of the student's own LoRA base with the adapters switched off):

* the weights: 2 B × parameters as loaded, a tied head counted once (Qwen3-0.6B: 596 M
  parameters, 1.11 GiB; Qwen3-1.7B: 1.72 B, 3.44 GB). A checkpoint can store a tied head twice, so
  its file size overstates the load (§9);
* LoRA's parameters, gradients and Adam state: `r × (d_in + d_out)` per adapted projection per
  layer — an estimate, not a measurement: small beside the rest at rank 32 on models this size;
* activations under checkpointing, estimated: about `batch × seq × hidden × 2 B` per layer
  boundary, plus one layer's recomputation;
* **the logits in fp32**: `batch × seq × vocab × 4 B` — 6 × 512 × 151,936 × 4 B = 1.87 GB at the
  recipe's geometry with a Qwen3 vocabulary, and the loss can hold more than one copy (an
  estimate). On a large vocabulary this term is what overflowed a small card first: Qwen3-0.6B at
  micro-batch 6 on an 8 GB card ([faq.md](faq.md#how-much-gpu-memory-does-a-run-need));
* the anchor itself: small — it evaluates sub-modules on 16 sampled vectors, not on the batch.

Measured anchors (RTX 3090, 24 GB): Qwen3-0.6B at the shipped recipe peaked at 8.63 GiB allocated
(2026-09-07, with a separately loaded teacher, which the default no longer loads); an unanchored
4-epoch Qwen3-0.6B run peaked at 7,325 MiB by `nvidia-smi` over init, train and evaluate
(2026-10-03). For Qwen3-1.7B, §9 has the GPU smoke's peak.

If batch 6 × 512 does not fit, keep the geometry and halve the micro-batch: `batch_size: 3`,
`gradient_accumulation_steps: 2`. The package does not pick a batch for you, on purpose: the
training frame must not depend on the card.

**Artifact build, GPU.** The model is loaded in **float32** for the fit, deliberately (the artifact
is a second-moment estimate, and bf16's 8-bit mantissa is a large error on a covariance): 4 B ×
parameters as loaded, about 6.9 GB for Qwen3-1.7B. Self-generation runs in bf16.

**Artifact build, host RAM: the layer group.** The fit's bill is the reservoirs — 200,000 vectors
per site in fp16 — at `reservoir_size × (2·hidden + pre_o_width) × itemsize` per layer, where
`pre_o_width = num_heads × head_dim` (which need not equal the hidden size):

| | per layer | all 28 layers |
|---|---:|---:|
| Qwen3-0.6B (hidden 1024, `pre_o` 2048) | 1.53 GiB | 42.7 GiB |
| Qwen3-1.7B (hidden 2048, `pre_o` 2048) | 2.29 GiB | 64.1 GiB |

On the self-generated route the build chooses the group itself (`choose_layer_group_size` in
`lfa/artifact/build.py`): the most layers whose reservoirs fit in half the RAM available when it
starts, at least one. It logs the choice:

```
layer_group_size=28 for Qwen/Qwen3-0.6B: ~42.7 GiB of reservoirs per group against 117.9 GiB available
```

Fewer layers per group means more passes over the corpus, not a different artifact: every site
draws its reservoir from its own seeded generator, so any group size gives the same artifact at a
fixed seed ([the-artifact.md](the-artifact.md#host-ram-the-layer-group)).

**Artifact size.** The stored statistics grow with `Σ d²` over the anchored sites. Qwen3-0.6B's 84
sites sum to 176,160,768 and its artifact is 110.3 MB (int8); Qwen3-1.7B's 84 sites, all 2,048
wide, sum to 352,321,536, twice that.

**Time.** On one RTX 3090 the full-frame Qwen3-0.6B build took about 80 minutes of generation and
2 h 22 min of fitting (2026-09-30); Qwen3-1.7B's generation took 88 minutes (§9). Plan for hours,
and see §4 on running detached.

## 4. Build the self-generated artifact, then probe it

p(h) is model-specific — it is that model's own hidden states — so a new model needs its own
artifact, built once. A new model has no published artifact for `init` to download
([the-artifact.md](the-artifact.md#published-artifacts)) until someone builds one and publishes it,
so `init` builds it.

**First a trial build**, to find a broken tokenizer reading or a refused corpus in minutes rather
than hours. It is off the recorded frame, and `train` will say so; that is expected:

```bash
lfa init runs/trial --model your/model --artifact self-generated \
         --n-raw 60 --max-new-tokens 128 --recipe recipes/your-model.yaml
```

No bundled recipe names a new model, so pass your copy of one with `--recipe` (§5 step 1 says how
to make it), or name it at every `train`. Without it, `init` adopts nothing; with the recipe of
*another* model, `init` warns that it is calibrated for that model and that λ does not port between
models.

**Then the full frame:**

```bash
lfa init runs/your-model --model your/model --artifact self-generated --recipe recipes/your-model.yaml
```

The model writes its own text and p(h) is fitted on it — at the recorded frame by default (2,500
raw documents of up to 2,048 tokens, 600k samples per site, K = 32), with no dataset downloaded —
and the result is kept in the local store (`~/.cache/lfa/artifacts`, or `$LFA_ARTIFACT_STORE`), so
every later workspace over that model reuses it ([the-artifact.md](the-artifact.md)). It starts each
document from the document-start token of §2. The optional chat-format share (`--n-chat`, outside
the recorded frame and off by default) starts from the user-turn opener; without a chat template
that share is skipped and the log says so.

**Run it detached.** The build takes hours. Launch it so that it survives the shell, the session
and any time limit your harness puts on background tasks, and watch the log rather than the
process:

```bash
setsid nohup lfa init runs/your-model --model your/model --artifact self-generated \
      --recipe recipes/your-model.yaml > build.log 2>&1 < /dev/null & disown
```

If it stops anyway — Ctrl-C, a crash, a reboot — run the same command again: generation resumes at
the next batch, and a complete corpus is fitted without generating again
([the-artifact.md](the-artifact.md#durability-and-resume)). A second build of the same entry is
refused while the first holds its lock.

**What to check, in order.**

* **The generation log.** One line per batch, `self-generated corpus: <n>/2500 documents (<e>
  empty)`. A corpus with fewer than 50 non-empty documents, or with more than 20 % of the documents
  asked for coming out empty or degenerate, is refused before any fit. **If it is refused:** read
  the refused draws' text before touching the limit. On Qwen3-1.7B the refusal was a filter that
  did not fit the model's text, not a broken model (§9).
* **What the model writes unprompted.** Read the corpus before the fit finishes: in a workspace it
  is `artifacts/v1.corpus.jsonl`, in the store `<entry>/corpus.jsonl` (`lfa list-artifacts` prints
  the entry's path; while the build runs, `corpus.jsonl.partial`). Rows are `{"text", "source"}`.
  Read twenty documents at random, and count the language and register:

  ```python
  import json, re
  docs = [json.loads(line)["text"] for line in open("corpus.jsonl")]
  cjk = re.compile(r"[\u4e00-\u9fff]")     # CJK Unified Ideographs, as in the recorded shares
  mostly_cjk = sum(len(cjk.findall(d)) > 0.3 * max(len(d), 1) for d in docs)
  print(f"{mostly_cjk}/{len(docs)} documents are mostly CJK")
  ```

  The corpus is what p(h) is fitted on, so a model that writes 40 % of its documents in another
  language puts that share of the artifact's mass on that language's hidden states. That is
  faithful to the model — it is what it does from a bare document start — but it is not the text
  you will train on, and whether it matters for anchoring on a domain in another language has not
  been measured. Record the share. Watch, too, for anything in the package that counts
  whitespace-separated words: scripts written without spaces (Chinese, Japanese, Thai) are one or
  two "words" a paragraph, and a word-count filter mis-handles them (§9).
* **The layer-group line and the collection line**, e.g. `Collected 84 sites, 603973 samples at
  the thinnest site` (Qwen3-0.6B, one group). With several groups each logs its own: a group of
  `g` layers that includes layer 0 has `3g − 1` sites (layer 0's `pre_qkv` is an exact embedding
  lookup, not a fitted site), and the last group adds `pre_lm_head`.
* **The store entry**: `lfa list-artifacts` shows `built <date>` for a finished entry.
* **The format contract.** The build writes a `__meta__` block naming the model, its hidden size
  and its depth, with `built_with: lfa-anchoring` (what a file is accepted on) and `format_version`
  (the stored layout, `lfa.artifact.ARTIFACT_FORMAT`). Every training run validates the artifact
  against the model it is about to anchor, so a mismatched pair fails at startup rather than
  anchoring toward the wrong function in silence:

  ```python
  from lfa.artifact.schema import load_artifact
  meta = load_artifact("runs/your-model/artifacts/v1.pt")["__meta__"]
  print({k: meta[k] for k in ("model_id", "hidden_size", "num_layers", "built_with",
                              "format_version", "provenance", "selfgen_frame")})
  ```

  A model that needs a new head type or site family changes the stored layout: that is a
  `format_version` bump, never a variant of format 1.

**The advanced alternative** is an artifact fitted on real text, over a downloaded seed corpus.
Over a seed corpus nothing chooses the layer group for you, so set it from the arithmetic in §3:

```bash
lfa prepare-seed-corpus --out data/seed_corpus_10to1.jsonl
lfa build-artifact --model your/model --corpus data/seed_corpus_10to1.jsonl \
                   --out data/distributions/your-model/distribution_stats.pt \
                   --layer-group-size 7
```

Everything about that route — the corpus composition, what is stored and what is deliberately not
— is in [the-artifact.md](the-artifact.md#advanced-an-artifact-fitted-on-real-text). What the
self-generated route was worth on Qwen3-0.6B (an artifact fitted on the model's own text matched
one fitted on real text at every λ tried, and was at least as good at the recipe's λ; one model,
one seed, one domain) is in [the-artifact.md](the-artifact.md#the-frame). On any other model
nothing has been measured: it gives you a first artifact, and §5 calibrates λ against it.

### Probe it

`lfa probe-artifact` measures, in minutes, how the artifact *prices* real update directions
against the model's real activations. It is a diagnostic: it gives no verdict, and a model is not
"supported" because its numbers look like another model's. It needs a **witness**: a LoRA adapter
trained on the model, whose update directions it prices.

**1. The calibration text.** The witness, and every run of §5, trains on the walkthrough's two
public Gutenberg books — *On the Origin of Species* (Darwin) and *Domestic Cookery* — cut into
documents of about 3,500 characters, every eighth held out. You need only the notebook's data cell,
not the notebook (whose full run is a 0.6B training session). This runs that one cell, which
downloads the two books (1.45 MB) and writes `data/darwin/{train,heldout}`,
`data/cookery/{train,heldout}` and `data/raw/`:

```python
import json
from pathlib import Path

nb = json.load(open("examples/two_domain_walkthrough.ipynb"))
cell = next(c for c in nb["cells"]
            if c["cell_type"] == "code" and "def prepare(" in "".join(c["source"]))
exec("".join(cell["source"]), {"DATA": Path("data")})   # strip_gutenberg, to_documents, prepare
```

It prints 193 Darwin training documents and 27 held out, 105 and 15 for cookery (2026-10-03).
Then have the base model write the question-and-answer supplement once, beside the training split
(in `data/darwin/train.supplement/`), where every later run on this text finds and reuses it. It
takes the recipe whose training side it is written from; §5 step 1 makes it:

```bash
lfa prepare-supplement --model your/model --corpus data/darwin/train --recipe recipes/your-model.yaml
```

**2. The witness: a separate unanchored run** (λ = μ = 0) on Darwin, 4 epochs, from a copy of
your recipe with `lambda_qkv`, `lambda_mlp` and `mu` at 0 (§5 step 1 makes the copies; this one is
`recipes/your-model-lam0.yaml`):

```bash
lfa init runs/witness --model your/model --artifact self-generated --recipe recipes/your-model-lam0.yaml
lfa train --workspace runs/witness --corpus data/darwin/train --epochs 4
```

**3. The probe:**

```bash
lfa probe-artifact --model your/model --artifact runs/your-model \
                   --adapter runs/witness/runs/stage1/final_model --output probe.json
```

**The witness adapter must come from an unanchored run.** An adapter trained anchored against an
artifact drifts into the directions that artifact underprices, so it makes the artifact look worse
than it is. Measured on Qwen3-0.6B's self-generated artifact (one artifact, RTX 3090,
2026-10-03): with an adapter trained anchored at λ = 100,000 against a sibling build of the same
corpus, the median SHAPE read 0.238 (linear sites) and 0.606 (MLP sites); with an unanchored
adapter, 0.056 and 0.108. The probe reads each adapter's run `config.json` and says when a witness
was anchored or its history is unknown. The recipe above — unanchored, 4 epochs, Darwin — is the
one the recorded numbers below were measured with, so keep to it when comparing against them; the
8-epoch control of §5 step 3 is a different witness.

**What it reports**, per probed site (default: five layers, first and last included) and as
medians by site class (linear = `pre_qkv` + `pre_o`; MLP = `pre_mlp`):

* **LEVEL** — the geometric-mean ratio of the artifact's price to the real price. A uniform
  mis-scaling, which λ absorbs. Reported, not judged.
* **SHAPE** — the spread of the log price ratios across directions: direction-dependent
  mispricing, which no scalar λ absorbs.
* **Floor** — SHAPE with a second, disjoint half of the real activations in place of the artifact:
  what real-vs-real noise alone gives.
* **Diagonal reference** — the real activations with every feature column independently permuted:
  every marginal kept, every correlation gone. What a perfect diagonal model of the real data
  would price; context, independent of the artifact.

**How to read it.** Read a new model's numbers against the numbers recorded for a model known to
work, **measured with the same witness recipe**. The recorded reference:

| Qwen3-0.6B, self-generated artifact at the recorded frame | floor | artifact | diagonal reference |
|---|---:|---:|---:|
| median SHAPE, linear sites (9) | 0.018 | 0.056 (3.1× floor) | 0.139 (7.8×) |
| median SHAPE, MLP sites (5) | 0.017 | 0.108 (6.5× floor) | 0.064 (3.9×) |

Witness recipe: one adapter from an unanchored run (λ = μ = 0) on the walkthrough's Darwin training
text, 4 epochs, the `qwen3-0.6b` recipe otherwise. Probe settings: WikiText-2 truth, `--n-real
30000 --n-model 30000 --n-random 4`, layers 0, 7, 14, 20, 27. One witness, one seed, RTX 3090,
2026-10-03; the probe took 17 s and peaked at 4,691 MiB by `nvidia-smi`.

* **A collapsed artifact** (a point mass, a degenerate corpus) shows as SHAPE many times these.
* **A scale or layer error** (the wrong model's artifact, layers out of order) shows in LEVEL,
  far from 1, and SHAPE may look ordinary.
* **The artifact against the diagonal reference is not a pass mark.** On the table above the MLP
  class reads *above* its diagonal reference. With a second unanchored adapter — from a different
  run, 20 epochs on another domain, probed over 3 seeds — the same artifact read 0.044 (linear) and
  0.034 (MLP) against references of 0.087 and 0.093, MLP now *below*. A comparison that flips with
  the witness says nothing on its own. More adapters (repeat `--adapter`) steady the medians.
* Pricing fidelity and preserved behaviour have been seen to come apart. The probe catches a
  broken artifact; it does not show that a sound one anchors well. Only §5 shows that.

## 5. Calibrate λ

**Do not port λ.** It is coupled to the LoRA rank, to the artifact, and to the corpus, and none of
those survives a change of model. The bundled 100,000 belongs to Qwen3-0.6B at rank 32 on its
self-generated artifact at the recorded frame, and means nothing elsewhere
([concepts.md](concepts.md) has why λ is coupled).

The protocol below picks λ by a stated rule, so the pick is not a guess. It uses perplexity only —
held-out domain perplexity and WikiText-2 — and the public text of the walkthrough notebook.

1. **Make the uncalibrated recipe.** Start from a copy of the bundled recipe with `model_id` and
   `calibrated_rank` set to your point ([recipes.md](recipes.md#writing-your-own)). Set
   `calibrated_artifact` to a placeholder such as `uncalibrated` until the sweep below has been
   read: the copy carries `self-generated`, which would silence the warning for a self-generated
   artifact of your model although nothing was measured on it, and the placeholder makes every
   stage say that λ was calibrated against something else, which is true. Keep
   `supplement_fraction` at the value you will train at: `train` mixes the supplement in during
   the sweep as it will afterwards, and λ is read in that mix.

   ```python
   import dataclasses
   from lfa import Recipe
   point = dataclasses.replace(Recipe.load("qwen3-0.6b"), name="your-model",
                               model_id="your/model", calibrated_artifact="uncalibrated")
   point.save("recipes/your-model.yaml")
   ```

   One copy per arm below, differing only in `lambda_qkv` = `lambda_mlp` (and `mu` for the
   control).

2. **The text and its supplement** are prepared once, in §4 ("Probe it", step 1): the
   walkthrough's two Gutenberg books from
   [`examples/two_domain_walkthrough.ipynb`](../examples/two_domain_walkthrough.ipynb)'s data cell,
   in `data/darwin/` and `data/cookery/`, and the base model's supplement beside
   `data/darwin/train`. Every run below reuses them.

3. **The unanchored control**, at 8 epochs (λ = μ = 0). It sets both ends of the scale: how far the
   domain can move, and what that costs on the general axis when nothing is preserved. (The
   probe's witness in §4 is a separate 4-epoch unanchored run: this one writes no 4-epoch
   checkpoint, and its learning-rate schedule is laid over 8 epochs.)

   ```bash
   lfa init runs/lam0 --model your/model --artifact self-generated --recipe recipes/your-model-lam0.yaml
   lfa train --workspace runs/lam0 --corpus data/darwin/train --epochs 8
   lfa evaluate --workspace runs/lam0 --corpus data/darwin/heldout
   ```

   `init` reuses the stored artifact; `evaluate` reports WikiText-2 and held-out domain perplexity,
   base and trained. It logs that a named corpus is "a fit, not a held-out measurement", because a
   named corpus is not split; `heldout/` was never in the training directory, so here it is held
   out. `lfa evaluate --compare-unanchored` would re-run each stage with λ = μ = 0 as a third
   column; one separate control run is cheaper when every arm is read against it.

4. **The λ ladder**, at a fixed rank (32) and a fixed **4 epochs** (the walkthrough's dose): three
   rungs a factor of 2–2.5 apart around a starting guess, each run as in step 3 with `--epochs 4`.
   With no starting guess, cover roughly a decade with rungs a factor of 2–3 apart. If the pick
   lands at an end of the ladder, extend it one rung outward and read again. Arms can run in
   parallel, one per card.
5. **The selection rule.** Among the ladder runs whose held-out domain improvement over the base
   model (the drop in held-out perplexity) is at least 90 % of the unanchored control's
   improvement, choose the one with the lowest WikiText-2 perplexity; ties within 1 % go to the
   smaller λ. Record the whole frontier — held-out domain perplexity against general perplexity —
   not only the pick. Pick from the frontier, never from the general axis alone: over-anchoring
   makes general perplexity look its best while domain quality collapses, and that failure is
   invisible unless you are watching the domain number.
6. **The dose check**: one 8-epoch run at the chosen λ. Record the epoch with the lowest held-out
   perplexity; the run's `training_history.json` holds the per-epoch validation curve.
7. **The stage-2 multiplier.** A two-stage chain, Darwin then cookery, at the chosen λ with
   `stage2_lambda_multiplier` 1 and 3, 4 epochs a stage:

   ```bash
   lfa init     runs/chain-m3 --model your/model --artifact self-generated --recipe recipes/your-model-m3.yaml
   lfa train    --workspace runs/chain-m3 --corpus data/darwin/train --epochs 4
   lfa extend   --workspace runs/chain-m3
   lfa train    --workspace runs/chain-m3 --corpus data/cookery/train --epochs 4
   lfa evaluate --workspace runs/chain-m3 --corpus data/darwin/heldout
   lfa evaluate --workspace runs/chain-m3 --corpus data/cookery/heldout
   ```

   Stage 2's supplement is written by that stage's entry model (the fused stage-1 model) when
   `train` runs, not by the base model. Choose the multiplier that keeps more of Darwin (held-out
   Darwin after stage 2, lower is better) while cookery's improvement stays within 90 % of the
   better arm's. Record both arms.
8. **Then, and only then, change rank.** Lower rank ⇒ lower λ, and the frontier has to be re-read.

μ = 0.05 is a reasonable starting backstop on a new model; it is a global shrinkage term rather
than a steering one, so it is far less sensitive than λ.

Scope everything you record: one model, one seed, one domain (Darwin; cookery for stage 2), a
perplexity frontier, the card and the date. No judged numbers: no judge is part of this package.

## 6. Write the recipe

A recipe is a joint operating point for one model: every value in it was tuned or measured
together ([recipes.md](recipes.md)). Write `lfa/recipes/<name>.yaml` from the uncalibrated copy of
§5, with:

* `name`, `model_id` (the Hub id exactly as users will pass it), `artifact: self-generated`;
* the adapter block (`lora_rank`, `lora_alpha`, `freeze_embed`, `full_weight`) as calibrated;
* `lambda_qkv` = `lambda_mlp` = the λ the selection rule chose; `mu`; the anchor schedule fields
  (`anchor_end_ratio`, `anchor_schedule`, `n_anchor_samples`) as calibrated;
* the run and optimizer fields (`epochs`, `learning_rate`, `lr_schedule`, `batch_size`,
  `gradient_accumulation_steps`, `warmup_steps`, `weight_decay`, `sequence_length`, `seed`,
  `keep_short_whole`, `val_fraction`, checkpointing) — the bundled recipe's values unless §3 or the
  smoke test showed the batch does not fit;
* `stage2_lambda_multiplier` from §5 step 7;
* `supplement_fraction` at the value λ was read in.

**The `calibrated_*` fields say what λ means.** `calibrated_rank` is the rank the ladder ran at.
`calibrated_artifact: self-generated` once the ladder has been read against the model's own
self-generated artifact, with `self_generated_frame` set to the frame it was built at (the recorded
frame, unless you calibrated at another). Every stage compares the run against these and says when
they differ, so they must be true.

**The header comment** states, plainly and with its scope, how λ was chosen: the protocol of §5, on
which text, at which rank and dose, on which card and date — one model, one seed, one domain. Say
what was not measured as plainly as what was.

**How it is found.** `Recipe.bundled_for(model_id)` matches a bundled YAML's own `model_id`
exactly, never its file name, and `lfa init` adopts that recipe when `--recipe` is omitted. A
workspace whose recipe names a different model gets a warning at `init` (a warning, not a refusal:
a local checkpoint path naming the same weights is a different id). With the recipe bundled, check
that `lfa init` on the model logs `Default recipe for this workspace`, and that `lfa train` over its
full-frame self-generated artifact logs no calibration note.

## 7. Tests to add

Copy the existing ones; each is short.

* **Adapter and geometry, CPU** — in the style of `tests/test_model_families.py`:
  * the real config on the meta device (no weights, no download): the adapter's name, the number of
    anchored sites (`3 × layers − 1 + 1`: layer 0's `pre_qkv` is skipped and `pre_lm_head` added),
    and every site's width (`_site_input_width` in `lfa/artifact/schema.py`);
  * the model's real class at a tiny config through build → sample → `init` → one training step on
    CPU. The suite's other fixtures are Llama, so this is the only test that runs the new class's
    own forward (Qwen3's attention carries `q_norm`/`k_norm`, for instance);
  * the layer group the build would choose at a stated RAM (`choose_layer_group_size`), and that the
    model gets a store entry of its own.
* **Tokenizer, `slow`** — add the id to `MODELS` in `tests/test_model_tokenizers.py`: document
  start (and that it encodes to one id), user opener, turn end, boundary markers. A template that
  injects a default system prompt is covered by a synthetic template in
  `tests/test_selfgen_generate.py`; add a case there for any new quirk.
* **GPU pipeline smoke, `gpu`** — add the id to the parametrization in `tests/test_pipeline_gpu.py`:
  a trial-frame build, `prepare-domain --supplement`, one epoch of `train`, `evaluate`, `fuse`, and a
  second `init` that must reuse the stored artifact. Before the model has a bundled recipe it runs on
  an uncalibrated copy, as that file shows. Run it one model at a time:
  `pytest tests/test_pipeline_gpu.py -m gpu -q -k <its parametrize id>` (e.g. `-k 1.7B`); give the
  new entry an id of its own in the parametrization's `ids`.
* **Recipe** — once the recipe is bundled, pin its operating point the way `tests/test_recipe.py`
  pins `qwen3-0.6b`, and that `Recipe.bundled_for` finds it.

The fast tier (`pytest -q`) runs everything but `gpu`, `slow` and `notebook`, and must stay green.

## 8. Acceptance checklist

A model is called supported only when every line holds. "Recorded" means written down with its
scope; it does not mean passed against a threshold.

- [ ] `get_adapter` places the model, and the CPU geometry test at the real config passes.
- [ ] The real class at a tiny config builds, samples and trains on CPU.
- [ ] The tokenizer test passes: document start (one id), user opener, turn end, boundary markers.
- [ ] A rendered supplement prompt has been read: thinking switch off, any default system turn
      known and accepted.
- [ ] The memory arithmetic is written down, and the GPU pipeline smoke passes on one 24 GB card at
      the recipe's batch geometry, its peak recorded.
- [ ] The full-frame self-generated artifact is built in the store (`lfa list-artifacts` shows it
      `built`), its meta checked, its generation time and layer-group line recorded.
- [ ] The unprompted corpus has been read: its language share and empty count recorded.
- [ ] The probe has run with a witness from an unanchored run, and its numbers are recorded beside
      the reference model's, with the witness recipe.
- [ ] The control, the λ ladder, the selection, the dose check and the stage-2 multiplier are
      recorded, with their scope.
- [ ] The recipe is bundled with true `calibrated_*` fields and a scoped header; `lfa init` on the
      model adopts it, and `lfa train` logs no calibration note.
- [ ] The tests of §7 are in, and the fast tier passes.

## 9. Worked example: Qwen3-1.7B

What integrating Qwen3-1.7B (`Qwen/Qwen3-1.7B`) took, step by step. Everything measured here is one
model, one seed, on an RTX 3090 (24 GB) in a host with 125 GiB of RAM, on 2026-10-03.

**The model.** Family Qwen3, `model_type` `qwen3` (`Qwen3ForCausalLM`) — the same family as the
bundled Qwen3-0.6B. Hidden size 2,048; 28 layers; 16 query and 8 key-value heads of width 128, so
`pre_o` is 2,048 wide; MLP 6,144; tied embeddings; vocabulary 151,936. It loads 1,720,574,976
parameters (3.44 GB in bfloat16); its 4.06 GB bfloat16 checkpoint stores the tied head twice, so
the download is larger than the model in memory.

**§1 Fit.** No new adapter: `LlamaLayoutAdapter` places it. `tests/test_model_families.py` proves it
on CPU: at the real config on the meta device the adapter is `llama_layout` with 84 anchored sites
(27 `pre_qkv` + 28 `pre_o` + 28 `pre_mlp` + 1 `pre_lm_head`), all 2,048 wide; a tiny
`Qwen3ForCausalLM` — attention with `q_norm`/`k_norm`, which act on the projections' outputs, so
they are neither sites nor targets of LoRA — builds, samples, initialises a workspace and trains a
step; and the two Qwen3 models get two store entries.

**§2 Tokenizer.** The same readings as Qwen3-0.6B, pinned by `tests/test_model_tokenizers.py`
(`slow`): document start `<|endoftext|>` (the declared `generation_config.bos_token_id`; one id),
user opener `"<|im_start|>user\n"`, turn end `<|im_end|>`, and the boundary markers hold
`<|endoftext|>` and `<|im_start|>`. Its template is Qwen3's, with the `enable_thinking` switch.

**§3 Memory.** `Σ d²` over the 84 sites is 352,321,536, 2.00× Qwen3-0.6B's. The fp32 load for the
build is about 6.9 GB. Reservoirs cost 2.29 GiB per layer, 64.1 GiB for all 28. The GPU pipeline
smoke (`tests/test_pipeline_gpu.py -k 1.7B`: a trial-frame build at `--n-raw 60 --max-new-tokens
128`, the supplement, one epoch at batch 6 × 512 with no accumulation, evaluate, fuse) passed in
482 s at a peak of 8,853 MiB, against Qwen3-0.6B's 323 s and 5,435 MiB on the same test. The peaks
are `nvidia-smi` sampled every 5 s over the whole test, so they include the fp32 model during the
trial build. Batch 6 × 512 did not run out of memory.

**§4 The build, and what the corpus showed.**

* **The first smoke failed.** The trial `init` was refused: 13 of 32 draws came out "empty or
  degenerate". The degeneracy filter measured repeated 8-grams of *whitespace-separated words*, and
  it rejected any draw with fewer than eight words — but Qwen3-1.7B writes much of its unprompted
  text in Chinese, where a whole paragraph is one or two such words. The fix (in
  `passes_filters`, `lfa/selfgen/generate.py`): a text with fewer than eight words has no 8-gram,
  so nothing in it repeats and only the length floor can reject it. The smoke passed after it.
* **The corpus language.** The full-frame corpus holds 2,500 documents and 5,058,082 tokens (median
  2,048 tokens a document, mean 2,023, minimum 72; 47 documents, 1.9 %, under 2,000 tokens; mean
  5,885 characters). **1,038 documents (41.5 %) are mostly CJK** (more than 30 % of characters in
  U+4E00–U+9FFF), against 6.2 % in Qwen3-0.6B's stored corpus. In the first 1,019 documents,
  22.9 % of the 128-token document heads had fewer than eight whitespace words, against 3.1 % over
  Qwen3-0.6B's whole corpus. So about two-fifths of this artifact's mass describes the model
  writing Chinese. That is what the model does from a bare document start; what it means for
  anchoring on an English domain has not been measured.
* **Generation** took 88 minutes for the 2,500 documents (for about 15 of those minutes, the GPU
  smoke runs shared the host on the other card), against about 80 minutes for Qwen3-0.6B.
* **The layer group.** With another job holding part of the host's memory, the build logged
  `layer_group_size=22 for Qwen/Qwen3-1.7B: ~50.4 GiB of reservoirs per group against 102.3 GiB available`
  — two corpus passes, of 22 layers and of 6 — and the first pass logged
  `Collected 65 sites, 600087 samples at the thinnest site` (22 layers × 3 sites, less layer 0's
  `pre_qkv`). At 118 GiB available the same arithmetic gives 25 (pinned in
  `tests/test_model_families.py`).
* **The trial recipe.** Before calibration the model ran on a copy of the `qwen3-0.6b` recipe naming
  `Qwen/Qwen3-1.7B` with `calibrated_artifact: uncalibrated`, and `train` said so at every stage.

**What differed from Qwen3-0.6B**, in short: nothing in the layout or the tokenizer; twice the
`Σ d²` and about 1.6× the smoke's peak memory; and a model that writes two-fifths of its unprompted
text in Chinese, which a word-counting filter could not read.

## 10. Extending to multimodal models (image and audio) — general advice, untested

**This section is design guidance. Nothing in it has been built or run, and nothing here claims
that it works.** It records how the framework's pieces map onto a vision-language or audio-language
model, so that whoever attempts one starts from the right questions.

* **What carries over.** The anchor acts on the *language backbone's* sub-module functions. A
  vision-language or audio-language model built as encoder → projector → decoder LM keeps the same
  decoder sites, so the loss, the sampler, the layer schedule and the recipe machinery apply to the
  decoder unchanged.
* **Loading.** Such models load with other auto classes (`AutoModelForImageTextToText`,
  `…ForConditionalGeneration`) and an `AutoProcessor`, where this package loads
  `AutoModelForCausalLM` and `AutoTokenizer` (`lfa/models.py`). The adapter must reach the decoder
  inside the wrapper (`model.language_model` or similar): an adapter whose `base_model` unwraps to
  the text decoder, and whose `lm_head_module` finds the head wherever the wrapper keeps it.
* **The encoder and the projector.** Freeze them — the common choice, and the only one this
  package's design covers — or extend anchoring to their sites, which needs p(h) estimates of the
  *encoder's* hidden states: new sites, a new collection path, and a new adapter surface.
* **p(h) under multimodal input.** The decoder's hidden-state distribution depends on whether image
  or audio tokens are present. Self-generation cannot produce images or audio, so the artifact
  corpus needs real (or separately generated) image-text or audio-text inputs. Consider fitting
  with modality-aware sampling: separate mixtures per modality, or recording which positions are
  media tokens and excluding or weighting them. Keep the format contract: a new head type or site
  family is a `format_version` bump, never a silent variant.
* **Data and the supplement.** `prepare-domain` and the supplement are text-only. A multimodal
  domain needs a data path for paired media, and a supplement writer that can see the media.
* **Memory.** Media tokens lengthen sequences — hundreds to thousands of tokens per image or per
  second of audio — and the encoder adds weights. Redo §3's arithmetic with the real sequence
  lengths; the activation and logits terms grow with them.
* **Evaluation.** Add a general axis per modality — captioning or VQA perplexity for vision, ASR
  word error rate for audio — next to the text axis, so that what each stage costs is read where it
  is paid.
* **Position encodings and masks** (multimodal rotary schemes, for one) are model-specific. The
  hooks read the modules' inputs, so they are unaffected, but the attention masks and media
  placeholders the processor builds must be passed through collection and training intact.
* **A suggested order.** Text-only use of the decoder first (this cookbook as written); then a
  frozen encoder with decoder anchoring on paired data; encoder anchoring last.
