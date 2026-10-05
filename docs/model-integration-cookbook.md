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
  parameters, 1.11 GiB; Qwen3-1.7B: 1.72 B, 3.20 GiB). A checkpoint can store a tied head twice, so
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

Measured anchors (RTX 3090, 24 GB), by two measures that are not read against each other. PyTorch's
`max_memory_allocated`: Qwen3-0.6B at the shipped recipe peaked at 8.63 GiB allocated (2026-09-07,
with a separately loaded teacher, which the default no longer loads). `nvidia-smi` memory.used,
which includes what PyTorch's caching allocator holds and so is an upper bound on what a run needs,
sampled every 10 s over `init`, `train` and `evaluate`: an unanchored 4-epoch Qwen3-0.6B run read
at most 7,325 MiB (about 7.2 GiB) while writing the supplement and at most 13,971 MiB (about
13.6 GiB) while training (2026-10-03). For Qwen3-1.7B, §9 has the GPU smoke and the training runs
by the second measure; a Qwen3-1.7B chain's second stage reached 19,291 MiB.

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
layer_group_size=28 for Qwen/Qwen3-0.6B: ~42.7 GiB of reservoirs per group against 103.9 GiB available
```

(Qwen3-0.6B on a host with 125 GiB of RAM, RTX 3090, 2026-10-04.)

Fewer layers per group means more passes over the corpus, not a different artifact: every site
draws its reservoir from its own seeded generator, so any group size gives the same artifact at a
fixed seed ([the-artifact.md](the-artifact.md#host-ram-the-layer-group)).

**Artifact size.** The stored statistics grow with `Σ d²` over the anchored sites. Qwen3-0.6B's 84
sites sum to 176,160,768 and its artifact is 110.0 MB (int8); Qwen3-1.7B's 84 sites, all 2,048
wide, sum to 352,321,536, twice that, and its artifact is 243.7 MB (both built 2026-10-03/04).

**Time.** On one RTX 3090 in a host with 125 GiB of RAM, the full-frame Qwen3-0.6B build took 231
minutes (3 h 51 min) from a cold store: 83 minutes of generation, about 27 minutes collecting hidden
states and about 2 h fitting the mixtures (2026-10-04; it ran alone on the host). Qwen3-1.7B's took
355.5 minutes (5 h 56 min), 87 of them generation (2026-10-03/04; §9). Plan for hours, and see §4 on
running detached.

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
* **The layer-group line and the collection line**, e.g. for Qwen3-0.6B (one group, 2026-10-04)
  `Collected 84 sites, 603008 samples at the thinnest site`.
  With several groups each logs its own: a group of `g` layers that includes layer 0 has `3g − 1`
  sites (layer 0's `pre_qkv` is an exact embedding lookup, not a fitted site), and the last group
  adds `pre_lm_head`.
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
one fitted on real text at every λ tried, and was at least as good at the paper's operating point; one model,
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
than it is. Measured on a Qwen3-0.6B self-generated artifact (one artifact, RTX 3090,
2026-10-03): with an adapter trained anchored at λ = 100,000 against a sibling build of the same
corpus, the median SHAPE read 0.238 (linear sites) and 0.606 (MLP sites); with an unanchored
adapter, 0.056 and 0.108. The probe reads each adapter's run `config.json` and says when a witness
was anchored or its history is unknown. The recipe above — unanchored, 4 epochs, Darwin — is the
one the recorded numbers below were measured with, so keep to it when comparing against them. It
is also §5's 4-epoch control; §5's recipe-dose control is a different witness.

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
| median SHAPE, linear sites (9) | 0.018 | 0.055 (3.1× floor) | 0.139 (7.8×) |
| median SHAPE, MLP sites (5) | 0.017 | 0.106 (6.4× floor) | 0.064 (3.9×) |

Witness recipe: one adapter from an unanchored run (λ = μ = 0) on the walkthrough's Darwin training
text, 4 epochs, the `qwen3-0.6b` recipe otherwise. Probe settings: WikiText-2 truth, `--n-real
30000 --n-model 30000 --n-random 4`, layers 0, 7, 14, 20, 27, probe seed 0. One witness, one
training seed, RTX 3090, 2026-10-04; the probe took 15 s. Over probe seeds 1 and 2 the artifact
read 0.055 and 0.053 (linear) and 0.102 and 0.095 (MLP). The same probe on Qwen3-0.6B read at most
4,691 MiB of `nvidia-smi` memory.used sampled every 5 s (2026-10-03): four samples over a run of
about 15 s, so a short spike could be missed. §9 has Qwen3-1.7B's numbers at the same witness
recipe.

* **A collapsed artifact** (a point mass, a degenerate corpus) shows as SHAPE many times these.
* **A scale or layer error** (the wrong model's artifact, layers out of order) shows in LEVEL,
  far from 1, and SHAPE may look ordinary.
* **The artifact against the diagonal reference is not a pass mark.** On the table above the MLP
  class reads *above* its diagonal reference. With a second unanchored adapter — from a different
  run, 20 epochs on another domain — another build of the Qwen3-0.6B artifact at the same frame
  read 0.044 (linear) and 0.034 (MLP) against references of 0.087 and 0.093, MLP now *below*
  (mean of 3 probe seeds, RTX 3090, 2026-10-03). A comparison that flips with
  the witness says nothing on its own. More adapters (repeat `--adapter`) steady the medians.
* The probe catches a broken artifact; it does not show that a sound one anchors well. Only §5
  shows that.

## 5. Calibrate λ

**Do not port λ.** It is coupled to the LoRA rank, to the artifact, and to the corpus, and none of
those survives a change of model. Each bundled λ belongs to its model at rank 32 on that model's
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

   One copy per arm below. The ladder's copies (step 4) differ only in `lambda_qkv` = `lambda_mlp`;
   the controls' copy (step 3) sets them and `mu` to 0; step 7's two copies carry the chosen λ and
   differ only in `stage2_lambda_multiplier` (1 and 3).

2. **The text and its supplement** are prepared once, in §4 ("Probe it", step 1): the
   walkthrough's two Gutenberg books from
   [`examples/two_domain_walkthrough.ipynb`](../examples/two_domain_walkthrough.ipynb)'s data cell,
   in `data/darwin/` and `data/cookery/`, and the base model's supplement beside
   `data/darwin/train`. Every run below reuses them.

3. **The unanchored controls** (λ = μ = 0), at two doses: the recipe's own epoch count — the dose
   you will train at, `epochs` in the recipe (15 in the bundled one) — and a short dose of 4 epochs.
   Together they are the unanchored model's dose curve: how far the domain can move, how soon,
   and what training on at the recipe dose costs on both axes when nothing is preserved. The
   4-epoch control is §4's probe witness (the same recipe copy, text and epochs), so evaluate
   `runs/witness` rather than training it again. Each dose is its own run, because the
   learning-rate schedule is laid over the run's epochs: a 15-epoch run passes through no 4-epoch
   state. An 8-epoch control is an optional third point on the curve.

   ```bash
   lfa init     runs/lam0-e15 --model your/model --artifact self-generated --recipe recipes/your-model-lam0.yaml
   lfa train    --workspace runs/lam0-e15 --corpus data/darwin/train --epochs 15
   lfa evaluate --workspace runs/lam0-e15 --corpus data/darwin/heldout
   lfa evaluate --workspace runs/witness  --corpus data/darwin/heldout
   ```

   `init` reuses the stored artifact; `evaluate` reports WikiText-2 and held-out domain perplexity,
   base and trained. It logs that a named corpus is "a fit, not a held-out measurement", because a
   named corpus is not split; `heldout/` was never in the training directory, so here it is held
   out. `lfa evaluate --compare-unanchored` would re-run each stage with λ = μ = 0 as a third
   column; separate control runs are cheaper when every arm is read against them. Record each
   run's validation curve as well: the run's `training_history.json` holds the per-epoch
   perplexity on the share of the training text that `train` held out (`val_fraction`).

   **Keep one WikiText-2 window count across everything you compare.** `evaluate` scores 100
   windows of 512 tokens (51,200 tokens) by default, and the whole test split with
   `--n-windows 0`. A figure at one count is not read against a figure at the other.

4. **The λ ladder**, at a fixed rank (32) and at the recipe dose (15 epochs, as the long control):
   three rungs a factor of 2–2.5 apart around a starting guess, each from its own recipe copy and
   run as the control above. With no starting guess, cover roughly a decade with rungs a factor of
   2–3 apart. Arms can run in parallel, one per card.

   ```bash
   lfa init     runs/lam1e6 --model your/model --artifact self-generated --recipe recipes/your-model-lam1e6.yaml
   lfa train    --workspace runs/lam1e6 --corpus data/darwin/train --epochs 15
   lfa evaluate --workspace runs/lam1e6 --corpus data/darwin/heldout
   ```

5. **The selection rule.** First read the long control. If it improves the held-out domain over
   the base model: among the ladder runs whose held-out domain improvement over the base model (the
   drop in held-out perplexity) is at least 90 % of the control's improvement, choose the one with
   the lowest WikiText-2 perplexity; ties within 1 % go to the smaller λ.

   **If the control over-trains at the recipe dose** — its held-out domain perplexity ends above the
   base model's — every rung that does not over-train worse than the control clears that 90 % bar,
   and the rule above reduces to the general axis alone. Then pick, on the recipe-dose frontier, the
   rung that is best on both axes (lowest held-out domain perplexity and lowest WikiText-2). If no
   rung is best on both, take the rung with the lowest held-out domain perplexity among the rungs
   within 1 % of the lowest WikiText-2 perplexity.

   Either way, if the choice lands at an end of the ladder, add a rung one step further out (a
   factor 2–2.5) and read again, until the ladder turns over and the choice is interior. Record the
   whole frontier — held-out domain perplexity against general perplexity, the controls included —
   not only the pick. Pick from the frontier, never from the general axis alone: over-anchoring
   makes general perplexity look its best while domain quality collapses, and that failure is
   invisible unless you are watching the domain number. For Qwen3-1.7B the control over-trained at
   15 epochs, and the rung best on both axes was λ = 1,000,000 (§9). For Qwen3-0.6B, on the same
   Darwin text, no rung was best on both and the fallback picked the same 1,000,000
   ([recipes.md](recipes.md#how-λ-was-chosen)). Two models, one corpus, one seed: that is not
   evidence that λ is independent of model size, and λ does not port, even within a family.
6. **The dose curve**: no new run. From the chosen rung's validation curve, record the epoch with
   the lowest validation perplexity and the value at the last epoch, beside the controls' curves.
   At the recipe dose the chosen rung's curve should not climb in the late epochs the way the
   control's does. If it does, the recipe's epoch count is too long for that λ: shorten `epochs` in
   the recipe copies, re-run the long control and the ladder at the new dose, and read them again
   (steps 3–5).
7. **The stage-2 multiplier.** A two-stage chain, Darwin then cookery, at the chosen λ with
   `stage2_lambda_multiplier` 1 and 3, the recipe dose (15 epochs) in each stage:

   ```bash
   lfa init     runs/chain-m3 --model your/model --artifact self-generated --recipe recipes/your-model-m3.yaml
   lfa train    --workspace runs/chain-m3 --corpus data/darwin/train --epochs 15
   lfa extend   --workspace runs/chain-m3
   lfa train    --workspace runs/chain-m3 --corpus data/cookery/train --epochs 15
   lfa evaluate --workspace runs/chain-m3 --corpus data/darwin/heldout
   lfa evaluate --workspace runs/chain-m3 --corpus data/cookery/heldout
   ```

   Stage 2's supplement is written by that stage's entry model (the fused stage-1 model) when
   `train` runs, not by the base model. Each evaluation compares stage 2 against its start, the
   fused stage-1 model. Both evaluations score WikiText-2 at the default 51,200 tokens, the ladder's
   count, so the arms read against each other and against the ladder on one measure. Choose the
   multiplier that keeps more of Darwin (held-out Darwin after stage 2, lower is better) while
   cookery's improvement stays within 90 % of the better arm's. Record both arms.
8. **Then, and only then, change rank.** Lower rank ⇒ lower λ, and the frontier has to be re-read.

μ = 0.05 is a reasonable starting backstop on a new model; it is a global shrinkage term rather
than a steering one, so it is far less sensitive than λ.

Scope everything you record: one model, one training seed, one domain (Darwin; cookery for stage
2), a perplexity frontier with its WikiText-2 window count, the card and the date. No judged
numbers: no judge is part of this package.

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
  second `init` that must reuse the stored artifact. Each model in it runs on its bundled recipe,
  which `--model` adopts. A model with no bundled recipe yet is smoke-tested on an uncalibrated
  copy of one (§5 step 1), passed with `--recipe`; the file's docstring shows how. Run it one
  model at a time:
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
- [ ] The controls (recipe dose and 4 epochs), the λ ladder at the recipe dose, the selection, the
      dose curve and the stage-2 multiplier are recorded, with their scope and WikiText-2 window
      count.
- [ ] The recipe is bundled with true `calibrated_*` fields and a scoped header; `lfa init` on the
      model adopts it, and `lfa train` logs no calibration note.
- [ ] The tests of §7 are in, and the fast tier passes.

## 9. Worked example: Qwen3-1.7B

What integrating Qwen3-1.7B (`Qwen/Qwen3-1.7B`) took, step by step. Everything measured here is on
Qwen3-1.7B except the Qwen3-0.6B figures each bullet names for comparison; one training seed;
RTX 3090 cards (24 GB, one run per card) in a host with 125 GiB of RAM; 2026-10-03 to 2026-10-05.

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
482 s, against Qwen3-0.6B's 323 s on the same test (2026-10-03). By `nvidia-smi` memory.used sampled
every 5 s over the whole test, the fp32 model of the trial build included, it sat mostly at 8.5–9
GiB, with a brief maximum of 14,781 MiB; Qwen3-0.6B's maximum on the same test was 10,235 MiB.
memory.used includes what PyTorch's caching allocator holds, so these are upper bounds on need.
Batch 6 × 512 did not run out of memory.

**§4 The build, and what the corpus showed.**

* **The first smoke failed.** The trial `init` was refused: 13 of 32 draws came out "empty or
  degenerate". The degeneracy filter measured repeated 8-grams of *whitespace-separated words*, and
  it rejected any draw with fewer than eight words — but Qwen3-1.7B writes much of its unprompted
  text in Chinese, where a whole paragraph is one or two such words. The fix (in
  `passes_filters`, `lfa/selfgen/generate.py`): a text with fewer than eight words has no 8-gram,
  so nothing in it repeats and only the length floor can reject it. The smoke passed after it.
* **The corpus.** The full-frame corpus the artifact was fitted on holds 2,500 documents and
  5,054,024 tokens (median 2,048 tokens a document, mean 2,022, minimum 3; 49 documents, 2.0 %,
  under 2,000 tokens; mean 5,865 characters), against Qwen3-0.6B's 5,060,966 tokens (median
  2,048, mean 2,024, minimum 3; 54 documents, 2.2 %, under 2,000 tokens; mean 7,684 characters).
  **1,043 of the 2,500 (41.7 %) are mostly CJK** (more than 30 % of characters in
  U+4E00–U+9FFF), against 154 (6.2 %) in Qwen3-0.6B's. All counted 2026-10-04, with each model's
  own tokenizer. So about two-fifths of this artifact's mass describes the model writing Chinese.
  That is what the model does from a bare document start; what it means for anchoring on an
  English domain has not been measured.
* **The build.** The build behind the stored artifact took 355.5 minutes (5 h 56 min) from a cold
  store and wrote 243.7 MB: generation 87 minutes for the 2,500 documents, none of them empty,
  against 83 minutes for Qwen3-0.6B; then the collection and fit (2026-10-03/04). It shared the
  host for most of its run with a second Qwen3-1.7B build, the 1.5 M-sample adequacy build below.
* **The layer group.** Two corpus passes, of 25 layers and of 3; the build logged (2026-10-03)
  `layer_group_size=25 for Qwen/Qwen3-1.7B: ~57.2 GiB of reservoirs per group against 115.0 GiB available`
  and the passes logged
  `Collected 74 sites, 600087 samples at the thinnest site` (25 layers × 3 sites, less layer 0's
  `pre_qkv`) and `Collected 10 sites, …` (3 layers × 3 sites, plus `pre_lm_head`). At 118 GiB
  available the same arithmetic gives 25 as well (pinned in `tests/test_model_families.py`).
* **The probe.** The witness followed §4's recipe on this model: one adapter from an unanchored
  run (λ = μ = 0) on the walkthrough's Darwin training text, 4 epochs, the `qwen3-0.6b` recipe's
  other fields (rank 32, batch 6 × 512). Over probe seeds 0, 1 and 2, at the probe's defaults, the
  stored artifact's median SHAPE read 0.058–0.067 (linear sites) and 0.137–0.139 (MLP sites),
  against floors of about 0.03 and 0.02 and diagonal references of about 0.14 and 0.06; LEVEL read
  about 0.97 (linear) and 0.76 (MLP). One witness, RTX 3090, 2026-10-04.
* **Sample adequacy.** A second build over the same corpus at 1.5 M samples per site (thinnest site
  1.28 M) priced no better than the 600k build at either class: 0.060–0.064 (linear) and
  0.136–0.141 (MLP) over the same three probe seeds and witness (RTX 3090, 2026-10-04). The
  recorded frame's 600k samples per site stands for this model.
* **Qwen3-0.6B, built twice.** The Qwen3-0.6B artifact of §4's table (231 minutes, 110.0 MB)
  and a second build of that model at the recorded frame, whose corpus was written under the
  word filter this section's first bullet replaced, read the same at that model's witness:
  median SHAPE 0.055 / 0.106, 0.055 / 0.102 and 0.053 / 0.095 (linear / MLP) at probe seeds 0, 1
  and 2, against 0.056 / 0.108, 0.056 / 0.105 and 0.053 / 0.096 (RTX 3090, 2026-10-04).
* **The trial recipe.** Before calibration the model ran on a copy of the `qwen3-0.6b` recipe naming
  `Qwen/Qwen3-1.7B` with `calibrated_artifact: uncalibrated`, and `train` said so at every stage.

**§5 Calibration.** Qwen3-1.7B on the stored artifact of the build above; the walkthrough's Darwin
text (cookery for stage 2); rank 32 (α 64), batch 6 × 512, μ = 0.05 on every anchored run and 0
on the control; one training seed; perplexity only; one run per RTX 3090; 2026-10-04 and
2026-10-05. Base model: WikiText-2 15.09 (the default 100 windows, 51,200 tokens), held-out
Darwin 21.45 (the 27 documents of `data/darwin/heldout`). The validation curve is the per-epoch
perplexity on the 10 % that `train` holds out of the training text.

* **The frontier at 15 epochs**, the dose the recipe trains at, with the unanchored control at 4,
  8 and 15 epochs (perplexity after training; change against the base model; WikiText-2 over the
  default 51,200 tokens):

  | run | epochs | WikiText-2 | held-out Darwin |
  |---|---:|---:|---:|
  | unanchored (λ = μ = 0) | 4 | 15.01 (−0.5 %) | 14.34 (−33.2 %) |
  | unanchored (λ = μ = 0) | 8 | 40.30 (+167 %) | 49.08 (+129 %) |
  | unanchored (λ = μ = 0) | 15 | 132.47 (+778 %) | 162.70 (+658 %) |
  | λ = 20,000 | 15 | 15.67 (+3.8 %) | 42.00 (+95.8 %) |
  | λ = 50,000 | 15 | 15.03 (−0.4 %) | 27.04 (+26.1 %) |
  | λ = 100,000 | 15 | 14.48 (−4.0 %) | 19.89 (−7.3 %) |
  | λ = 200,000 | 15 | 14.10 (−6.6 %) | 16.47 (−23.2 %) |
  | λ = 500,000 | 15 | 13.43 (−11.0 %) | 14.00 (−34.7 %) |
  | **λ = 1,000,000** | 15 | **13.15 (−12.8 %)** | **13.63 (−36.5 %)** |
  | λ = 2,500,000 | 15 | 13.31 (−11.8 %) | 13.82 (−35.6 %) |
  | λ = 5,000,000 | 15 | 13.91 (−7.8 %) | 14.24 (−33.6 %) |

* **The control over-trains; at λ = 1,000,000, 15 epochs are safe on this corpus.** Darwin's
  training text, about 189,000 tokens an epoch, is learned in two to four epochs: every unanchored
  run's validation curve is lowest at epoch 2 (10.83 to 11.01) and climbs after it, to 144.81 at
  epoch 15. At 4 epochs the unanchored run is a good result on its own (held-out Darwin −33.2 %,
  WikiText-2 −0.5 %); on a corpus this small a few epochs suffice. At 15 epochs it is far worse than
  the base model on both axes. At λ = 1,000,000 the 15-epoch run ends below the 4-epoch unanchored
  run on both axes (13.63 against 14.34 on Darwin, 13.15 against 15.01 on WikiText-2), and from
  epoch 10 on its validation curve stays between 12.69 and 12.76 (lowest at epoch 11; 12.71 at epoch
  15). Weaker anchors still over-train late: at 20,000 the validation curve is lowest at epoch 3
  (11.79) and ends at 38.83; at 100,000 it is lowest at epoch 5 (12.25) and ends at 18.45; at
  500,000 it is lowest at epoch 9 (12.65) and ends at 13.11.
* **How λ was chosen.** §5's first rule takes the unanchored control's domain improvement as its
  reference. At 15 epochs the control makes held-out Darwin worse, not better, so every rung
  qualifies and that rule would pick on WikiText-2 alone. §5's over-training branch applies
  instead (step 5, "If the control over-trains at the recipe dose"): the rung best on both axes,
  with the ladder extended outward while its top rung was best — 200,000 beat 100,000 on both
  axes, 500,000 and 1,000,000 beat 200,000, and 2,500,000 and 5,000,000 were worse than 1,000,000
  on both. **λ = 1,000,000 is the interior
  optimum**, ahead of 2,500,000 by 1.2 % on WikiText-2 and 1.4 % on held-out Darwin.
* **WikiText-2 below the base model.** Every rung from λ = 50,000 up ends with WikiText-2 below
  the base model's 15.09: by 0.4 % at 50,000, and by up to 12.8 % (13.15) at 1,000,000. That is
  what was measured; why has not been established.
* **The stage-2 multiplier.** A two-stage chain at the chosen λ — Darwin 15 epochs, `extend`,
  cookery 15 epochs — with `stage2_lambda_multiplier` 1 and 3, and the same pair at λ = 100,000.
  Each stage's change is against that stage's start (for stage 2, the fused stage-1 model);
  held-out cookery is its 15 held-out documents; WikiText-2 is over the default 100 windows
  (51,200 tokens), as on the frontier table:

  | stage-1 λ | multiplier | held-out cookery, stage 2 | held-out Darwin after stage 2 | WikiText-2 (100 windows, 51,200 tokens) after stage 2 |
  |---:|---:|---:|---:|---:|
  | 1,000,000 | 1× | 16.16 → 11.90 (−26.4 %) | 13.64 → 16.24 (+19.0 %) | 13.09 → 13.16 (+0.5 %) |
  | 1,000,000 | **3×** | 16.20 → 11.84 (−27.0 %) | 13.65 → 15.40 (+12.8 %) | 13.13 → 12.79 (−2.6 %) |
  | 100,000 | 1× | 22.35 → 20.38 (−8.8 %) | 19.43 → 29.76 (+53.1 %) | 14.36 → 14.53 (+1.2 %) |
  | 100,000 | 3× | 22.75 → 13.92 (−38.8 %) | 19.88 → 19.90 (+0.1 %) | 14.41 → 13.87 (−3.8 %) |

  At λ = 1,000,000, 3× came out ahead of 1× on all three axes: it kept more of Darwin (15.40
  against 16.24), it left WikiText-2 lower (12.79 against 13.16), and cookery ended slightly lower
  (11.84 against 11.90). §5's multiplier rule picks 3×. At λ = 100,000 3× was ahead on all three
  as well, by wider margins.
* **Run-to-run spread.** Repeated stage-1 runs of the same recipe and training seed (the ladder run
  and the two chains' first stages), on the frontier table's measures: at λ = 1,000,000, WikiText-2
  13.15, 13.09 and 13.13 (51,200 tokens) and held-out Darwin 13.63, 13.64 and 13.65; at λ = 100,000,
  held-out Darwin 19.89, 19.43 and 19.88, 2.4 % apart. The 1.2–1.4 % margin of 1,000,000 over
  2,500,000 is larger than the spread at 1,000,000 (0.5 % on WikiText-2, 0.15 % on Darwin) and
  smaller than the Darwin spread at 100,000; the 0.5 % cookery margin between the multipliers is
  smaller than the latter.
* **Training memory.** `nvidia-smi` memory.used, sampled every 10 s over a whole run (`init`,
  `train`, `evaluate`; batch 6 × 512). It includes what PyTorch's caching allocator holds, so it is
  an upper bound on what a run needs, and it is not comparable with PyTorch's allocated figure of
  §3. Anchored 15-epoch rungs (every rung, 20,000 to 5,000,000): about 12.7 GiB while training,
  maxima 13,033–13,056 MiB, and about 7 GiB while evaluating. Unanchored (8 and 15 epochs): about
  11.0 GiB while training, maximum 11,236 MiB. The two-stage chains' stage-2 training on the
  fused model: maxima 16,206–19,291 MiB (λ = 1,000,000: 16,206 at 1× and 19,291 at 3×, which it
  held for the last 5 minutes or so of stage 2; λ = 100,000: 17,511 at 1× and 16,842 at 3×).
  19,291 MiB is about 18.8 GiB (20.2 GB): more than a 16 GB card holds.
* **The recipe.** The bundled `qwen3-1.7b` recipe (`lfa/recipes/qwen3-1.7b.yaml`): `lambda_qkv` =
  `lambda_mlp` = 1,000,000; every other field as `qwen3-0.6b`, including `stage2_lambda_multiplier`
  3.0 and 15 epochs, which the runs above support on this model. The Qwen3-0.6B recipe was
  calibrated the same way on the same Darwin text and landed on the same λ; two models, one corpus, one
  seed, so that is not evidence that λ is independent of model size.

**What differed from Qwen3-0.6B**, in short: nothing in the layout or the tokenizer; twice the
`Σ d²`, and a GPU smoke whose `nvidia-smi` memory.used sat mostly at 8.5–9 GiB (maximum 14,781 MiB,
against Qwen3-0.6B's 10,235 MiB); a model that writes two-fifths of its unprompted text in Chinese,
which a word-counting filter could not read. The calibrated λ, 1,000,000, is the one the same
procedure gave Qwen3-0.6B on the same Darwin text.

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
