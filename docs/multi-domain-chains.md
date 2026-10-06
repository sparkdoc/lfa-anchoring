# Chains: several domains, one after another

A second domain is not a second training run. Stage two anchors **base + A**, against a p(h) that
has been extended with A, at a harder λ than stage one used. Get any of those three wrong and the
run still completes, still reports a plausible loss, and quietly measures something else.

That is what a workspace is for: it owns which model the next stage adapts, which artifact version
that stage anchors against, and the order the two may be done in.

## The state machine

```
lfa init      →  stage 0, artifact v1, current model = base
lfa train A   →  stage 1  (pending_extend)
lfa extend    →  models/stage1_fused, artifact v2   (pending_extend cleared)
lfa train B   →  stage 2, λ × stage2_lambda_multiplier
lfa extend    →  models/stage2_fused, artifact v3
lfa train C   →  stage 3
```

`lfa regenerate-artifact` can stand wherever `lfa extend` does: it clears the same pending stage
and writes the next artifact version by a different route ([below](#the-regenerate-route)).

Training a **different** corpus while a stage is still pending is refused (`StageOrderError`):
without the extension the new stage would adapt the previous stage's starting model and anchor
against a p(h) that does not describe what it is anchoring. Training the **same** corpus again —
more epochs on the domain in progress — is the same stage, keeps the same λ, and gets its own run
directory (`runs/stage2_run2`) so neither run's history is overwritten. That latest run is the one
`lfa evaluate`, `lfa fuse`, `lfa extend` and `lfa regenerate-artifact` then read; `train` says so
when it starts and when it ends, and the earlier run stays on disk, unread by them. `--resume`
continues the latest run in its own directory instead.

## What `extend` does

Two things, and the next stage needs both:

* **Merge.** The stage's adapter is merged into the model it was trained over, giving
  `models/stage{N}_fused` — a plain checkpoint. That is what stage N+1 adapts.
* **Extend p(h).** The domain just learned is run through the *fused* model, its activations are
  fitted as a small mixture **in the base artifact's own PCA basis**, and the two mixtures are
  merged as a sample-weighted union:

  ```
  p(h) = (1 − a) · p_base(h) + a · p_domain(h),      a = n_domain / (n_base + n_domain)
  ```

  The union is exact — it is the distribution of "draw from the base pool with probability 1 − a,
  else from the domain pool" — so nothing is refitted, and the merged artifact is a sufficient
  statistic for the round after that as well.

Only the *new* domain is ever read. Each round adds `k_domain` components per site, so the cost per
round is flat in the number of rounds — against replay's requirement to rehearse every prior corpus
at every stage.

```bash
lfa extend --workspace runs/chain --need 40000 --k-domain 8
```

* `--need` (default 40,000): activations collected per site. This is the memory bill — see
  [faq.md](faq.md).
* `--k-domain` (default 8): components fitted per site, capped at one per 200 activations.

The collection runs under the same `keep_short_whole` setting the stage trained under, because
the new components have to describe the training stream the model actually saw. A caller using
`lfa.artifact.extend.extend_artifact` directly passes `keep_short_whole` itself; the workspace
reads it off the stage's history entry. Nothing else about the chunking is a setting: collection
never re-chunks, so it reads the epoch-0 cut.

## The regenerate route

`lfa regenerate-artifact` (`Workspace.regenerate_artifact`) is the alternative to `extend` between
two domains. It does the first half of `extend` the same way — merge the stage's adapter into
`models/stage{N}_fused` — and then, instead of extending p(h), fits a **fresh** p(h) from the fused
model's own text: 2,500 documents written from its document boundary, with no chat-format share,
fitted at 600k samples per site, K = 32, with no base component and no merge. The result is
`artifacts/v{N+1}.pt`, its corpus and manifest beside it as `v{N+1}.corpus.jsonl`, and the
workspace records `artifact_route: "regenerate"` (an `extend` records `"extend"`; every stage's
history entry carries the route its artifact came by). The next `train` then writes its supplement
with the same fused model that wrote the artifact's corpus.

```bash
lfa regenerate-artifact --workspace runs/chain       # the full recorded frame; no flags to shrink it
```

In a chain it is one top-level field, applied at every stage boundary:

```yaml
artifact: regenerate            # or extend, the default
domains:
  - {name: philosophy, corpus: data/domain_a}
  - {name: archaeology, corpus: data/domain_c}
```

A per-domain `artifact` key is refused at load: the route is chosen once for the chain.

**What the research runs found.** A three-domain chain anchored this way was, after its third stage,
as good as the chain anchored on the real-seed-corpus artifact on every judge, perplexity and skill
benchmark (at stage 2 it had one judged deficit, on the second domain, which stage 3 erased). The
text it was anchored on drifted toward the last domain learned: in the corpora written by the base
model, the first-domain model and the two-domain model, 0.5 / 68.2 / 2.5 % of the documents
mentioned the first domain, and 2.7 / 9.6 / 86.1 % the second — yet the anchor fitted on that
drifting text did not compound into a worse chain. Scope: one model (Qwen3-0.6B), rank 4, one
seed, three domains, judged by `gpt-5.6-luna@medium`. These are the research runs' measurements,
not this package's.

**λ on this route.** That chain ported λ at 2× from stage 2; the recipe's
`stage2_lambda_multiplier` is 3×. Neither is calibrated for your chain: the multiplier is a starting point on either route, to
be re-tuned on the two axes. A regenerated artifact is also not the calibrated one — it was written
by the fused stage model, not by the recipe's model — so from stage 2 on `train` warns that the
self-generated artifact describes the fused model's path rather than the recipe's model, and adds
that regenerating the artifact from each stage's model was measured on one configuration (rank 4,
one seed) and the stage multiplier is a starting point there, not a calibrated constant. The advice is the warning's: re-tune λ against held-out domain
perplexity.

## λ from stage 2 on

`Recipe.to_train_config(stage=N, …)` multiplies both λs by `stage2_lambda_multiplier` for every
`N ≥ 2`. It is a **level, not a compounding factor**: stage 3 anchors at the same multiple as stage
2, not at the square of it.

The shipped 3.0 was measured on one pair of domains (the walkthrough's Darwin, then cookery; one
seed, perplexity), where for both bundled models it was ahead of 1× on retention and on general
text and at least level on the new domain ([recipes.md](recipes.md)).

λ is coupled to the corpus
([recipes.md](recipes.md)), so a chain over your own pair of domains re-tunes it — judged on the
same two axes, on the domain the stage is learning and on what it is supposed to be keeping.
`lfa train --lambda X` sets one stage's λ exactly, with no multiplier on top: at stage 2,
`--lambda 3000000` is what the shipped recipe would have used, and the run's note names the
recipe's value beside the one given ([recipes.md](recipes.md#trying-another-λ)). A chain spec
has no per-domain λ; a stage trained with `--lambda` is run with `lfa train` and `lfa extend` by
hand.

## Running a whole chain

```bash
lfa chain domains.yaml --workspace runs/chain
```

```yaml
artifact: extend              # optional; `regenerate` is the other route (above)
domains:
  - name: philosophy          # the run directory under runs/ (optional)
    corpus: data/domain_a     # relative paths resolve against THIS file
    epochs: 15                # optional per-stage override
  - name: archaeology
    corpus: data/domain_c
```

Every domain is folded in before the next one starts — that is what a chain *is*. A spec asking for
anything else (`extend_between: false`) is rejected with the reason, at the top level and inside a
domain entry alike. To train several domains from the *same* starting point instead, run them as
separate workspaces; that is a different experiment, not a chain.

A domain entry takes exactly `name`, `corpus` and `epochs`; anything else is refused at load,
naming the spec file and the field — the same rule a recipe file lives under, and for the same
reason: a field nobody reads is worse than a refusal, because the run starts and does not do what
the file says. Two domains may not share a `name` either, since the name *is* the run directory,
and a `corpus` that is not there is refused rather than discovered later.

**Every domain is checked before the first one trains**, not as the chain reaches it: a spec's
cost is paid as a whole, so a typo in domain 3 that surfaced when domain 3 started would already
have spent two stages and two extensions.

[`examples/chain_three_domains.py`](../examples/chain_three_domains.py) runs the whole thing on
three generated stand-in corpora, offline, so you can watch the sequence before spending a day of
GPU time on it. Its corpora are filler and teach the model nothing — the script says so — so it
exercises the machinery and nothing else.

For a chain on text that means something, with numbers you can read,
[`examples/two_domain_walkthrough.ipynb`](../examples/two_domain_walkthrough.ipynb) runs two real
domains — two public-domain books it downloads itself — with each stage repeated at λ = μ = 0, so
what `extend` and the stage-2 λ actually bought is measured rather than asserted. 35.4 minutes on
one RTX 3090, recorded with the self-generated artifact at the recorded frame already in the store
and the supplement on, at demo scale rather than the recipe's. Its optional companion,
[`examples/what_the_anchor_does.ipynb`](../examples/what_the_anchor_does.ipynb), re-runs each
control at its own best number of epochs, which is not the same answer.

## Reading a chain

`lfa evaluate` reads the **last** stage against the model it started from: what this stage learned,
and what it kept relative to where it began. For the chain as a whole, read the history — every
stage's entry carries its corpus, its applied λ, its artifact version and the route that artifact
came by (`artifact_route`), its document counts, its supplement (`supplement`: the pairs file, pairs
used, the achieved fraction and the writer's hash) and its perplexities:

```python
import json
for entry in json.load(open("runs/chain/history.json")):
    print(entry["stage"], entry["corpus"], entry["lambda_applied"],
          entry["artifact_version"], entry.get("perplexity"))
```

To measure an *earlier* domain after a later stage, score it explicitly — hand `evaluate` that
domain's held-out text with `--corpus`, or call `lfa.evaluate.domain_perplexity` on the fused model
directly.

## What the paper found after a third domain

Stated as the paper states it: in a three-domain chain, **perplexity** on the earlier domains keeps
accumulating rather than degrading, while **judged answering** on those earlier domains falls after
the third stage — and that fall is carried by the question-and-answer **pairs** in the stage's
generated supplement rather than by the anchor. The paper calls that supplement double-edged: the
pairs are the interference, but a stage trained *without* the supplement answers on the earlier
domain **worse** on the same judge, so "drop the Q&A" is not the reading.

This package writes a supplement too (every stage's entry model writes its own
domain's pairs; [concepts.md](concepts.md#what-the-supplement-does-and-does-not-do) says what it
is for), but nothing here computes a judged score. Every number it reports is a perplexity computed
locally. Read that finding as a caution about what perplexity does and does not tell you about a
chain; `--no-supplement` exists, but the finding above is the reason not to read it as the fix.
