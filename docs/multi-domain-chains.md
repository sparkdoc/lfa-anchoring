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

Training a **different** corpus while a stage is still pending is refused (`StageOrderError`):
without the extension the new stage would adapt the previous stage's starting model and anchor
against a p(h) that does not describe what it is anchoring. Training the **same** corpus again —
more epochs on the domain in progress — is the same stage, keeps the same λ, and gets its own run
directory (`runs/stage2_run2`) so neither run's history is overwritten.

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

The collection runs under the frame the stage trained under (`keep_short_whole`), because the new
components have to describe the training stream the model actually saw. A caller using
`lfa.artifact.extend.extend_artifact` directly passes `keep_short_whole` itself; the workspace
reads it off the stage's history entry.

## λ from stage 2 on

`Recipe.to_train_config(stage=N, …)` multiplies both λs by `stage2_lambda_multiplier` for every
`N ≥ 2`. It is a **level, not a compounding factor**: stage 3 anchors at the same multiple as stage
2, not at the square of it.

The shipped 3.0 is a starting default rather than a calibrated constant. λ is coupled to the corpus
([recipes.md](recipes.md)), so a chain over your own pair of domains re-tunes it — judged on the
same two axes, on the domain the stage is learning and on what it is supposed to be keeping.

## Running a whole chain

```bash
lfa chain domains.yaml --workspace runs/chain
```

```yaml
domains:
  - name: philosophy          # the run directory under runs/ (optional)
    corpus: data/domain_a     # relative paths resolve against THIS file
    epochs: 15                # optional per-stage override
  - name: archaeology
    corpus: data/domain_c
```

Every domain is folded in before the next one starts — that is what a chain *is*. A spec asking for
anything else (`extend_between: false`) is rejected with the reason. To train several domains from
the *same* starting point instead, run them as separate workspaces; that is a different experiment,
not a chain.

[`examples/chain_three_domains.py`](../examples/chain_three_domains.py) runs the whole thing on
three generated stand-in corpora, offline, so you can watch the sequence before spending a day of
GPU time on it.

## Reading a chain

`lfa evaluate` reads the **last** stage against the model it started from: what this stage learned,
and what it kept relative to where it began. For the chain as a whole, read the history — every
stage's entry carries its corpus, its applied λ, its artifact version, its document counts and its
perplexities:

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
the third stage — and that fall is carried by the stage's generated question-and-answer supplement
rather than by the anchor.

This package has no QA supplement: the loader mixes nothing into the corpus you hand it, and
nothing here computes a judged score. Every number it reports is a perplexity computed locally.
Read that finding as a caution about what perplexity does and does not tell you about a chain, not
as a knob in this repository.
