# GPU verification: what the usable-pipeline pass did not run

Written 2026-09-30 by the session that implemented the usable-pipeline plan, for an agent (or the
owner) on a machine with a CUDA card that has only this git repository. The implementation pass
ran the CPU test tier only. Everything below needs a GPU, and none of it has been run against this
code yet. Nothing from the implementing session's memory is needed.

| read | what it gives you |
|---|---|
| this file | what to run on the card, what each run has to show, which documents the measurements go into, and the owner's items |
| `docs/superpowers/handoffs/2026-09-28-start-here.md` | machine setup: clone, repo-local git identity, the virtualenv, the fast-tier baseline. Follow its "Machine setup" steps 1, 3 and 5 before anything here |
| `docs/superpowers/specs/2026-09-28-usable-pipeline-design.md` | **the authority** on intended behaviour; §2.2 (store), §2.3 (durable build) and §5 (this handoff) matter here. Where the code and the spec disagree, report it; do not decide it |

## What you have, and who you are

- The repository at `https://github.com/sparkdoc/lfa-anchoring`. This work is on the branch
  `usable-pipeline`. Unless the owner has already merged it, verify on that branch
  (`git switch usable-pipeline`).
- **Git identity.** Every commit in this repository is by `sparkdoc <sparkdoc@users.noreply.github.com>`.
  Set it repo-locally before your first commit, as start-here step 1 says, and check with
  `git config user.email`.
- The owner's machine: two RTX 3090s (24 GB each). **Pin one card per job with
  `CUDA_VISIBLE_DEVICES`, never shard.** Every command below sets `CUDA_VISIBLE_DEVICES=0`. To use
  the second card for a job that can run alongside another, set it to `1`. The package's
  default device, `cuda:0`, then means that card.
- Commands assume the checkout's virtualenv is active (`. .venv/bin/activate`), so `lfa`,
  `pytest` and `python` are its own. Every command carries `LFA_SKIP_TOOLCHAIN_CHECK=1`, which is
  harmless where the Python development headers exist.
- Runs that write workspaces, corpora or notebook scratch go **outside the checkout**, in
  `~/lfa-verify`. Several of them write relative paths (`runs/...`, `lfa_demo/`) that are not
  gitignored, and the last step commits from the checkout.

```bash
REPO=$(pwd)                     # the checkout, with the venv active
mkdir -p ~/lfa-verify
```

## 1. State

- Written against commit `85c8fe80889462ac7cab7dece8d1577f68875c81` ("Examples and notebooks build
  their own artifact; a GPU pipeline test for the follow-up"), the last implementation commit on
  `usable-pipeline`. This handoff's own commit comes after it, and so may fixes from the
  whole-branch review. `git log --oneline 85c8fe8..` lists them. Read their messages before
  starting, because one of them may change something described here (the Ctrl-C message in
  item 3 is the likeliest).
- This pass ran the **fast tier only**. Its final run, at 85c8fe8 on Python 3.13.7:

```
$ LFA_SKIP_TOOLCHAIN_CHECK=1 .venv/bin/python -m pytest -q
661 passed, 5 skipped, 14 deselected in 114.35s (0:01:54)
```

  No failures. The 14 deselected tests are the 11 `gpu` tests, the 2 `notebook` tests and the
  1 `slow` test (`pyproject.toml` `addopts = "-m 'not gpu and not slow and not notebook'"`).
- **No GPU test has run against any commit of this branch**, and neither has any example script
  or notebook. The self-generation GPU tests last ran on 2026-09-26 on an RTX 2070, before this
  branch. Since then the store, the durable build, init's required `--artifact` and
  beside-the-corpus supplements have all been added, and those tests now go through them.
- The full-frame self-generated build (2,500 documents × 2,048 tokens, fit at 600k samples per
  site, K = 32) **has never been run end to end on any card**. The docs say "not timed" wherever
  its cost belongs. Item 5 replaces that.

## 2. The GPU tier

Run the fast tier first, as a baseline on this machine. `tests/test_release.py` scans the whole
checkout, so remove finished worktrees first (`git worktree list`; nothing under `.worktrees/`).

```bash
cd "$REPO"
LFA_SKIP_TOOLCHAIN_CHECK=1 pytest -q                                   # expect: no failures
CUDA_VISIBLE_DEVICES=0 LFA_SKIP_TOOLCHAIN_CHECK=1 pytest -m gpu -q     # 11 tests
```

`-m gpu` on the command line overrides the `addopts` deselection. It selects every GPU test:
`tests/test_gpu_smoke.py` (3), `tests/test_pipeline_gpu.py` (1) and `tests/test_selfgen_gpu.py` (7).
The first run needs the network: it downloads Qwen3-0.6B, and `evaluate` reads WikiText-2 from the
Hub. After that, `HF_HUB_OFFLINE=1` works.

**Stores.** Neither new-style GPU test file touches your real store. `tests/test_selfgen_gpu.py`
has an autouse fixture (`isolated_store`) that gives **each test its own `LFA_ARTIFACT_STORE`**, so
every test that calls `Workspace.init(..., artifact="self-generated")` builds its own small artifact
(`SMALL`: 16 raw + 4 chat documents × 128 tokens, K = 4) rather than reusing one.
`tests/test_pipeline_gpu.py` sets its own store under its `tmp_path`. Do not expect either to reuse
the full-frame entry from item 3, and do not expect item 3's timing to change because they ran.

What the new assertions prove. `tests/test_pipeline_gpu.py::test_the_pipeline_end_to_end_at_a_trial_frame`
runs the CLI, in order:

| step | assertion | what it proves on the card |
|---|---|---|
| `lfa init ws --artifact self-generated --n-raw 60 --max-new-tokens 128` | exit 0 | the store path builds a real artifact from a cold store: generation, durable corpus, fit, copy-in |
| `lfa prepare-domain src --out corpus --supplement --model Qwen/Qwen3-0.6B` | exit 0, and `corpus.supplement/*/supplement.jsonl` exists | data preparation has the base model write the supplement **beside the corpus** |
| `lfa train --workspace ws --corpus corpus --epochs 1` | the log has `Supplement reused` | `train` finds the beside-the-corpus supplement, because its writer (the base model) is the workspace's current model, and does not write a second one |
| same | the log has `different frame` | the recipe's note for a self-generated artifact built at a trial frame, naming the fields that differ (spec §2.1) |
| `lfa evaluate --workspace ws --n-windows 5`, `lfa fuse --workspace ws` | exit 0 | both axes read and the export writes, after a store-built artifact |
| a second `lfa init ws2` at the same trial frame | the log has `Reused the self-generated artifact` | the store is keyed on checkpoint and frame: the same frame reuses without generating |

In `tests/test_selfgen_gpu.py`, what is new in this pass is the store isolation above, plus the
removal of `calibrated_self_generated` from `_small_recipe()`. The three tests that call
`Workspace.init(..., artifact="self-generated", selfgen=SMALL)` (`test_init_self_generated_...`,
`test_a_stage_trains_with_the_self_written_supplement_mixed_in` and
`test_a_two_stage_chain_under_regenerate_...`) now build through the store, so they are this
branch's first on-card check of `obtain_self_generated`.

`tests/test_gpu_smoke.py` is unchanged: bfloat16 load, one pinned-device stage, and the CUDA
generator in the mixture fit.

If a GPU test fails, stop and report it to the owner with the failing assertion and the log tail.
Do not loosen a test to make it pass.

## 3. The full-frame build: timed, interrupted once, resumed

This is the build every later item reuses. Run it on one card, with the host otherwise quiet
during the fit. The fit chooses its layer group from the host RAM available when it starts, and
its wall time is one of the numbers being measured. Keep the default store (do not set
`LFA_ARTIFACT_STORE`), because the examples and notebooks in item 4 look there.

**Before.** `lfa list-artifacts`. If it already lists a finished `Qwen/Qwen3-0.6B  2500 documents x
2048 tokens, K=32` entry, init would reuse it in seconds. To time a real build, add `--rebuild` to
the **first** command below only. That moves the old entry aside to
`<entry>.replaced-<timestamp>/`; it does not delete it.

The log has no timestamps (`lfa` logs `%(message)s` to stderr), so stamp each line. The stamping
stage ignores SIGINT and `tee -i` ignores it too, so Ctrl-C reaches only `lfa`, and its
interrupt message still lands in the log:

```bash
cd ~/lfa-verify
stamp() { trap '' INT; while IFS= read -r line; do printf '%(%H:%M:%S)T %s\n' -1 "$line"; done; }

# run 1: interrupt it
{ time CUDA_VISIBLE_DEVICES=0 LFA_SKIP_TOOLCHAIN_CHECK=1 \
    lfa init runs/verify --model Qwen/Qwen3-0.6B --artifact self-generated; } \
  2>&1 | stamp | tee -i -a build-1.log; echo "exit ${PIPESTATUS[0]}"
```

1. Watch the per-batch line, `self-generated corpus: n/2500 documents (e empty)` (32 documents per
   batch at the default batch size). After about a third, roughly 800/2500, press **Ctrl-C**.
   Expected: exit status 130, no traceback.
2. **The Ctrl-C message: record what it prints.** Spec §2.3 says the line names the store entry and
   says the same command resumes. At 85c8fe8 the handler (`lfa/cli.py`, the `except
   KeyboardInterrupt` branch of `main`) prints a generic message: it says an interrupted
   self-generated build resumes when the same `lfa init` or `lfa build-artifact` command is run
   again, and it names no path. The whole-branch review may have changed it since, so check it
   against the spec rather than against a fixed string. Does it name the entry directory? Does it
   say the same command resumes? Put the verbatim line in your report to the owner. If it still
   names no entry, report that as a spec deviation. It is a code change, not part of this
   verification.
3. After the interrupt, confirm the state on disk:
   - `lfa list-artifacts` shows the entry as `in progress: n/2500 documents` with its path. Call
     that path `ENTRY` below.
   - `ls -a "$ENTRY"`: `entry.json`, `corpus.jsonl.partial` and `corpus.jsonl.progress.json` are
     there. `.lock` and `artifact.pt` are not (the lock is released on the way out).
   - `runs/verify` does not exist: a failed init removes the directories it created, and never
     touches the store.
   - Optional, and worth doing: `cat "$ENTRY/corpus.jsonl.progress.json"` (`rows_written` and each
     share's `batch_index`) and `wc -l "$ENTRY/corpus.jsonl.partial"`. The line count is at least
     `rows_written`.

```bash
# run 2: the same command, to the end
{ time CUDA_VISIBLE_DEVICES=0 LFA_SKIP_TOOLCHAIN_CHECK=1 \
    lfa init runs/verify --model Qwen/Qwen3-0.6B --artifact self-generated; } \
  2>&1 | stamp | tee -i -a build-2.log; echo "exit ${PIPESTATUS[0]}"
```

4. **It resumes at the next batch.** Near the top, `Resuming the self-generated corpus at
   <ENTRY>/corpus.jsonl from N documents on disk`, where N is `rows_written` from step 3. The
   first per-batch line then counts on from N (N plus that batch's kept documents, at most 32),
   not from zero. A `Dropping k rows written after the last progress record` line is legitimate
   if Ctrl-C landed mid-append. Exit 0, and the last line is `Next: lfa train --workspace ...
   --corpus <your documents>`.
5. **The finished corpus has 2,500 rows.** Run `wc -l "$ENTRY/corpus.jsonl"` (expected 2500) and
   `python -c "import json,sys; print(json.load(open(sys.argv[1]))['counts'])"
   "$ENTRY/corpus.jsonl.manifest.json"` (expected `raw` 2500, `chat` 0). The `.partial` and
   `.progress.json` files are gone.
6. **Record the costs separately**, from the stamps and the `time` output:
   - generation = run 1 from its first line to the Ctrl-C, plus run 2 from its first line to the
     `self-generated corpus: 2500/2500` line (both include a model load);
   - fit = from the `layer_group_size=...` line to `Self-generated artifact stored at ...`;
   - each run's `real` time.
   Also record the **host-RAM layer-group choice**, which is the `layer_group_size=K for
   Qwen/Qwen3-0.6B: ~X GiB of reservoirs per group against Y GiB available` line, verbatim. Record
   the artifact's size too (the `Wrote ... (NNN MB)` line), and the card
   (`nvidia-smi --query-gpu=name,memory.total --format=csv`).
7. **The store and the workspace.** `lfa list-artifacts` now shows the entry `built` with a date
   and a size. `runs/verify/artifacts/` holds `v1.pt`, `v1.corpus.jsonl` and its manifest.
   `runs/verify/workspace.json` records `artifact_id` `self-generated:<12 hex>`.

Optional, for spec §2.2's lock: while run 2 is still generating, run the same `lfa init` with a
different workspace path (`runs/verify-2`) in a second terminal. It must refuse at once (exit 2)
with `... is being built by process <pid> (lock <ENTRY>/.lock) ...`. The pid must be run 2's, and
`runs/verify-2` must not be left behind.

## 4. Examples and notebooks

**Item 3 must finish first.** Both examples and both notebooks then reuse the stored full-frame
artifact. A cold store would put a multi-hour build inside them. In the notebook tier it would
also fail: `tests/test_notebook.py` executes each cell with `timeout=5400` (90 min), and the
walkthrough's artifact cell would be doing the whole build. If you must run the notebooks without
the default store, set `LFA_ARTIFACT="$ENTRY/artifact.pt"`, which the setup cell reads to skip
the build.

**Card geometry.** On a 24 GB card (the owner's RTX 3090s) the bundled recipe fits as shipped
(`batch_size: 6`, `gradient_accumulation_steps: 1`). On an 8 GB card, use batch 3 × gradient
accumulation 2 through a recipe copy passed with `--recipe`:

```bash
sed -e 's/^batch_size: 6$/batch_size: 3/' \
    -e 's/^gradient_accumulation_steps: 1$/gradient_accumulation_steps: 2/' \
    "$REPO/lfa/recipes/qwen3-0.6b.yaml" > ~/lfa-verify/qwen3-0.6b-8gb.yaml
```

The copy loads as recipe `qwen3-0.6b` with the same artifact and calibration fields. The
examples take `--recipe PATH`. The notebooks load the bundled recipe by name, so run them on a
24 GB card.

### 4a. The notebooks: execute, then save them with their outputs

First as tests, which checks what they must produce. Note that the companion test executes the
walkthrough again before itself, so this costs about three walkthroughs' worth of GPU time:

```bash
cd "$REPO"
CUDA_VISIBLE_DEVICES=0 LFA_SKIP_TOOLCHAIN_CHECK=1 pytest tests/test_notebook.py -m notebook -q
```

`pytest` runs them in a `tmp_path` and keeps nothing, so to re-record the shipped outputs, execute
both in one scratch directory and write them back over `examples/`. Run this from a neutral path,
because the outputs print absolute paths. Unlike the test tier, this has no per-cell timeout:

```bash
mkdir -p ~/lfa-verify/nb && cd ~/lfa-verify/nb
CUDA_VISIBLE_DEVICES=0 LFA_SKIP_TOOLCHAIN_CHECK=1 python - "$REPO" <<'EOF'
import sys, time, nbformat, nbclient
from pathlib import Path
examples = Path(sys.argv[1]) / "examples"
for name in ("two_domain_walkthrough.ipynb", "what_the_anchor_does.ipynb"):   # this order
    path, t0 = examples / name, time.time()
    notebook = nbformat.read(path, as_version=4)
    nbclient.NotebookClient(notebook, timeout=None, kernel_name="python3",
                            resources={"metadata": {"path": "."}}).execute()
    nbformat.write(notebook, path)
    print(f"{name}: {(time.time() - t0) / 60:.1f} min")
EOF
```

Record both wall times: they replace the "19 min" family in item 5. Then read the new outputs
before trusting them:

- The walkthrough's artifact cell prints the stored `artifact.pt` and its store entry, with no
  build. At a warm store the load time is seconds.
- The recipe cell prints `off-calibration warnings: none` for the full-frame artifact.
- Each training stage logs a supplement.
- The five-model table is present.
- `git -C "$REPO" diff --stat examples/` shows only the two notebooks.
- Grep the saved notebooks for your home directory. The store path is printed, and if it is
  `~/.cache/...` it carries the user name. The last recording set the precedent, and walkthrough
  cell 0 states it: "the filesystem paths in them rewritten to a neutral `/tmp/lfa-nb-run/…`;
  no number is altered". Do the same, or change that sentence to say what you did.

**Settle `EPOCHS` before you keep a recording.** `EPOCHS = 4` (walkthrough cell 2, repeated in the
companion's setup cell 2) was chosen from an 8-epoch held-out curve measured with the retired
artifact and no supplement (walkthrough cell 10). The artifact and the training mix have both
changed, so measure the curve again. Use the Darwin text the recording above downloaded, in a
workspace of its own:

```bash
cd ~/lfa-verify
CUDA_VISIBLE_DEVICES=0 LFA_SKIP_TOOLCHAIN_CHECK=1 lfa init runs/epochs8 \
    --model Qwen/Qwen3-0.6B --artifact self-generated                # reuses the stored artifact
CUDA_VISIBLE_DEVICES=0 LFA_SKIP_TOOLCHAIN_CHECK=1 lfa train --workspace runs/epochs8 \
    --corpus nb/lfa_demo/data/darwin/train --epochs 8 2>&1 | tee epochs8.log
grep "Held-out:" epochs8.log                                         # one line per epoch
```

This is the anchored stage-1 run the notebook makes: the bundled recipe, with the supplement.
Only the epoch count differs. One cost to expect: `train` writes its own supplement here
(`Writing the supplement for ...`). It looks in this workspace's `supplements/`, then beside the
corpus, and the notebook's copy is in neither place. The supplement is seeded, so it should be
the same text, but the write takes generation time. The epoch with the lowest held-out
perplexity is the new `EPOCHS`.

- **If it is 4:** keep the recording. Walkthrough cell 10 still gets the new curve (item 5).
- **If it is not 4:** set `EPOCHS` to that epoch in **both** notebooks' cell 2. The companion's
  cell 2 says it must equal the walkthrough's. Change every place the prose depends on 4
  epochs:
  - walkthrough cell 0, the table row `| epochs | 4 | 15 |`;
  - walkthrough cell 10;
  - walkthrough cell 12, "same 4 epochs" and "4 epochs is the *anchored* run's best dose";
  - walkthrough cell 18, its `lfa train ... --epochs 4` line;
  - walkthrough cell 30, "a quarter of its epochs";
  - companion cell 3;
  - companion cell 7, its three "4-epoch" mentions.

  Then delete `~/lfa-verify/nb/lfa_demo` and run the recording snippet again. A
  recording made at the old dose does not match the notebook's own code.

### 4b. The example scripts

Any directory of a dozen or more long documents works as the domain. The walkthrough's downloaded
Darwin chapters are at hand after 4a, in `~/lfa-verify/nb/lfa_demo/data/darwin/train`.

```bash
cd ~/lfa-verify
CUDA_VISIBLE_DEVICES=0 LFA_SKIP_TOOLCHAIN_CHECK=1 lfa prepare-domain \
    nb/lfa_demo/data/darwin/train --out data/darwin --supplement --model Qwen/Qwen3-0.6B
CUDA_VISIBLE_DEVICES=0 LFA_SKIP_TOOLCHAIN_CHECK=1 python "$REPO/examples/quickstart.py" \
    --model Qwen/Qwen3-0.6B --artifact self-generated --corpus data/darwin \
    --out runs/quickstart --epochs 4
CUDA_VISIBLE_DEVICES=0 LFA_SKIP_TOOLCHAIN_CHECK=1 python "$REPO/examples/chain_three_domains.py" \
    --model Qwen/Qwen3-0.6B --artifact self-generated --epochs 1 --out runs/three_domains
```

(On an 8 GB card add `--recipe ~/lfa-verify/qwen3-0.6b-8gb.yaml` to both scripts.) What to see:

- Both scripts log `Reused the self-generated artifact built <date> from <ENTRY>` and do not
  generate.
- `prepare-domain` writes `data/darwin.supplement/<12 hex>/supplement.jsonl`.
- `quickstart.py`'s training logs `Supplement reused`, and no recipe note, because this is the
  full frame.
- The chain's three stages finish. Each stage logs `Writing the supplement for ...`, because the
  synthetic corpora have none prepared. From stage 2 on, the writer is the fused model before
  it.
- Both exit 0.

These are smoke checks of the scripts; no number from them goes into the docs.

## 5. Docs to update from the measurements

Every replacement names the card, and every number that came from a notebook is labelled with
card, artifact (self-generated, the recorded frame: 2,500 documents × 2,048 tokens, K = 32) and
supplement (on). Do not extrapolate a 3090 measurement to an 8 GB card; say which card it was.

**The build cost ("not timed").** At 85c8fe8:

| file:line | now says |
|---|---|
| `README.md:52` | "hours on an 8 GB card, not timed" |
| `docs/quickstart.md:82` | "Several hours on an 8 GB card; not timed" |
| `docs/faq.md:111` | "takes several hours on an 8 GB card; not timed" (the FAQ's timing table sits just above; add the full frame as a row: generation and fit separately) |
| `docs/the-artifact.md:75` | "Several hours on an 8 GB card for the full frame; not timed" |

`docs/the-artifact.md` around line 102 quotes a layer-group choice ("7 ... about 24 GiB
available"). Set it beside the value your log recorded, naming the host.

Afterwards this must print nothing (this handoff and the plan are in `docs/superpowers/`, which
goes in item 7):

```bash
grep -rn "not timed" "$REPO" --include='*.md' --include='*.py' --include='*.ipynb' \
    --include='*.yaml' --include='*.toml' \
  | grep -v -e '/\.venv/' -e '/docs/superpowers/' -e '/\.superpowers/' -e '/\.worktrees/'
```

**The notebooks' recorded outputs, and every number quoted from them.**

- Both notebooks' **"Recorded outputs."** markdown cell (cell 1 of each) says the outputs were
  recorded on 2026-09-08 with the since-retired published artifact, with no supplement. Rewrite it
  for the new recording: date, card, self-generated artifact at the recorded frame, supplement on.
- Walkthrough cell 0: "about twenty minutes on one RTX 3090", the table rows "wall clock per
  training run | 2-4 min" and "whole notebook | 19 min", and the sentence "The outputs were
  recorded on 2026-09-08 on one RTX 3090, with the ...". While you are there, check "One CUDA card
  with about 12 GB free" and "version 0.1.0 here" against the run.
- Walkthrough cell 28: the note that the end-to-end figure is the recorded time minus 3.5 minutes
  of controls now in the companion. Re-derive it from the new outputs, or drop it if the new
  recording makes it moot.
- **The notebooks' own narrative about their outputs.** Some markdown cells, and one code
  comment, describe the 2026-09-08 outputs: they quote numbers and generations and draw
  conclusions from them. Each one falls into one of two groups.

  **Rewrite from the new outputs.** This covers any prose that describes an output the new
  recording produces, wherever it sits: directly under that output, in a section introduction
  that looks back at it, or in a closing recap. A kept version would contradict an output
  somewhere in the same notebook and cite text no reader can find. Where the new outputs no
  longer support a conclusion (the anchored arm fitting the new domain less well, `N1`/`Q1` as a
  pair, `P3` testing nothing), say what they do show, including when that cuts against the
  anchor, as the current cells do.
  - companion cell 0: "Every cell after it carries the output recorded on 2026-09-08,
    unaltered." Give the new recording's date, or drop it.
  - companion cell 3: the stage-1 control's trainer note ("lowest at epoch 2 and 12 % above it
    by epoch 4"), "The stage-2 control ends only 5 % above its own minimum" and "its curve turns
    at epoch 3". Re-read all three from the new outputs of the walkthrough's two control runs.
  - companion cell 7: "A 2-epoch run is not simply a truncated 4-epoch run" (cell 5 derives the
    dose from the new control curve), and "Reporting only the 4-epoch controls would have
    overstated the anchor by the difference between the two control rows here". That second one
    is a direction; confirm it in the new cell-6 table.
  - companion cell 8: "the base model pays for them too (see `G2`)". Confirm it in the new `G2`.
  - companion cell 11: from "**What the controls cost, stated before it is used.**" onward.
    That includes the table numbers it quotes ("17.45 on Darwin against the base model's
    30.12", "18.87 for that control against 16.09 anchored, base 17.80", "A · Darwin: 27.02 for
    the control, 18.92 anchored", "B · cookery 12.99 against 15.13"), and its quotations of the
    `B1`, `B2`, `G1`, `G2`, `I1`, `I2` and `A1`/`A2` generations. Two parts are the exception:
    "under plain greedy decoding ... opened with" in the `G2` paragraph belongs to the dated
    record below, and the rest of that paragraph (the penalised answers) is rewritten. The
    rewrite also covers the claim that "exactly one contains a repeated span" across the thirty
    generations, and "The count" (better on four, worse on two, four indistinguishable).
  - companion cell 12, "Outward transfer is only a virtue if it is selective": a section
    introduction that recaps section 2's stage-1 control ("asked what photosynthesis is, it
    answers correctly for one sentence and then drifts into invented Victorian biology; asked
    for two sentences for a ten-year-old, it returns a passage about island faunas with a
    fabricated statistic"). It reads the new `G1` and `I1`, not the old ones.
  - companion cell 15: the quoted `N1`, `Q1`, `Q2`, `Q3`, `P1`, `P2`, `P3` and `N2` generations,
    the word counts ("165 words", "thirteen words", "thirty of them"), and the per-probe verdicts
    and summary.
  - companion cell 16: its recap of `N1` ("another 165 words", "answers in four sentences and
    stops") and of `Q1`.
  - walkthrough cell 10: the 8-epoch held-out curve "(18.46, 17.64, 17.08, **16.84**) ... 16.93,
    17.64, 17.94, 18.25", given as "this exact configuration" and used to justify `EPOCHS = 4`.
    Replace it with the curve from `epochs8.log` (item 4a), dated and labelled with the
    artifact and supplement.
  - walkthrough cell 24, comparison 2: "The unanchored run reaches the domain *harder*; the
    general column is where it pays for it". This is a direction read off the old table.
    Confirm it holds in the new one, or reword it.

  **Keep as a dated record.** These describe a decode of the 2026-09-08 models **without**
  repetition controls. The new recording decodes with them and cannot reproduce that decode.
  They are the reason the decode settings were chosen, so they stay, marked in the text as
  recorded 2026-09-08, with the earlier artifact, under plain greedy decoding. The exception: if
  you re-decode the new models without the controls, you may replace them with what that decode
  shows, and say so.
  - companion cell 8, "**Why those two repetition controls.**": "an earlier decode of these same
    ten probes had every trained arm looping, one of them five times over".
  - companion cell 11, first paragraph ("**The decoder first ...**"): the looping `B1` answer
    ("Put a pound of butter in a large cask ..." five times). Of the `G2` paragraph, only its
    first observation belongs here: "Under plain greedy decoding all three stage-1 models opened
    with 'Three prime numbers greater than 10 are'".
  - companion cell 9, the code comment `repetition_penalty=1.15,   # the standard mild band; 1.1
    left some answers still looping`.

  This list came from reading every markdown cell and every code-cell comment of both notebooks
  at 85c8fe8. If you find another passage, sort it by the same test: can the new recording show
  what it describes?
- `README.md:145-149`: the Darwin perplexities "17.45 → 18.92 ... 31.45", "Recorded 2026-09-08
  with the since-retired published artifact on the raw books alone; a run today ... will differ",
  and "19 minutes of training and tables on one RTX 3090".
- `docs/quickstart.md:216-223`: "19 minutes on one RTX 3090" and the "recorded outputs were made
  on 2026-09-08 with the since-retired published artifact" sentence.
- `docs/multi-domain-chains.md:161`: "19 minutes on one RTX 3090".
- `pyproject.toml:57` (the `notebook` marker's description, "~19 minutes") and
  `tests/test_notebook.py:22-25` (module docstring: "about 19 minutes ... roughly another 20 for
  the companion").

Once no output names the retired artifact, **switch the shipped-text scan back on for notebook
outputs.** `tests/test_wording.py` today scans notebook cell *sources* only (`_shipped_files`
yields `"".join(cell["source"])` per cell). That is a recorded exemption: the old outputs still
name the retired artifact, which its module docstring says. With new outputs:

- make `_shipped_files` also yield each code cell's output text (stream `text`, and `text/plain`
  in `data`), and drop the outputs exemption from the module docstring;
- the pin test `test_a_notebooks_sources_are_scanned_and_its_recorded_outputs_are_not` is
  vacuous. It asserts that an output-only string (`"112.8 MB"`) is absent from the scanned
  sources, but never that the string is present in the raw notebook, and the re-recording
  removes it anyway. Replace it with a test that the outputs **are** scanned: pick a string that
  appears only in a recorded output of the new walkthrough, assert that it is in the raw `.ipynb`
  JSON, and assert that it is among the scanned texts;
- confirm `test_the_scan_sees_a_planted_hit` still passes, then run the fast tier.

Run the full fast tier after the doc and test edits (expect no failures), and commit them together
under the repository's identity.

## 6. Owner items

These are outward-facing. The owner does them, or explicitly says to.

- **Merge and push.** Merge `usable-pipeline` into `main` (at 85c8fe8, `main` is an ancestor of the
  branch, so this is a fast-forward), then `git push origin main`.
- **The retired artifact's tag and release.** The remote still has the tag `artifacts-v1`
  (`git ls-remote --tags origin` listed it on 2026-09-30, at `e0f3b1b`, which is also where
  `v0.1.0` points; deleting `artifacts-v1` leaves `v0.1.0` alone). The same day,
  `gh release view artifacts-v1 --repo sparkdoc/lfa-anchoring` answered "release not found". Check
  again, because a release may exist under another name, or `gh` may not have been authenticated.
  Decide first; then:

```bash
git push origin :refs/tags/artifacts-v1      # the remote tag
git tag -d artifacts-v1                      # the local tag
# and the GitHub release, if one exists: the repository's Releases page, or `gh release delete`
```

- **The parked item: per-site reservoir generators** in `lfa/artifact/collect.py`. One
  `torch.Generator` is made per collection pass (`collect.py:222`) and shared by every site's
  `SiteStats` (`collect.py:224`), so how layers are grouped changes each site's reservoir draws,
  and with them the fitted mixtures (not the exact moments). Seeding a generator per site (from
  the seed and the site's key) would make the layer group a memory choice only. This is out of
  scope for this pass (spec §6). If it is done, it changes every artifact built after it, and
  `docs/the-artifact.md`'s "The group size is part of the build" paragraph changes with it.

## 7. Last step

Once every item above is done, delete the construction documents and commit:

```bash
cd "$REPO"
git rm -r docs/superpowers
LFA_SKIP_TOOLCHAIN_CHECK=1 pytest -q        # expect: no failures
git commit -m "Remove the usable-pipeline construction documents; GPU verification done"
```

`tests/test_wording.py` lists `docs/superpowers` in `EXEMPT_DIRS` and names it in its docstring.
Both are harmless once the folder is gone; drop them in the same commit if you like.

## What not to do

- Do not push, tag, delete tags, or touch GitHub releases unless the owner has said to (item 6).
- Do not commit under any identity but the no-reply one.
- Do not shard a model across the two cards, and do not run two GPU jobs on one card.
- Do not name the private research repository, its claim ids, or the retired artifact in shipped
  text. `tests/test_wording.py` enforces it, and after item 5 it enforces it in notebook outputs
  too.
- Do not describe the question-and-answer supplement as protecting skills. It makes the domain's
  knowledge answerable.
- Do not fix a failing GPU test, a spec deviation, or a surprising measurement by editing the
  test or the docs to match. Report it to the owner with the log. This rule stops measurements
  being bent to fit the prose. It does not stop prose being rewritten to fit new measurements:
  rewriting the notebooks' narrative, and every doc number quoted from them, to say what the new
  recording shows is item 5's task. What must not happen is picking or re-running a recording
  until it matches the old prose.
- Do not delete `docs/superpowers/` before items 1–6 are done.
