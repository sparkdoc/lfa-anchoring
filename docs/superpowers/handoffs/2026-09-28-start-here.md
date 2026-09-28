# Start here: executing the usable-pipeline plan on a fresh machine

Written 2026-09-28 by the session that wrote the spec and plan, for an agent that has only this
git repository. Everything the work needs is committed; nothing from that session's memory is
required. Read this file, then the spec, then the plan.

| read | what it gives you |
|---|---|
| `docs/superpowers/specs/2026-09-28-usable-pipeline-design.md` | **the authority**: intent, every owner decision, the design. Where the plan and spec disagree, the spec wins. |
| `docs/superpowers/plans/2026-09-28-usable-pipeline.md` | 9 tasks with tests, code, commands and commit messages; its Global Constraints apply to every task |
| this file | machine setup, identity, branch, baseline, execution, and what not to do |

## What this repository is

`lfa-anchoring` is the pip package for Layerwise Function Anchoring (LFA): adapting a language
model to a new domain while penalising changes to each sub-module's output on hidden states
sampled from a fitted estimate `p(h)` of the model's own hidden-state distribution (the
"artifact"). It was ported from a private research repository that is **not available to you and
must not be named in shipped text**; everything the port needs from it is already in the code and
docs. `README.md` and `docs/concepts.md` explain the method.

## Decisions already made — do not re-ask them

All in the spec's §1. In short: no published artifacts at all (users build one by
self-generation; `--artifact` is required); built artifacts are reused automatically from a local
store (`--rebuild` opts out); data preparation offers the existing question-and-answer supplement,
framed as making the domain answerable and **never** as protecting skills; no rehearsal method is
added; construction documents are deleted; the implementation pass runs **no GPU tests** — GPU
verification is a separate follow-up (plan Task 9 writes its handoff), even on a machine with
GPUs, unless the owner says otherwise.

## Machine setup

1. **Clone and set the repo-local git identity before any commit.** The identity lives in
   `.git/config`, which a clone does not carry, and every commit in this repository's history is
   by the no-reply identity. A commit under a personal address would have to be rewritten.

```bash
git clone https://github.com/sparkdoc/lfa-anchoring.git && cd lfa-anchoring
git config user.name  sparkdoc
git config user.email sparkdoc@users.noreply.github.com
git log -1 --format='%an <%ae>'          # the plan's commits, for comparison
```

   The owner may already have a checkout: `git pull` it and still check `git config user.email`.

2. **Confirm you have the plan's commits**: `git log --oneline -5` must show the handoff commit
   (this file), `Plan: a usable end-to-end pipeline (9 tasks)...` and `Spec: a usable end-to-end
   pipeline...` on `main`.

3. **Environment** (Python ≥ 3.11, a virtualenv):

```bash
python -m venv .venv && . .venv/bin/activate
pip install -c constraints-tested.txt -e '.[dev,html]'
```

   `constraints-tested.txt` pins the tested stack (torch 2.10.0+cu128, transformers 4.57.6,
   accelerate 1.14.0, peft 0.18.1). If the interpreter has no development headers (`Python.h`),
   `lfa` warns once; set `LFA_SKIP_TOOLCHAIN_CHECK=1` to silence it. The plan's commands carry
   that variable; it is harmless where the headers exist.

4. **Keep scratch out of commits.** Subagent-driven development keeps its ledger and briefs in
   `.superpowers/`, and worktrees go in `.worktrees/` (already ignored). Add the first to the
   local exclude file, not to `.gitignore`:

```bash
echo ".superpowers/" >> .git/info/exclude
```

5. **Baseline.** On a branch (next section), run the fast tier:

```bash
LFA_SKIP_TOOLCHAIN_CHECK=1 pytest -q
```

   Expected: no failures. Recorded on 2026-09-27 at 61d27a6 (the code these docs-only commits
   sit on), Python 3.14: **621 passed, 15 skipped**. Counts differ slightly with the Python
   version and installed extras; failures do not. If anything fails before Task 1, stop and
   report it to the owner rather than working around it — the plan assumes a green baseline.
   `tests/test_release.py` scans the whole checkout, so a stray `.worktrees/` copy of the tree
   makes it fail: remove finished worktrees before running the suite from the main checkout.

## Branch and execution

- Work on a branch, never directly on `main`: `git switch -c usable-pipeline` (or a worktree
  under `.worktrees/usable-pipeline`). Merging, pushing and opening a PR are the owner's
  decisions at the end.
- Execute with `superpowers:subagent-driven-development` (the plan's header says so): one
  implementer per task, a task review after each, a whole-branch review at the end. If this
  machine's `~/.claude/agents/` defines `opus-high` and `fable-medium`, the owner's rule is:
  `opus-high` for implementers and task reviewers, `fable-medium` for the final whole-branch
  review; never `general-purpose` with a model override.
- Tasks run in order 1 → 9; each depends on the interfaces of the ones before it (listed in each
  task's **Interfaces** block). Do not parallelise implementers.
- Every task ends with the full fast tier green and a commit. Commit messages end with the
  `Co-Authored-By` line the plan gives.
- Rulings you make on anything the spec and plan leave open go in the ledger with the reason;
  do not stop to ask the owner about something the spec already decides.

## GPUs on this machine

The owner's machine has two RTX 3090s (24 GB each). The implementation pass does not use them
(spec §1, owner decision). The plan's GPU test (`tests/test_pipeline_gpu.py`) is written in Task 8
and deselected by default; Task 9's handoff describes the GPU verification, which the owner can
start on this same machine afterwards. On a 24 GB card the bundled recipe fits as shipped;
pin one card per job with `CUDA_VISIBLE_DEVICES`, never shard.

## What not to do

- Do not push, tag, delete tags, or touch the GitHub release — the owner does those (Task 9's
  handoff lists them).
- Do not commit under any identity but the no-reply one above.
- Do not name the private research repository, its claim ids (C12, C14, C15…), or the retired
  artifact in shipped text; `tests/test_wording.py` (Task 6) enforces it.
- Do not describe the supplement as protecting skills.
- Do not delete `docs/superpowers/`: the spec, plan and handoffs stay until the GPU follow-up
  finishes (its last step removes the folder).
