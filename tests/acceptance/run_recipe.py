#!/usr/bin/env python3
"""The acceptance run: the shipped Layerwise Function Anchoring (LFA) recipe, end to end.

This trains ``lfa``'s bundled ``qwen3-0.6b`` recipe on the paper's own domain-A corpus and scores
the resulting epoch-15 checkpoint on the paper's own two axes, so that the companion package can
be shown to land on the published operating point rather than merely to run.

Two instruments are reported, and the distinction matters:

* **The research instrument decides.** The paper's ``domain 8.76`` is the mean conditional
  perplexity of held-out *chat-formatted* domain Q&A, and its ``seed 16.36`` is WikiText-2 over
  the full test split. Both come out of the research code's ``scripts/eval_domain_perplexity.py``, which
  this script shells out to *in the research code's own virtualenv* -- the same call
  ``scripts/judge_search.sh`` made when the published numbers were produced. Held-out raw-prose
  perplexity is a different quantity and is never comparable with the Q&A one, so the acceptance
  band is checked against these two numbers and nothing else.
* **The companion's own metrics are reported beside them**, un-banded:
  :func:`lfa.evaluate.domain_perplexity` on the stage's held-out document split and
  :func:`lfa.evaluate.wikitext2_perplexity` with ``n_windows=0``. The WikiText-2 pair should
  agree closely with the research instrument (it is the same computation); the two domain numbers
  should *not* be expected to agree, because they measure different text.

the research code is read-only here. Every byte this script writes goes under ``--out``: the workspace,
the training run, and the scorer's output directory (``--output`` is passed explicitly so that
nothing lands in the checkout the corpus and the artifact came from).

Idempotent, because the training run is roughly an hour of GPU time: a workspace that already
carries a trained stage is not retrained, and a scoring directory that already holds a result is
not rescored. Delete ``--out`` to start over.

Usage::

    python tests/acceptance/run_recipe.py --out tests/acceptance/_runs/2026-09-07

Run it on ONE pinned card (the default ``--cuda-visible-devices 0``); the anchoring forward is
launch-bound at this model size and a second tenant on the card silently corrupts the timing.
"""

from __future__ import annotations

import argparse
import dataclasses
import importlib.util
import json
import logging
import os
import subprocess
import sys
import time
from pathlib import Path

#: The research checkout. The companion lives inside it, so its grandparent is the research code's root;
#: ``--the research code`` (or ``LFA_RESEARCH_ROOT``) overrides that for a checkout kept elsewhere.
DEFAULT_RESEARCH_ROOT = Path(__file__).resolve().parents[3]

MODEL_ID = "Qwen/Qwen3-0.6B"
RECIPE = "qwen3-0.6b"

#: The corpus the paper's run trained on: the raw Chalmers prose PLUS the generated Q&A
#: supplement at a 0.13 token share. The supplement is part of the shipped point, not an extra --
#: lambda is coupled to the corpus composition, so training the raw prose alone is a different
#: regularization problem and does not reproduce these numbers.
CORPUS_REL = "data/domain/chalmers_qa0.15"

#: The int8 p(h) artifact. The paper's e15 run anchored on the fp16 ``gmm1543k``; int8 is the
#: shippable form of the same estimate and costs ~1-1.5 pp of seed preservation, which the
#: acceptance band covers.
ARTIFACT_REL = "data/distributions/qwen3-0.6b-gmm1543k-int8/distribution_stats.pt"

#: The held-out direct-QA set the published domain number is measured on.
QA_REL = ("data/domain_perplexity_questions/chalmers/"
          "domain_perplexity_qa_direct_20260310_123856.jsonl")

#: Which checkpoint carries the shipped dose. The recipe trains 15 epochs, so ``final_model`` is
#: the same weights; the epoch-named directory is what the research runs were scored from.
CHECKPOINT = "checkpoint_epoch_15"

EXPECTED_FILE = Path(__file__).with_name("expected.json")


class MissingInput(RuntimeError):
    """A required read-only input is not on this machine (the test turns this into a skip)."""


# ==============================================================================================
# Inputs
# ==============================================================================================

def resolve_inputs(research: Path, corpus: str | Path | None = None,
                   artifact: str | Path | None = None,
                   qa_file: str | Path | None = None) -> dict[str, Path]:
    """Locate everything this run reads, or say precisely which piece is missing.

    Raises:
        MissingInput: naming the path that is absent. The corpora, artifacts and checkpoints are
            gigabytes and are not distributed with the companion, so a machine without the
            research checkout should skip this run rather than fail it.
    """
    research = Path(research).expanduser().resolve()
    wanted = {
        "research": (research, "the the research code checkout"),
        "python": (research / ".venv" / "bin" / "python", "the research code's virtualenv"),
        "scorer": (research / "scripts" / "eval_domain_perplexity.py", "the research code's scorer"),
        "metrics": (research / "scripts" / "comparison_metrics.py", "the research code's extractors"),
        "corpus": (Path(corpus).expanduser().resolve() if corpus else research / CORPUS_REL,
                   "the Chalmers corpus with its Q&A supplement"),
        "artifact": (Path(artifact).expanduser().resolve() if artifact
                     else research / ARTIFACT_REL, "the p(h) artifact"),
        "qa_file": (Path(qa_file).expanduser().resolve() if qa_file else research / QA_REL,
                    "the held-out direct-QA set"),
    }
    resolved = {}
    for key, (path, what) in wanted.items():
        if not path.exists():
            raise MissingInput(
                f"{what} is not at {path}. This run reads the research checkout (it is never "
                "written to); point --the research code at yours, or set LFA_RESEARCH_ROOT."
            )
        resolved[key] = path
    return resolved


def _load_module(path: Path, name: str):
    """Import a module from a file path without putting its directory on ``sys.path``.

    ``comparison_metrics`` is dependency-light by design, so it imports cleanly into this
    process; going through the file rather than the package keeps the research code's ``scripts/`` out of
    the import path, where its common module names would shadow the companion's own.
    """
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ==============================================================================================
# The run
# ==============================================================================================

def train_stage(out: Path, corpus: Path, artifact: Path, *, device: str,
                keep_short_whole: bool, val_fraction: float) -> tuple[object, dict, float]:
    """Train stage 1 of a fresh workspace on ``corpus``, or reuse the one already at ``out``.

    Returns ``(workspace, history_entry, wall_clock_seconds)``; the wall clock is ``0.0`` for a
    reused run, which is what tells the caller not to quote it as a timing.
    """
    from lfa import Recipe, Workspace

    workspace_dir = out / "workspace"
    if (workspace_dir / "workspace.json").exists():
        workspace = Workspace.open(workspace_dir)
        if workspace.history:
            print(f"[reuse] stage already trained at {workspace.history[-1]['output_dir']}")
            return workspace, workspace.history[-1], 0.0
    else:
        # `fetch=False` and a local artifact path: the published registry digests are
        # placeholders, so a path is the supported route to the real file. Artifact *extension*
        # (the multi-domain path) is not exercised by this run.
        workspace = Workspace.init(workspace_dir, MODEL_ID, artifact=str(artifact),
                                   recipe=RECIPE, fetch=False)

    recipe = Recipe.load(RECIPE)
    if val_fraction != recipe.val_fraction:
        recipe = dataclasses.replace(recipe, val_fraction=val_fraction)

    started = time.time()
    entry = workspace.train(corpus, recipe, keep_short_whole=keep_short_whole, device=device)
    return workspace, entry, time.time() - started


def score_with_research(inputs: dict[str, Path], checkpoint: Path, out_dir: Path,
                         env: dict[str, str]) -> dict:
    """Score ``checkpoint`` with the research instrument, in the research code's own virtualenv.

    This is the call ``scripts/judge_search.sh`` makes, with ``--output`` added so the result
    lands under this run rather than in the checkpoint directory. The checkpoint is a PEFT
    adapter directory and the research code's ``src/model_loading.py`` loads one directly, exactly as it
    did for the published runs.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    results = sorted(out_dir.glob("domain_perplexity_results_*.json"))
    if results:
        print(f"[reuse] research scoring already at {results[-1]}")
    else:
        command = [
            str(inputs["python"]), "scripts/eval_domain_perplexity.py",
            "--model", str(checkpoint),
            "--qa-file", str(inputs["qa_file"]),
            "--wikitext-perplexity",
            "--output", str(out_dir),
        ]
        print("[run]", " ".join(command))
        subprocess.run(command, cwd=inputs["research"], env=env, check=True)
        results = sorted(out_dir.glob("domain_perplexity_results_*.json"))
        if not results:
            raise RuntimeError(f"the scorer wrote no result into {out_dir}")

    payload = json.loads(results[-1].read_text())
    metrics = _load_module(inputs["metrics"], "research_comparison_metrics")
    return {
        "domain_direct_ppl": metrics.extract_domain_direct_ppl(payload),
        "seed_ppl": metrics.extract_seed_ppl_domain(payload),
        "results_file": str(results[-1]),
        "qa_file": str(inputs["qa_file"]),
    }


def score_with_companion(workspace, *, device: str) -> dict:
    """The companion's own two axes, reported beside the research instrument's.

    ``Workspace.evaluate`` scores the stage's base model and the adapted model on the stage's
    held-out document split (raw prose plus Q&A files, the tenth of the corpus the run never
    trained on) and on the full WikiText-2 test split. The WikiText-2 numbers are the same
    quantity the research instrument reports; the domain numbers are not -- they are held-out
    *documents*, not held-out chat-formatted Q&A.
    """
    scores = workspace.evaluate(n_windows=0, device=device)
    return {
        "domain_ppl_heldout_docs": scores["after"]["domain"],
        "base_domain_ppl_heldout_docs": scores["before"]["domain"],
        "wikitext2_ppl": scores["after"]["general"],
        "base_wikitext2_ppl": scores["before"]["general"],
        "table": scores["table"],
    }


# ==============================================================================================
# The band
# ==============================================================================================

def check_band(domain_ppl: float | None, seed_ppl: float | None,
               expected: dict | None = None) -> list[dict]:
    """Compare the research instrument's two numbers with the published band.

    Returns one row per axis: the measured value, the interval it had to fall in, and whether it
    did. A missing measurement is a failure, not a skip -- it means the scorer produced no number.
    """
    expected = expected or json.loads(EXPECTED_FILE.read_text())
    domain, seed = expected["domain_ppl"], expected["seed_ppl"]

    domain_low = domain["ref"] * (1 - domain["tol_rel"])
    domain_high = domain["ref"] * (1 + domain["tol_rel"])

    drift = None if seed_ppl is None else (seed_ppl - seed["base"]) / seed["base"] * 100
    drift_low = seed["drift_pct_ref"] - seed["tol_abs_pct"]
    drift_high = seed["drift_pct_ref"] + seed["tol_abs_pct"]

    return [
        {
            "name": "domain direct QA perplexity (research instrument)",
            "measured": domain_ppl,
            "reference": f"{domain['ref']} (seed 42) / {domain['ref_seed2']} (seed 1337)",
            "band": [domain_low, domain_high],
            "ok": domain_ppl is not None and domain_low <= domain_ppl <= domain_high,
        },
        {
            "name": "WikiText-2 seed drift %% vs base %.2f (research instrument)" % seed["base"],
            "measured": drift,
            "measured_ppl": seed_ppl,
            "reference": f"{seed['drift_pct_ref']}% (seed PPL {seed['ref']})",
            "band": [drift_low, drift_high],
            "ok": drift is not None and drift_low <= drift <= drift_high,
        },
    ]


def format_checks(checks: list[dict]) -> str:
    lines = []
    for check in checks:
        measured = "n/a" if check["measured"] is None else f"{check['measured']:.4f}"
        low, high = check["band"]
        lines.append(f"  [{'PASS' if check['ok'] else 'FAIL'}] {check['name']}: {measured} "
                     f"(band {low:.4f}..{high:.4f}; reference {check['reference']})")
    return "\n".join(lines)


# ==============================================================================================
# Entry point
# ==============================================================================================

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--the research code", default=os.environ.get("LFA_RESEARCH_ROOT")
                        or str(DEFAULT_RESEARCH_ROOT),
                        help="the research checkout to read (never written to)")
    parser.add_argument("--out", required=True,
                        help="where the workspace, the run and the scoring output go")
    parser.add_argument("--device", default="cuda:0", help="device for training and evaluation")
    parser.add_argument("--cuda-visible-devices", default="0",
                        help="pin the run to one card; ignored if the variable is already set. "
                             "Pass an empty string to leave the environment alone.")
    parser.add_argument("--corpus", default=None,
                        help=f"override the corpus (default <the research code>/{CORPUS_REL})")
    parser.add_argument("--artifact", default=None,
                        help=f"override the p(h) artifact (default <the research code>/{ARTIFACT_REL})")
    parser.add_argument("--qa-file", default=None,
                        help=f"override the held-out QA set (default <the research code>/{QA_REL})")
    parser.add_argument("--val-fraction", type=float, default=0.1,
                        help="documents held out of training (the recipe's own value)")
    keep = parser.add_mutually_exclusive_group()
    keep.add_argument("--keep-short-whole", dest="keep_short_whole", action="store_true",
                      help="keep short documents in every epoch (NOT the published frame)")
    keep.add_argument("--no-keep-short-whole", dest="keep_short_whole", action="store_false",
                      help="the research loader frame the published numbers were measured under")
    parser.set_defaults(keep_short_whole=False)
    parser.add_argument("--skip-research-scoring", action="store_true",
                        help="train and report the companion's own metrics only")
    parser.add_argument("--strict", action="store_true",
                        help="exit non-zero when a measured number falls outside the band")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    # The package logs through `logging` and configures no handler of its own (a library should
    # not). Without this an hour-long run is silent, and a stall is indistinguishable from
    # progress; with it, every epoch's losses and both perplexity computations are on the log.
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    # Pinned before torch is imported anywhere below, and passed to the scorer's environment too.
    if args.cuda_visible_devices and "CUDA_VISIBLE_DEVICES" not in os.environ:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices
    # The machine's network is flaky and everything this run needs is cached; a Hub check that
    # hangs would strand an hour-long run.
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

    inputs = resolve_inputs(args.research, args.corpus, args.artifact, args.qa_file)
    out = Path(args.out).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)

    workspace, entry, wall_clock = train_stage(
        out, inputs["corpus"], inputs["artifact"], device=args.device,
        keep_short_whole=args.keep_short_whole, val_fraction=args.val_fraction,
    )

    checkpoint = Path(entry["output_dir"]) / CHECKPOINT
    if not checkpoint.is_dir():                      # a shorter dose than the shipped one
        checkpoint = Path(entry["adapter"])

    research = None
    if not args.skip_research_scoring:
        research = score_with_research(inputs, checkpoint, out / "scoring", dict(os.environ))
    companion = score_with_companion(workspace, device=args.device)

    checks = check_band(None if research is None else research["domain_direct_ppl"],
                        None if research is None else research["seed_ppl"])

    from lfa import __version__ as lfa_version

    results = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "lfa_version": lfa_version,
        "companion_commit": _git_revision(Path(__file__).resolve().parents[2]),
        "research": str(inputs["research"]),
        "research_revision": _git_revision(inputs["research"]),
        "device": args.device,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "wall_clock_seconds": round(wall_clock, 1),
        "corpus": str(inputs["corpus"]),
        "artifact": str(inputs["artifact"]),
        "recipe": RECIPE,
        "keep_short_whole": entry["keep_short_whole"],
        "val_fraction": entry["val_fraction"],
        "n_train_docs": entry["n_train_docs"],
        "n_val_docs": entry["n_val_docs"],
        "epochs": entry["epochs"],
        "checkpoint": str(checkpoint),
        "research_instrument": research,
        "companion_instrument": companion,
        "checks": checks,
        "passed": all(check["ok"] for check in checks),
    }
    (out / "results.json").write_text(json.dumps(results, indent=2))

    print("\n" + companion["table"])
    print(f"\nResearch instrument ({'skipped' if research is None else research['results_file']}):")
    print(format_checks(checks))
    print(f"\nWrote {out / 'results.json'}")
    return 0 if (results["passed"] or not args.strict) else 1


def _git_revision(repo: Path) -> str | None:
    try:
        return subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"],
                              capture_output=True, text=True, check=True).stdout.strip()
    except (subprocess.CalledProcessError, OSError):
        return None


if __name__ == "__main__":
    sys.exit(main())
