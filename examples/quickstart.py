#!/usr/bin/env python3
"""One domain, end to end: init a workspace, train it, read both axes, export the model.

This is the single-domain flow of Layerwise Function Anchoring (LFA) written out as Python, so
that the four things a run does are visible in one file:

1. :meth:`lfa.Workspace.init` makes a workspace: it records which model to adapt and which
   recipe to use, and builds (or reuses) the p(h) artifact and copies it in as
   ``artifacts/v1.pt``.
2. :meth:`lfa.Workspace.train` adapts the model to a corpus with the anchor switched on.
3. :meth:`lfa.Workspace.evaluate` reads the stage on both axes -- what it learned (held-out
   domain perplexity) and what it kept (WikiText-2) -- against the model it started from.
4. :meth:`lfa.Workspace.fuse` merges the adapter into a plain checkpoint anybody can load.

The same four steps are four commands in ``docs/quickstart.md``; nothing here is decided
differently from the way the CLI decides it.

Every number this prints is a perplexity computed locally. Nothing calls out to a model API.

Example::

    lfa prepare-domain ~/papers --out data/my_domain --supplement --model Qwen/Qwen3-0.6B
    python examples/quickstart.py \\
        --model Qwen/Qwen3-0.6B \\
        --artifact self-generated \\
        --corpus data/my_domain \\
        --out runs/my_domain

The first line turns your documents into a corpus of ``.txt`` files and has the model write the
question-and-answer supplement beside it, in ``data/my_domain.supplement/``, where training finds
it (``docs/preparing-your-data.md``). ``--artifact self-generated`` has the model write its own
text and fits p(h) on it; that costs hours once per model, and every later run over the same
model reuses the finished artifact from the local store (``~/.cache/lfa/artifacts``, or
``$LFA_ARTIFACT_STORE``). ``--artifact`` also takes the path to an artifact file -- another
workspace's ``artifacts/v1.pt``, say -- which is copied in instead of building one.

From an installed wheel, where there is no checkout to run a path from, the same script is
``python -m lfa.examples.quickstart``.

Add ``--compare-unanchored`` to train the control that says what the anchor bought: the same run
with lambda = mu = 0. It costs a second training run.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from lfa import Workspace


def windows(value: str) -> int | None:
    """``--n-windows``: a count, ``0`` for the whole split, or ``none`` to skip the axis.

    The general axis reads WikiText-2 from the Hugging Face Hub. On a machine with no network,
    pass ``none`` and read the domain axis alone.
    """
    if value.lower() == "none":
        return None
    return int(value)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", required=True,
                        help="a Hub id or a local checkpoint path")
    parser.add_argument("--artifact", required=True,
                        help="self-generated, or an artifact file")
    parser.add_argument("--corpus", required=True,
                        help="a file or directory of documents to adapt to")
    parser.add_argument("--out", required=True,
                        help="the workspace directory to create")
    parser.add_argument("--recipe",
                        help="a bundled recipe name or a path (default: the bundled recipe that "
                             "names this model, if there is one)")
    parser.add_argument("--epochs", type=int,
                        help="override the recipe's number of epochs")
    parser.add_argument("--device", default="cuda:0",
                        help="device to run on (default: %(default)s)")
    parser.add_argument("--n-windows", type=windows, default=100, metavar="N",
                        help="WikiText-2 windows for the general axis; 0 scores the whole split "
                             "and 'none' skips it (default: %(default)s)")
    parser.add_argument("--compare-unanchored", action="store_true",
                        help="also train the lambda = mu = 0 control and report it as a third "
                             "column")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    # 1. A workspace records which model it adapts (by id or path -- the checkpoint is not
    #    copied in) and carries the p(h) artifact and the history. It is what keeps the next
    #    domain from silently anchoring against the wrong p(h).
    workspace = Workspace.init(args.out, args.model, artifact=args.artifact,
                               recipe=args.recipe)

    # 2. One domain. The anchor draws its hidden states from the artifact, never from the corpus
    #    of any earlier domain -- that is what "data-free at adaptation time" means.
    entry = workspace.train(args.corpus, epochs=args.epochs, device=args.device)
    print(f"\nStage {entry['stage']} trained into {entry['output_dir']} "
          f"({entry['n_train_docs']} documents, {entry['n_val_docs']} held out)")

    # 3. Both axes at once. One of them alone says nothing: a run that only reports the domain it
    #    learned has not said what it gave up, and a run that only reports WikiText-2 has not said
    #    whether it learned anything.
    scores = workspace.evaluate(n_windows=args.n_windows,
                                compare_unanchored=args.compare_unanchored,
                                device=args.device)
    print("\n" + scores["table"])

    # 4. A plain checkpoint: no PEFT wrapper, loads with AutoModelForCausalLM.from_pretrained.
    exported = workspace.fuse()
    print(f"\nMerged checkpoint: {exported}")
    print(f"History:           {Path(args.out) / 'history.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
