"""The ``lfa`` command line: argv in, a call into the library out.

This module parses and dispatches. Every subcommand is one call into :class:`lfa.workspace.
Workspace` or into a pipeline function (:func:`~lfa.artifact.build.build_artifact`,
:func:`~lfa.artifact.fetch.fetch_artifact`, :func:`~lfa.seed_corpus.prepare_seed_corpus`,
:func:`~lfa.prepare_domain.prepare_domain`), and nothing is decided here that the library does
not already decide the same way for a caller who imports it. That is deliberate: the stage
ordering, the lambda multiplier, the loader frame and the device policy are the parts of
Layerwise Function Anchoring (LFA) that are easy to get silently wrong, and a CLI that
re-implemented any of them would be a second place for them to drift.

The one thing this layer does own is how a *refusal* reads. The library raises rather than
guesses -- a chain out of order, a workspace that is not there, a device map that would shard the
model, an artifact that is not published yet, a recipe field that does not validate -- and each of
those is a message written to be read by the person who typed the command, usually ending in the
command to run instead. So every exception the library raises *at the user*
(:data:`USER_FACING_ERRORS`) is caught at the top level and printed as one line with exit status
2; a traceback would bury the sentence that says what to do. Anything *not* in that tuple is a
bug, and a bug should show its traceback.

Run ``lfa --help``, or ``lfa <subcommand> --help``, for the flags.
"""

from __future__ import annotations

import argparse
import logging
import sys

from .artifact.build import build_artifact
from .artifact.fetch import (ArtifactNotPublished, ChecksumMismatch, DownloadFailed,
                            fetch_artifact,
                             list_artifacts)
from .evaluate import DatasetUnavailable
from .models import (DEFAULT_DEVICE, TEACHER_MODES, MissingBuildToolchain,
                     NoTrainableParameters, ShardingRefused)
from .prepare_domain import MissingExtra, prepare_domain
from .seed_corpus import SourceUnavailable, prepare_seed_corpus
from .train import ResumeSourceHasNoAdapter
from .workspace import StageOrderError, Workspace, WorkspaceNotReady

__all__ = ["main"]


def _lfa_version() -> str:
    """The installed package version, imported lazily so the CLI never depends on import order."""
    from . import __version__

    return __version__

#: Every exception the library raises **at the user** rather than at a caller: a chain out of
#: order, a workspace that is not there or is already there or is not ready for what was asked, a
#: device map that would shard the model, an artifact that is unpublished or arrives corrupted, a
#: dataset that cannot be reached (the seed corpus, or WikiText-2 for the general axis), a resume
#: with no adapter to continue, a student with nothing trainable, and any value a recipe, a chain
#: spec or a flag fails validation on. Each carries a message
#: written to be read, so each is reported as one line rather than as the last line of a traceback.
#: Anything outside this tuple is a bug and keeps its traceback.
#:
#: They are named classes rather than the bare ``RuntimeError`` they subclass, deliberately: torch
#: raises ``RuntimeError`` for real faults -- a CUDA OOM, a shape mismatch -- and those must keep
#: their traceback rather than be collapsed to a line.
USER_FACING_ERRORS = (
    StageOrderError, WorkspaceNotReady, ShardingRefused, ArtifactNotPublished, ChecksumMismatch,
    DownloadFailed,
    SourceUnavailable, DatasetUnavailable, ResumeSourceHasNoAdapter, NoTrainableParameters,
    MissingBuildToolchain, MissingExtra, FileNotFoundError, FileExistsError, ValueError,
)


# ==============================================================================================
# Argument types
# ==============================================================================================

def _windows(value: str) -> int | None:
    """``--n-windows``: a count, ``0`` for the whole split, or ``none`` to skip the axis."""
    if value.lower() == "none":
        return None
    try:
        return int(value)
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"expected a number of windows, 0 for the whole split, or 'none' to skip the "
            f"general axis; got {value!r}")


# ==============================================================================================
# Handlers -- one call each
# ==============================================================================================

def _open(args) -> Workspace:
    """The workspace a workspace subcommand acts on (``--workspace``, default: here)."""
    return Workspace.open(args.workspace)


def _init(args) -> int:
    workspace = Workspace.init(args.path, args.model, artifact=args.artifact,
                               recipe=args.recipe, artifact_id=args.artifact_id)
    # `Workspace.init` already LOGS that the workspace was created, and the CLI configures
    # logging, so printing the same sentence here showed it twice. Say the next step instead.
    print(f"Next: lfa train --workspace {workspace.path} --corpus <your documents>")
    return 0


def _fetch_artifact(args) -> int:
    print(fetch_artifact(args.artifact_id, args.dest))
    return 0


def _list_artifacts(args) -> int:
    for entry in list_artifacts():
        state = "published" if entry["published"] else "not published yet"
        print(f"{entry['id']}  {entry['model_id']}  {entry['size_mb']} MB  "
              f"{entry['kind']}  [{state}]")
    return 0


def _train(args) -> int:
    entry = _open(args).train(
        args.corpus, args.recipe, epochs=args.epochs, device=args.device,
        allow_sharding=args.allow_sharding, resume=args.resume,
        full_weight=args.full_weight, teacher_mode=args.teacher_mode,
    )
    loss = entry["final_loss"]
    cost = f" (final loss {loss:.4f})" if loss is not None else ""
    print(f"Stage {entry['stage']} written to {entry['output_dir']}{cost}")
    return 0


def _extend(args) -> int:
    print(_open(args).extend(need=args.need, k_domain=args.k_domain, device=args.device))
    return 0


def _evaluate(args) -> int:
    print(_open(args).evaluate(args.corpus, compare_unanchored=args.compare_unanchored,
                               n_windows=args.n_windows, device=args.device)["table"])
    return 0


def _fuse(args) -> int:
    print(_open(args).fuse(args.out))
    return 0


def _chain(args) -> int:
    entries = _open(args).chain(args.spec, device=args.device,
                                allow_sharding=args.allow_sharding,
                                need=args.need, k_domain=args.k_domain)
    print(f"{len(entries)} stage(s) trained: "
          f"{', '.join(entry['output_dir'] for entry in entries)}")
    return 0


def _build_artifact(args) -> int:
    print(build_artifact(args.model, args.corpus, args.out, max_samples=args.max_samples,
                         pca_variance=args.pca_variance, gmm_k=args.gmm_k,
                         layer_group_size=args.layer_group_size, quantize=args.quantize,
                         device=args.device, seed=args.seed))
    return 0


def _prepare_seed_corpus(args) -> int:
    print(prepare_seed_corpus(args.out, n_pretraining=args.n_pretraining,
                              n_instruction=args.n_instruction, cache_dir=args.cache_dir))
    return 0


def _prepare_domain(args) -> int:
    # No report of its own: `prepare_domain` already logs what it wrote and what it skipped.
    prepare_domain(args.inputs, args.out, min_length=args.min_length, combine=args.combine)
    return 0


# ==============================================================================================
# The parser
# ==============================================================================================

def _add_workspace(parser: argparse.ArgumentParser) -> None:
    """``--workspace``: which directory of LFA state to act on. Defaults to the one you are in."""
    parser.add_argument("--workspace", default=".", metavar="PATH",
                        help="the workspace directory (default: the current directory)")


def _add_device(parser: argparse.ArgumentParser, sharding: bool = False) -> None:
    parser.add_argument("--device", default=DEFAULT_DEVICE,
                        help=f"device to run on (default: {DEFAULT_DEVICE})")
    if sharding:
        parser.add_argument("--allow-sharding", action="store_true",
                            help="permit a device map that spreads the model over several "
                                 "devices; refused by default because it costs speed and only "
                                 "buys memory")


def build_parser() -> argparse.ArgumentParser:
    """The whole command line, subcommand by subcommand."""
    parser = argparse.ArgumentParser(
        prog="lfa",
        description="Layerwise Function Anchoring (LFA): adapt a model to a new domain while "
                    "preserving what its sub-modules compute on the hidden states it actually "
                    "sees.",
    )
    # The first thing anyone types into a bug report, and the version is the first thing anyone
    # reading that report asks for.
    parser.add_argument("--version", action="version",
                        version=f"lfa-anchoring {_lfa_version()}")
    subcommands = parser.add_subparsers(dest="command", required=True, metavar="<subcommand>")

    # ---------------------------------------------------------------------------------- init
    init = subcommands.add_parser(
        "init", help="create a workspace over a model and put its first p(h) artifact in place")
    init.add_argument("path", help="the workspace directory to create")
    init.add_argument("--model", required=True, metavar="ID",
                      help="a Hub id or a local checkpoint path")
    init.add_argument("--artifact", default="qwen3-0.6b-gmm1543k-int8", metavar="ID",
                      help="a published artifact id or a path to an artifact file "
                           "(default: %(default)s)")
    init.add_argument("--artifact-id", dest="artifact_id", metavar="ID",
                      help="the published artifact id a locally-passed artifact FILE is a copy "
                           "of, so the recipe's calibration is read against it rather than "
                           "against a path (ignored when --artifact is itself an id)")
    init.add_argument("--recipe", metavar="NAME",
                      help="the workspace's default recipe: a bundled name or a path (default: "
                           "the bundled recipe that names this model, if there is one)")
    init.set_defaults(handler=_init)

    # ------------------------------------------------------------------------- fetch-artifact
    fetch = subcommands.add_parser(
        "fetch-artifact", help="download a published p(h) artifact and verify its checksum")
    fetch.add_argument("artifact_id", metavar="ID", help="the artifact to fetch")
    fetch.add_argument("--dest", default=".", metavar="DIR",
                      help="where to write it (default: the current directory)")
    fetch.set_defaults(handler=_fetch_artifact)

    # -------------------------------------------------------------------------- list-artifacts
    listing = subcommands.add_parser("list-artifacts", help="show the published p(h) artifacts")
    listing.set_defaults(handler=_list_artifacts)

    # --------------------------------------------------------------------------------- train
    train = subcommands.add_parser("train", help="train one domain into the workspace's model")
    _add_workspace(train)
    train.add_argument("--corpus", required=True, metavar="DIR",
                       help="a file or directory of documents to adapt to")
    train.add_argument("--recipe", metavar="NAME|PATH",
                       help="a bundled recipe name or a path (default: the workspace's own)")
    train.add_argument("--epochs", type=int, metavar="N",
                       help="override the recipe's number of epochs (the learning-rate schedule "
                            "is laid over whatever this says)")
    train.add_argument("--full-weight", dest="full_weight", action="store_true", default=None,
                       help="train full weights instead of LoRA; outside the paper's validated "
                            "envelope")
    train.add_argument("--teacher-mode", dest="teacher_mode", default=None,
                       choices=list(TEACHER_MODES), metavar="MODE",
                       help="where the frozen teacher comes from: adapter_disabled reads it out "
                            "of the student's own LoRA base and loads no second model, separate "
                            "loads one, auto (the default) is the first under LoRA and the "
                            "second for --full-weight. The two are bit-identical; the choice is "
                            "memory, not results")
    train.add_argument("--resume", action="store_true",
                       help="continue the run already in this stage's output directory")
    _add_device(train, sharding=True)
    train.set_defaults(handler=_train)

    # -------------------------------------------------------------------------------- extend
    extend = subcommands.add_parser(
        "extend", help="fold the trained stage into the model and into p(h)")
    _add_workspace(extend)
    extend.add_argument("--need", type=int, default=40_000, metavar="N",
                        help="activations to collect per site (default: %(default)s)")
    extend.add_argument("--k-domain", type=int, default=8, metavar="K",
                        help="mixture components to fit per site (default: %(default)s)")
    _add_device(extend)
    extend.set_defaults(handler=_extend)

    # ------------------------------------------------------------------------------ evaluate
    evaluate = subcommands.add_parser(
        "evaluate", help="read the last stage on both axes: what it learned and what it kept")
    _add_workspace(evaluate)
    evaluate.add_argument("--corpus", metavar="DIR",
                          help="text to measure domain perplexity on, scored whole "
                               "(default: the stage's own held-out split -- the documents "
                               "its val_fraction kept out of training)")
    evaluate.add_argument("--compare-unanchored", action="store_true",
                          help="re-run the stage with lambda = mu = 0 and report it as a third "
                               "column: the control that says what the anchor bought")
    evaluate.add_argument("--n-windows", type=_windows, default=100, metavar="N",
                          help="WikiText-2 windows for the general axis; 0 scores the whole "
                               "split and 'none' skips it (default: %(default)s)")
    _add_device(evaluate)
    evaluate.set_defaults(handler=_evaluate)

    # ---------------------------------------------------------------------------------- fuse
    fuse = subcommands.add_parser(
        "fuse", help="export the current model with the last stage merged into it")
    _add_workspace(fuse)
    fuse.add_argument("--out", metavar="DIR",
                      help="where to write the merged checkpoint (default: "
                           "models/stage{N}_fused_export inside the workspace)")
    fuse.set_defaults(handler=_fuse)

    # --------------------------------------------------------------------------------- chain
    chain = subcommands.add_parser(
        "chain", help="run a whole sequence of domains from a YAML spec: train, extend, train...")
    chain.add_argument("spec", metavar="domains.yaml", help="the chain spec")
    _add_workspace(chain)
    # The same two knobs `lfa extend` takes, because a chain runs an extension between every
    # pair of domains and `--need` is the one to turn down when host memory is tight.
    chain.add_argument("--need", type=int, default=40_000, metavar="N",
                       help="activations to collect per site at each extension "
                            "(default: %(default)s)")
    chain.add_argument("--k-domain", type=int, default=8, metavar="K",
                       help="mixture components to fit per site at each extension "
                            "(default: %(default)s)")
    _add_device(chain, sharding=True)
    chain.set_defaults(handler=_chain)

    # ------------------------------------------------------------------------- build-artifact
    build = subcommands.add_parser(
        "build-artifact", help="collect and fit a p(h) artifact for a model over a seed corpus")
    build.add_argument("--model", required=True, metavar="ID",
                       help="a Hub id or a local checkpoint path")
    build.add_argument("--corpus", required=True, metavar="JSONL",
                       help="the seed corpus (see `lfa prepare-seed-corpus`)")
    build.add_argument("--out", required=True, metavar="PATH",
                       help="where to write distribution_stats.pt")
    build.add_argument("--max-samples", type=int, default=1_500_000, metavar="N",
                       help="hidden vectors to collect per site (default: %(default)s)")
    build.add_argument("--gmm-k", type=int, default=32, metavar="K",
                       help="mixture components per site (default: %(default)s)")
    build.add_argument("--pca-variance", type=float, default=0.95, metavar="V",
                       help="variance the stored basis must span (default: %(default)s)")
    build.add_argument("--layer-group-size", type=int, metavar="N",
                       help="collect this many layers at a time; host RAM is the binding "
                            "constraint, so this is normally set (7 for Qwen3-0.6B)")
    build.add_argument("--no-quantize", dest="quantize", action="store_false",
                       help="store the large fields in full precision instead of int8, which "
                            "doubles the file")
    build.add_argument("--seed", type=int, default=0, metavar="N",
                       help="base seed for the reservoir draws and the GMM fits "
                            "(default: %(default)s)")
    _add_device(build)
    build.set_defaults(handler=_build_artifact)

    # -------------------------------------------------------------------- prepare-seed-corpus
    seed = subcommands.add_parser(
        "prepare-seed-corpus",
        help="download and mix the 10:1 pretraining:instruction corpus p(h) is estimated over")
    seed.add_argument("--out", required=True, metavar="JSONL", help="the JSONL to write")
    seed.add_argument("--n-pretraining", type=int, default=12_000, metavar="N",
                      help="pretraining documents to aim for (default: %(default)s)")
    seed.add_argument("--n-instruction", type=int, default=20_000, metavar="N",
                      help="instruction pairs to aim for (default: %(default)s)")
    seed.add_argument("--cache-dir", metavar="DIR",
                      help="a non-default Hugging Face cache directory")
    seed.set_defaults(handler=_prepare_seed_corpus)

    # ------------------------------------------------------------------------- prepare-domain
    domain = subcommands.add_parser(
        "prepare-domain", help="extract and clean domain documents from text, HTML and PDF")
    domain.add_argument("inputs", nargs="+", metavar="INPUTS",
                        help="files and/or directories to read")
    domain.add_argument("--out", required=True, metavar="DIR", help="directory to write into")
    domain.add_argument("--min-length", type=int, default=1000, metavar="N",
                        help="drop a document shorter than this many characters after cleaning "
                             "(default: %(default)s)")
    domain.add_argument("--combine", action="store_true",
                        help="write one combined file instead of one file per input. The loader "
                             "reads a file as one document, so a combined corpus is a single "
                             "document contributing every chunk -- which training will warn "
                             "about. Use it to inspect the cleaned text, not to train on")
    domain.set_defaults(handler=_prepare_domain)

    return parser


# ==============================================================================================
# Entry point
# ==============================================================================================

def main(argv: list[str] | None = None) -> int:
    """Parse ``argv`` (``sys.argv[1:]`` when ``None``), run the subcommand, return the status.

    Returns:
        ``0``; ``2`` for anything the library refused -- printed as a single line on stderr,
        because those messages are written to be read rather than traced; ``130`` for Ctrl-C,
        the shell's convention for a command killed by SIGINT.
    """
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    try:
        return args.handler(args)
    except USER_FACING_ERRORS as error:
        # Collapsed to one line: the library's messages are sentences, and a wrapped traceback
        # would put the instruction they end with out of sight.
        print(f"lfa: {' '.join(str(error).split())}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        # Ctrl-C during training is the most ordinary way anyone stops a run, and a traceback
        # ending in `KeyboardInterrupt` reads like a crash -- which the documented rule would
        # then call a bug in this package. Say what happened and how to pick it up instead.
        print("\nlfa: interrupted. A run that reached a checkpoint can be continued with the "
              "same command plus --resume; one interrupted before its first checkpoint has "
              "nothing saved and can simply be started again.", file=sys.stderr)
        return 130


if __name__ == "__main__":                                          # pragma: no cover
    raise SystemExit(main())
