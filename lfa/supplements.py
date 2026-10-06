"""The question-and-answer supplement's cache: where a supplement is looked for, and when one
found is the right one.

A supplement is the entry model's own question-and-answer pairs over the training side of a
corpus; its measured effect is on whether the domain's knowledge can be reached when the model is
asked about it, not on protecting skills. It is keyed on the training side's hash, the writer
checkpoint's hash, the template's hash and the domain description, so it is reused only when all
four match. A supplement prepared with the data (``lfa prepare-domain --supplement --model ...``)
sits beside the corpus; one written by ``train`` sits in the workspace; ``train`` looks in both.
"""

from __future__ import annotations

import json
import logging
import os
import re
from pathlib import Path

from .corpus import min_documents_for_held_out, split_documents
from .prepare_domain import SUGGESTED_SPLIT_CHARS
from .recipe import Recipe
from .selfgen.generate import checkpoint_sha256, sha256_text
from .selfgen.supplement import SupplementOptions, template_sha256, write_supplement

logger = logging.getLogger(__name__)

__all__ = ["domain_description_for", "beside_corpus", "recipe_for", "supplement_for",
           "prepare_supplement"]


# Container names that say where text sits, not what it is about (compared after `_`/`-` -> space).
_GENERIC_NAMES = frozenset({
    "train", "training", "test", "val", "valid", "validation", "heldout", "held out", "dev",
    "data", "dataset", "datasets", "corpus", "corpora", "text", "texts", "docs", "documents",
    "raw", "clean", "cleaned", "input", "inputs", "src"})


def _readable(name: str) -> str:
    return re.sub(r"[_\-]+", " ", name).strip()


def domain_description_for(corpus_path: Path) -> str:
    """What the template says the text is on: the corpus's name with `_`/`-` as spaces.

    The name is the directory's, or the file's stem for a file. A generic container name
    (``train``, ``data``, ``corpus`` ...) says nothing about the domain, so the nearest ancestor
    directory whose name is not generic is used instead (``data/darwin/train`` -> ``darwin``):
    first along the path as given, then, when a relative path runs out, on up from the working
    directory, stopping below the home directory (whose name is the user's, not a domain's).
    ``..`` is folded; the callers resolve symlinks before they get here. ``"the domain"`` when no
    name on the way is anything else.
    """
    corpus_path = Path(corpus_path)
    full = Path(os.path.abspath(corpus_path))   # the given path is its tail; `..` folded
    home = Path.home().resolve()                # callers pass resolved corpus paths
    stop = {home, *home.parents}
    for i, step in enumerate([full, *full.parents]):
        if step in stop:
            break
        readable = _readable(step.stem if i == 0 and corpus_path.is_file() else step.name)
        if readable and readable.lower() not in _GENERIC_NAMES:
            return readable
    return "the domain"


def beside_corpus(corpus_path: Path) -> Path:
    """Where a supplement prepared with the data lives: ``<corpus>.supplement/`` next to it.

    Outside the corpus directory on purpose -- the loader reads every file under the corpus as a
    document, so pairs written inside it would be trained on as domain text.
    """
    corpus_path = Path(corpus_path)
    return corpus_path.with_name(corpus_path.name + ".supplement")


def recipe_for(model_id: str, recipe: Recipe | str | Path | None = None) -> Recipe:
    """The recipe whose training side a supplement for ``model_id`` is written from.

    ``recipe`` as given (a :class:`~lfa.recipe.Recipe`, a bundled name or a YAML path), else the
    bundled recipe whose own ``model_id`` is this one. Resolved before anything is written, so a
    command that also prepares the corpus refuses whole rather than halfway.

    Raises:
        ValueError: no ``recipe`` was given and no bundled recipe names ``model_id``.
    """
    if recipe is None:
        recipe = Recipe.bundled_for(model_id)
        if recipe is None:
            raise ValueError(
                f"No bundled recipe names {model_id!r}, and the supplement is written from the "
                "recipe's training side (its held-out fraction and seed). Pass --recipe <name or "
                "YAML path>.")
    return recipe if isinstance(recipe, Recipe) else Recipe.load(recipe)


def supplement_for(corpus_path: Path, writer_id: str, recipe: Recipe, *, write_root: Path,
                   search_roots: list[Path], domain_description: str | None = None,
                   device: str = "cuda:0", force: bool = False) -> tuple[Path, dict]:
    """The supplement ``writer_id`` writes for ``corpus_path``: reused when found, else written.

    Each of ``search_roots`` is looked in, in order, for ``<corpus hash[:12]>/supplement.jsonl``
    whose manifest matches the training side's hash, the writer checkpoint's hash, the template's
    hash and the domain description (explicit, or derived from the corpus path) -- so a different
    writer or a different ``domain_description`` writes a new one rather than being silently
    ignored. When none matches (or ``force``), the writer writes into ``write_root``. The training
    side is the recipe's: its held-out fraction and seed decide which documents the pairs may
    come from, so held-out text never reaches the supplement.

    Every route to the writer comes through here -- ``lfa train``, ``lfa prepare-supplement``,
    ``lfa prepare-domain --supplement`` and their Python calls -- so this is where a corpus with
    nothing to hold out is refused: before any supplement is looked for or the writer loads.

    Returns:
        The supplement's path and its manifest.

    Raises:
        ValueError: the recipe holds documents out (``val_fraction`` above 0) and the corpus has
            too few to hold any out -- one long file, typically. The pairs are minutes of
            generation, for a run with no held-out curve to choose its epochs by and a domain
            number that is a fit.
    """
    corpus_path = Path(corpus_path).expanduser().resolve()
    train_docs, held_out = split_documents(corpus_path, recipe.val_fraction, recipe.seed)
    if recipe.val_fraction > 0 and not held_out:
        raise ValueError(_nothing_held_out(corpus_path, len(train_docs), recipe.val_fraction))
    corpus_hash = sha256_text(train_docs)
    writer_hash = checkpoint_sha256(str(writer_id))
    description = domain_description or domain_description_for(corpus_path)
    if not force:
        for root in search_roots:
            out = Path(root) / corpus_hash[:12] / "supplement.jsonl"
            manifest_path = Path(str(out) + ".manifest.json")
            if out.is_file() and manifest_path.is_file():
                manifest = json.loads(manifest_path.read_text())
                if (manifest.get("corpus_sha256") == corpus_hash
                        and manifest.get("writer_sha256") == writer_hash
                        and manifest.get("template_sha256") == template_sha256()
                        and manifest.get("domain_description") == description):
                    logger.info("Supplement reused: %s", out)
                    return out, manifest
    out = Path(write_root) / corpus_hash[:12] / "supplement.jsonl"
    logger.info("Writing the supplement for %s with %s (%d training documents)", corpus_path,
                writer_id, len(train_docs))
    manifest = write_supplement(str(writer_id), train_docs, out, domain_description=description,
                                options=SupplementOptions(), corpus_sha256=corpus_hash,
                                device=device)
    return out, manifest


def _nothing_held_out(corpus_path: Path, n_documents: int, val_fraction: float) -> str:
    """The refusal for a supplement over a corpus that holds nothing out.

    The same facts and the same fix as ``lfa prepare-domain``'s refusal, worded for a corpus that
    already exists.
    """
    return (
        f"{corpus_path} holds {n_documents} document(s), and training holds out whole documents "
        f"(val_fraction {val_fraction:g}), which takes at least "
        f"{min_documents_for_held_out(val_fraction)}. With fewer, nothing is held out: there is "
        "no per-epoch held-out curve to choose the number of epochs by, and the domain number "
        "is a fit rather than a measurement -- so the supplement, minutes of generation, is not "
        "written. Fix: a long file (a book, a report) is one document until it is split: `lfa "
        f"prepare-domain <file> --out <new dir> --split-chars {SUGGESTED_SPLIT_CHARS}` "
        f"(split_chars={SUGGESTED_SPLIT_CHARS} from Python) cuts every file into documents of "
        f"about {SUGGESTED_SPLIT_CHARS:,} characters at paragraph boundaries; or add more files. "
        "To train on it as it is, pass --no-supplement (supplement=False from Python).")


def prepare_supplement(corpus, model_id: str, *, recipe: Recipe | str | Path | None = None,
                       domain_description: str | None = None, device: str = "cuda:0",
                       force: bool = False) -> Path:
    """Have ``model_id`` write (or reuse) the supplement for ``corpus``, with no workspace.

    The file lands beside the corpus (:func:`beside_corpus`), where ``train`` finds it when the
    workspace's current model is this same checkpoint -- a chain's later stage, whose entry model
    is a fused one, writes its own. ``recipe`` supplies the training side (held-out fraction and
    seed); left out, it is the bundled recipe for ``model_id`` (:func:`recipe_for`).

    Raises:
        ValueError: no ``recipe`` was given and no bundled recipe names ``model_id``.
    """
    corpus_path = Path(corpus).expanduser().resolve()
    recipe = recipe_for(model_id, recipe)
    root = beside_corpus(corpus_path)
    path, _ = supplement_for(corpus_path, model_id, recipe, write_root=root, search_roots=[root],
                             domain_description=domain_description, device=device, force=force)
    return path
