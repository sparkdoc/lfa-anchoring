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
import re
from pathlib import Path

from .corpus import split_documents
from .recipe import Recipe
from .selfgen.generate import checkpoint_sha256, sha256_text
from .selfgen.supplement import SupplementOptions, template_sha256, write_supplement

logger = logging.getLogger(__name__)

__all__ = ["domain_description_for", "beside_corpus", "recipe_for", "supplement_for",
           "prepare_supplement"]


def domain_description_for(corpus_path: Path) -> str:
    """The corpus directory's name with `_`/`-` as spaces; what the template says the text is on."""
    name = corpus_path.stem if corpus_path.is_file() else corpus_path.name
    return re.sub(r"[_\-]+", " ", name).strip() or "the domain"


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

    Returns:
        The supplement's path and its manifest.
    """
    corpus_path = Path(corpus_path).expanduser().resolve()
    train_docs, _ = split_documents(corpus_path, recipe.val_fraction, recipe.seed)
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
