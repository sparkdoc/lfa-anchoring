"""Domain documents: PDF/HTML/Markdown/plain text on disk to one cleaned ``.txt`` per input.

This is the front door of a Layerwise Function Anchoring (LFA) run: whatever the new domain
arrives as, :func:`prepare_domain` turns it into the flat directory of ``.txt`` files that
:func:`lfa.corpus.load_texts` reads.

Extraction is per format -- ``.txt``/``.md`` pass straight through, ``.html``/``.htm`` go through
BeautifulSoup + markdownify, ``.pdf`` through marker -- and every document then goes through the
same :func:`clean_text` pass, which removes the markup artifacts both extractors leave behind
(footnote wrappers, link targets, image references) and unwraps mid-paragraph line breaks so a
paragraph is one line. A document shorter than ``min_length`` *characters after cleaning* is
dropped: a page that extracted to a nav bar and a cookie notice is not training data.

HTML and PDF support are optional extras (``pip install 'lfa-anchoring[html]'`` /
``'lfa-anchoring[pdf]'``); a missing one raises :class:`MissingExtra` naming the extra rather than
silently skipping the files, so a corpus is never quietly half-prepared.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Iterable
from pathlib import Path

logger = logging.getLogger(__name__)


class MissingExtra(ImportError):
    """Raised when a document needs an optional extra that is not installed.

    An ``ImportError`` subclass, so a caller that catches ``ImportError`` still catches it -- but
    a *named* one, so :data:`lfa.cli.USER_FACING_ERRORS` can collapse it to the single line it
    already is without also swallowing every other import failure in the process, which is what
    listing bare ``ImportError`` there would do. The message names the file and the extra to
    install; nothing about it is a defect in this package, so a traceback would only bury it.
    """

TEXT_EXTENSIONS = {".txt", ".md"}
HTML_EXTENSIONS = {".html", ".htm"}
PDF_EXTENSIONS = {".pdf"}
SUPPORTED_EXTENSIONS = TEXT_EXTENSIONS | HTML_EXTENSIONS | PDF_EXTENSIONS

COMBINED_NAME = "combined_domain_data.txt"
_SEPARATOR = "\n\n" + "=" * 80 + "\n\n"


# ----------------------------------------------------------------------------------------------
# Per-format extraction
# ----------------------------------------------------------------------------------------------

def extract_html(path: Path) -> str:
    """The readable content of an HTML file, as markdown.

    ``script``/``style``/``nav``/``footer``/``header``/``aside``/``head`` are dropped before the
    conversion, and the ``body`` is preferred over the whole document so no DOCTYPE artifact
    survives.

    Raises:
        MissingExtra: if the ``[html]`` extra is not installed.
    """
    try:
        import markdownify
        from bs4 import BeautifulSoup
    except ImportError as exc:
        raise MissingExtra(
            f"Reading {path.name} needs BeautifulSoup and markdownify: "
            "pip install 'lfa-anchoring[html]'"
        ) from exc

    soup = BeautifulSoup(path.read_text(encoding="utf-8", errors="replace"), "html.parser")
    for tag in soup.find_all(["script", "style", "nav", "footer", "header", "aside", "head"]):
        tag.decompose()
    content = soup.body or soup
    return markdownify.markdownify(str(content), heading_style="ATX", strip=["img"])


def make_pdf_converter():
    """A marker ``PdfConverter``, with its models loaded (seconds, and several GB of VRAM).

    Build it once and pass it to every :func:`extract_pdf` call, as :func:`prepare_domain` does.

    Raises:
        MissingExtra: if the ``[pdf]`` extra is not installed.
    """
    try:
        from marker.config.parser import ConfigParser
        from marker.converters.pdf import PdfConverter
        from marker.models import create_model_dict
    except ImportError as exc:
        raise MissingExtra(
            "Reading PDFs needs marker: pip install 'lfa-anchoring[pdf]'"
        ) from exc

    parser = ConfigParser({"output_format": "markdown",
                           "disable_image_extraction": True,
                           "page_separator": "\n\n"})
    return PdfConverter(config=parser.generate_config_dict(),
                        artifact_dict=create_model_dict(),
                        processor_list=parser.get_processors(),
                        renderer=parser.get_renderer())


def extract_pdf(path: Path, converter) -> str:
    """The text of a PDF, as markdown, using a converter from :func:`make_pdf_converter`."""
    from marker.output import text_from_rendered                        # pragma: no cover - env

    text, _, _ = text_from_rendered(converter(str(path)))
    return text


def extract(path: Path, pdf_converter=None) -> str:
    """Dispatch one file to the extractor for its extension."""
    suffix = path.suffix.lower()
    if suffix in TEXT_EXTENSIONS:
        return path.read_text(encoding="utf-8", errors="replace")
    if suffix in HTML_EXTENSIONS:
        return extract_html(path)
    if suffix in PDF_EXTENSIONS:
        return extract_pdf(path, pdf_converter if pdf_converter is not None
                           else make_pdf_converter())
    raise ValueError(f"Unsupported file type: {path}")                  # pragma: no cover - guarded


# ----------------------------------------------------------------------------------------------
# Cleaning
# ----------------------------------------------------------------------------------------------

def _is_structural(line: str) -> bool:
    """Whether a line is markdown structure (heading, list, table, quote) rather than prose."""
    return (
        not line
        or line.startswith("#")
        or line.startswith("- ")
        or line.startswith("* ")
        or line.startswith("|")
        or line.startswith("> ")
        or bool(re.match(r"\d+\.\s", line))
    )


def clean_text(text: str) -> str:
    """Strip extraction artifacts and unwrap paragraphs.

    Removes residual HTML tags, footnote wrappers, editorial brackets, link targets (keeping the
    link text), image references and horizontal rules; joins consecutive prose lines into one
    paragraph line (markdownify preserves the source HTML's own wrapping, which would otherwise
    train the model on line breaks that mean nothing); and collapses runs of blank lines and
    spaces.
    """
    # Residual HTML from marker (sup/sub footnotes, span wrappers).
    text = re.sub(r"<sup>(.*?)</sup>", r"\1", text)
    text = re.sub(r"<sub>(.*?)</sub>", r"\1", text)
    text = re.sub(r"</?span[^>]*>", "", text)

    # Footnote wrappers \*[[ ... ]] and editorial brackets [[ ... ]]; both before link stripping,
    # since either can contain a markdown link.
    text = re.sub(r"\\\*\[{2,3}(.*?)\]{2,3}", r"\1", text, flags=re.DOTALL)
    text = re.sub(r"\[\[(.*?)\]\]", r"\1", text, flags=re.DOTALL)
    text = re.sub(r"\[\\\*\]", "", text)

    # Images before links: ![alt](path) would otherwise leave a stray "!".
    text = re.sub(r"!\[[^\]]*\]\([^)]*\)", "", text)
    text = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", text)

    # Unescape runs of \_ (fill-in-the-blank lines), leaving single \* notation alone.
    text = re.sub(r"(\\_){2,}", lambda m: "_" * (len(m.group(0)) // 2), text)

    # Bold-only lines are boilerplate (author affiliations, emails).
    text = re.sub(r"^\*\*[^*]+\*\*$", "", text, flags=re.MULTILINE)
    text = re.sub(r"\n-{3,}\n", "\n\n", text)

    lines: list[str] = []
    for raw_line in text.split("\n"):
        line = raw_line.strip()
        if lines and not _is_structural(lines[-1]) and not _is_structural(line):
            lines[-1] += " " + line
        else:
            lines.append(line)
    text = "\n".join(lines)

    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r" {2,}", " ", text)
    return text.strip()


# ----------------------------------------------------------------------------------------------
# Preparation
# ----------------------------------------------------------------------------------------------

def find_input_files(inputs: str | Path | Iterable[str | Path],
                     recursive: bool = True) -> list[Path]:
    """Every supported file reachable from ``inputs``, sorted and de-duplicated.

    An input that is a file is taken if its extension is supported; an input that is a directory
    is searched (recursively unless ``recursive=False``).
    """
    if isinstance(inputs, (str, Path)):
        inputs = [inputs]

    found: set[Path] = set()
    for item in inputs:
        path = Path(item)
        if path.is_file():
            if path.suffix.lower() in SUPPORTED_EXTENSIONS:
                found.add(path)
        elif path.is_dir():
            glob = path.rglob if recursive else path.glob
            for extension in SUPPORTED_EXTENSIONS:
                found.update(p for p in glob(f"*{extension}") if p.is_file())
        else:
            raise FileNotFoundError(f"Input path not found: {path}")
    return sorted(found)


def _unique_output(out_dir: Path, stem: str) -> Path:
    """``out_dir/<stem>.txt``, suffixed ``_1``, ``_2``, ... if that name is taken."""
    path, counter = out_dir / f"{stem}.txt", 1
    while path.exists():
        path = out_dir / f"{stem}_{counter}.txt"
        counter += 1
    return path


def prepare_domain(
    inputs: str | Path | Iterable[str | Path],
    out_dir: str | Path,
    *,
    min_length: int = 1000,
    combine: bool = False,
    recursive: bool = True,
) -> list[Path]:
    """Extract, clean and write the domain documents; return the files written, in order.

    Args:
        inputs: files and/or directories to read. Directories are searched for
            ``.txt``/``.md``/``.html``/``.htm``/``.pdf``; anything else is ignored.
        out_dir: directory to write into; created if absent. Existing files are never
            overwritten -- a name collision gets a ``_1``, ``_2``, ... suffix.
        min_length: drop a document shorter than this many characters *after cleaning*.
        combine: write one ``combined_domain_data.txt`` (documents separated by a rule and headed
            by ``# Source: <filename>``) instead of one file per input. It is suffixed like any
            other output when that name is taken, so a second ``--combine`` run over different
            inputs writes ``combined_domain_data_1.txt`` rather than replacing the first corpus.
        recursive: search directories' subdirectories.

    Raises:
        ValueError: if no supported file is found under ``inputs``.
        MissingExtra: if a found file needs the ``[html]`` or ``[pdf]`` extra and it is missing.
    """
    out_dir = Path(out_dir)
    files = find_input_files(inputs, recursive=recursive)
    if not files:
        raise ValueError(
            f"No supported files ({', '.join(sorted(SUPPORTED_EXTENSIONS))}) found in {inputs}")

    # Load the marker models once, and only if a PDF is actually in the batch.
    pdf_converter = None
    if any(p.suffix.lower() in PDF_EXTENSIONS for p in files):
        logger.info("Loading marker models for %d PDF(s)...",
                    sum(1 for p in files if p.suffix.lower() in PDF_EXTENSIONS))
        pdf_converter = make_pdf_converter()

    out_dir.mkdir(parents=True, exist_ok=True)

    written: list[Path] = []
    documents: list[str] = []
    n_skipped, n_characters = 0, 0

    for file_path in files:
        text = clean_text(extract(file_path, pdf_converter))
        if len(text) < min_length:
            logger.info("Skipping %s: %d characters after cleaning (min_length=%d)",
                        file_path.name, len(text), min_length)
            n_skipped += 1
            continue

        n_characters += len(text)
        if combine:
            documents.append(f"# Source: {file_path.name}\n\n{text}")
        else:
            out_path = _unique_output(out_dir, file_path.stem)
            out_path.write_text(text, encoding="utf-8")
            written.append(out_path)

    if combine and documents:
        # Through `_unique_output` like every per-file write: under `combine` this one file IS
        # the corpus, so an overwrite costs the whole of a previous preparation rather than one
        # document of it -- and the docstring above promises it does not happen.
        out_path = _unique_output(out_dir, Path(COMBINED_NAME).stem)
        out_path.write_text(_SEPARATOR.join(documents), encoding="utf-8")
        written.append(out_path)

    logger.info("Prepared %d document(s) into %s (%d skipped, %d characters, ~%d tokens)",
                len(files) - n_skipped, out_dir, n_skipped, n_characters, n_characters // 4)
    return written
