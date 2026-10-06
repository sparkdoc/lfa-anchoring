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

A file is one document unless ``split_chars`` is given, and the trainer holds out whole
documents: one downloaded book prepared as it stands is a corpus nothing can be held out of. With
``split_chars`` every cleaned file is cut into documents of about that many characters at
paragraph boundaries (:func:`split_into_documents`). A corpus that would still hold too few
documents for a held-out split is warned about -- or, under ``require_held_out``, refused before
anything is written -- with that fix named.

HTML and PDF support are optional extras (``pip install -c constraints-tested.txt -e '.[html]'``
/ ``'.[pdf]'`` in the lfa-anchoring checkout); a missing one raises :class:`MissingExtra` naming
the extra rather than silently skipping the files, so a corpus is never quietly half-prepared.
"""

from __future__ import annotations

import hashlib
import logging
import re
from collections.abc import Iterable
from pathlib import Path

from .corpus import MIN_CHUNK_TOKENS, load_texts, min_documents_for_held_out

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

#: The document size the messages and docs suggest for ``split_chars``: the two-domain
#: walkthrough's, about 850 tokens, so most documents are one or two 512-token chunks.
SUGGESTED_SPLIT_CHARS = 3500

#: The smallest ``split_chars`` accepted. A split document can be as short as about half the
#: size (a file's remainder under half joins the document before it; one at half or more stands
#: alone), and the trainer's chunker drops a whole document under its
#: :data:`~lfa.corpus.MIN_CHUNK_TOKENS`-token minimum without a word. 250 characters is 10 tokens
#: only at 25 characters a token, several times what any prose tokenizes to, so at this floor no
#: split document is that short; well below it a split can lose whole documents silently.
MIN_SPLIT_CHARS = 500

#: The held-out share a prepared corpus is checked against unless told otherwise: the bundled
#: recipes' ``val_fraction``.
DEFAULT_VAL_FRACTION = 0.1
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
            "pip install -c constraints-tested.txt -e '.[html]' in your lfa-anchoring checkout"
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
            "Reading PDFs needs marker: pip install -c constraints-tested.txt -e '.[pdf]' in your "
            "lfa-anchoring checkout"
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
# Splitting a long file into documents
# ----------------------------------------------------------------------------------------------

#: A sentence end: terminal punctuation, any closing quotes or brackets, then whitespace. The cut
#: falls after the punctuation and its closers; the whitespace goes with neither side.
_SENTENCE_END = re.compile(r"[.!?\u2026]+[\"'\u201d\u2019)\]]*(?=\s)")
_WHITESPACE = re.compile(r"\s+")


def _cut_paragraph(paragraph: str, target_chars: int) -> list[str]:
    """Cut one over-long paragraph into pieces of at least ``target_chars``, the last excepted.

    Each cut is the first sentence end at or past ``target_chars`` that leaves the piece no longer
    than twice it; failing that, the first whitespace past ``target_chars``; failing that (a run
    with no whitespace at all), ``target_chars`` itself. A paragraph is only cut while what is left
    of it is longer than twice ``target_chars``, so the last piece is at most that.
    """
    pieces: list[str] = []
    start = 0
    while len(paragraph) - start > 2 * target_chars:
        lo, hi = start + target_chars, start + 2 * target_chars
        cut = next((m.end() for m in _SENTENCE_END.finditer(paragraph, lo - 1, hi)
                    if lo <= m.end() <= hi), None)
        if cut is None:
            space = _WHITESPACE.search(paragraph, lo, hi)
            cut = space.start() if space is not None else lo
        pieces.append(paragraph[start:cut].strip())
        start = cut
    pieces.append(paragraph[start:].strip())
    return [piece for piece in pieces if piece]


def split_into_documents(text: str, target_chars: int = SUGGESTED_SPLIT_CHARS) -> list[str]:
    """Cut one cleaned text into documents of about ``target_chars`` characters.

    The walkthrough notebook's ``to_documents``, made to keep every character: blank-line
    separated paragraphs are accumulated, and a document is closed as soon as it reaches
    ``target_chars``, so documents are cut at paragraph boundaries and run from ``target_chars``
    to one paragraph more. A paragraph longer than twice ``target_chars`` is the one exception: it
    is cut inside, at a sentence end where there is one in reach and at whitespace where there is
    not (:func:`_cut_paragraph`), and each piece of it but the last closes its document. Where the
    notebook drops a short remainder, here a remainder under half of ``target_chars`` joins the
    document before it, so no text is lost; a text that never reaches ``target_chars`` is one
    document. Deterministic: the same text and size always give the same documents.

    Raises:
        ValueError: ``target_chars`` below 1.
    """
    if target_chars < 1:
        raise ValueError(f"split size must be a positive number of characters, got {target_chars}")

    documents: list[str] = []
    buffer: list[str] = []
    size = 0
    #: Whether `buffer` opens with the continuation of a paragraph cut at the previous document's
    #: end, which must rejoin that document with a space rather than a paragraph break.
    continues = False

    for paragraph in (p.strip() for p in re.split(r"\n\s*\n", text)):
        if not paragraph:
            continue
        pieces = (_cut_paragraph(paragraph, target_chars)
                  if len(paragraph) > 2 * target_chars else [paragraph])
        for index, piece in enumerate(pieces):
            if not buffer:
                continues = index > 0
            buffer.append(piece)
            size += len(piece) + 2
            if size >= target_chars or index < len(pieces) - 1:
                documents.append("\n\n".join(buffer))
                buffer, size = [], 0

    if buffer:
        remainder = "\n\n".join(buffer)
        if documents and len(remainder) < target_chars / 2:
            documents[-1] += (" " if continues else "\n\n") + remainder
        else:
            documents.append(remainder)
    return documents


def held_out_shortfall(n_documents: int, val_fraction: float = DEFAULT_VAL_FRACTION) -> int:
    """How many documents short of a held-out split a corpus of ``n_documents`` is; ``0`` if none.

    The minimum is the trainer's own (:func:`lfa.corpus.min_documents_for_held_out`): 2 at the
    bundled recipes' ``val_fraction`` of 0.1.
    """
    return max(0, min_documents_for_held_out(val_fraction) - n_documents)


def _too_few_documents(out_dir: Path, n_documents: int, val_fraction: float, *, combine: bool,
                       written: bool, at_most: bool = False) -> str:
    """The sentence a corpus too small for a held-out split is warned or refused with.

    ``at_most`` when the count is an upper bound taken before any file was read (every input as
    one document, none of them dropped for ``min_length``).
    """
    needed = min_documents_for_held_out(val_fraction)
    if combine:
        fix = ("--combine writes one file, which the loader reads as one document: prepare "
               f"without it, with --split-chars {SUGGESTED_SPLIT_CHARS} for a long file")
    else:
        fix = (f"a long file (a book, a report) is one document until it is split: "
               f"--split-chars {SUGGESTED_SPLIT_CHARS} (split_chars={SUGGESTED_SPLIT_CHARS} from "
               f"Python) cuts every file into documents of about "
               f"{SUGGESTED_SPLIT_CHARS:,} characters at paragraph boundaries; or add more files")
    if written:
        fix += (". Prepare into a fresh --out directory: a second run into this one adds its "
                "documents beside these")
    holds = "holds" if written else ("would hold at most" if at_most else "would hold")
    return (f"{out_dir} {holds} {n_documents} document(s), and "
            f"training holds out whole documents (val_fraction {val_fraction:g}), which takes "
            f"at least {needed}. With fewer, nothing is held out: there is no per-epoch held-out "
            f"curve to choose the number of epochs by, and the domain number is a fit rather "
            f"than a measurement. Fix: {fix}.")


#: Sentences shorter than this many characters (after whitespace is normalised) are not compared:
#: headings, chapter numbers, short lines and stock phrases recur across unrelated texts, and
#: counting them would make any two books look alike. The unit is the sentence rather than the
#: paragraph for two measured reasons (review of 2026-10-05): a text of short paragraphs (an FAQ
#: corpus, a play, verse) has few or no paragraphs long enough to compare, and `--split-chars`
#: cuts a long paragraph -- at its sentence ends where it can -- so an earlier preparation split
#: at 500 characters no longer holds the paragraphs a re-run has, while it does hold the sentences.
OVERLAP_MIN_SENTENCE_CHARS = 60

#: The share of an input's sentence text -- by characters, over its sentences of at least
#: :data:`OVERLAP_MIN_SENTENCE_CHARS` -- already in ``out_dir`` or in an earlier input of the same
#: call, at or above which the input is refused as the same text prepared again. Measured on the
#: user trial's Gutenberg books and two synthetic corpora (review of 2026-10-05): Darwin re-run
#: against its own earlier 500-character split 0.986; a re-run with the Gutenberg header and
#: licence stripped 1.0; a one-file play re-split, and an FAQ corpus prepared twice, 1.0; different
#: books sharing only the Project Gutenberg licence 0.041 and 0.02. The two kinds are some 25 times
#: apart, so the threshold is not delicate. Above 0 and below it the input is prepared, with a
#: warning naming the shared text's files: that is typically a licence or front matter.
REPREPARATION_SHARE = 0.5


def _sentences(text: str) -> list[str]:
    """The text's sentences of at least :data:`OVERLAP_MIN_SENTENCE_CHARS`, whitespace normalised.

    A sentence ends where :func:`split_into_documents` would cut one (:data:`_SENTENCE_END`) or at
    a paragraph break, so a sentence of a text is a sentence of every split of it: the splitter
    cuts only at paragraph breaks and sentence ends, bar a sentence longer than the split size.
    """
    sentences: list[str] = []
    for block in re.split(r"\n\s*\n", text):
        normalised = " ".join(block.split())
        start = 0
        for match in _SENTENCE_END.finditer(normalised):
            sentences.append(normalised[start:match.end()].strip())
            start = match.end()
        sentences.append(normalised[start:].strip())
    return [sentence for sentence in sentences if len(sentence) >= OVERLAP_MIN_SENTENCE_CHARS]


def _sentence_hash(sentence: str) -> str:
    return hashlib.sha256(sentence.encode("utf-8")).hexdigest()


def _sentences_already_in(out_dir: Path) -> dict[str, set[tuple[bool, str]]]:
    """Every long sentence of the ``.txt``/``.md`` documents under ``out_dir``, by hash, mapped to
    the files holding it as ``(False, path)`` -- the flag marks an input of the current call."""
    index: dict[str, set[tuple[bool, str]]] = {}
    if out_dir.is_dir():
        for path in sorted(p for p in out_dir.rglob("*") if p.is_file()):
            if path.suffix.lower() in TEXT_EXTENSIONS:
                text = path.read_text(encoding="utf-8", errors="replace")
                for sentence in _sentences(text):
                    index.setdefault(_sentence_hash(sentence), set()).add((False, str(path)))
    return index


def _percent(share: float) -> str:
    """``"100%"``, ``"37%"``, or ``"1.9%"`` -- a decimal only where a whole percent would read 0."""
    return f"{share:.0%}" if share >= 0.1 else f"{share:.1%}"


def _sources_phrase(by_source: dict[tuple[bool, str], int], total: int) -> str:
    """The three sources holding most of the shared text, each with its share, and how many more."""
    ranked = sorted(by_source.items(), key=lambda item: (-item[1], item[0][1]))
    named = [f"{'the earlier input ' if is_input else ''}{label} ({_percent(chars / total)})"
             for (is_input, label), chars in ranked[:3]]
    more = f", and {len(ranked) - 3} more file(s)" if len(ranked) > 3 else ""
    return ", ".join(named) + more


def _check_text_already_there(out_dir: Path, whole_texts: list[tuple[str, str]]) -> list[str]:
    """Refuse an input whose text is mostly in ``out_dir`` or in an earlier input; warn on less.

    The same text twice in one corpus is invisible to every count, and the trainer's split then
    holds out a document whose text it trains on. The case this exists for is a one-file corpus
    prepared unsplit, warned about, and prepared again into the same directory with
    ``--split-chars`` -- perhaps after the boilerplate was stripped as the docs advise -- which
    leaves the whole book on the training side beside every one of its held-out pieces. Whole-text
    hashes miss any edit, so the comparison is by sentence: the share of an input's sentence text
    (sentences of :data:`OVERLAP_MIN_SENTENCE_CHARS` or more) already present. At
    :data:`REPREPARATION_SHARE` or more it is refused; above 0 it is warned about and returned.

    Args:
        whole_texts: ``(input path, cleaned text)`` for each input, in the order prepared.

    Returns:
        The warnings, one per input that shares some text.

    Raises:
        ValueError: naming the input, the share, and the files that hold most of it.
    """
    index = _sentences_already_in(out_dir)
    notes: list[str] = []
    for source, text in whole_texts:
        sentences = _sentences(text)
        total = sum(len(sentence) for sentence in sentences)
        shared = 0
        by_source: dict[tuple[bool, str], int] = {}
        for sentence in sentences:
            holders = index.get(_sentence_hash(sentence))
            if holders:
                shared += len(sentence)
                for holder in holders:
                    by_source[holder] = by_source.get(holder, 0) + len(sentence)
        share = shared / total if total else 0.0

        if share >= REPREPARATION_SHARE:
            in_call = all(is_input for is_input, _ in by_source)
            fix = ("Give each text once." if in_call else
                   f"If this is the same text prepared again, edited or not, prepare into a fresh "
                   f"--out directory, or delete the earlier preparation of this text from "
                   f"{out_dir} and run again.")
            raise ValueError(
                f"{source}: {_percent(share)} of its sentence text (sentences of "
                f"{OVERLAP_MIN_SENTENCE_CHARS} characters or more) is already in "
                f"{'this call' if in_call else out_dir}; the largest shares are in "
                f"{_sources_phrase(by_source, total)}. Prepared beside it, that text would be in "
                f"the corpus twice, and training could hold out a document whose text it trains "
                f"on. {fix} Nothing was written.")
        if shared:
            notes.append(
                f"{source} shares {_percent(share)} of its sentence text with "
                f"{_sources_phrase(by_source, total)} -- typically a licence or front matter. "
                f"Shared text is in the corpus twice and can fall on both sides of the held-out "
                f"split: see \"Strip the boilerplate first\" in docs/preparing-your-data.md. "
                f"Prepared anyway.")
        for sentence in sentences:
            index.setdefault(_sentence_hash(sentence), set()).add((True, source))
    return notes


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
    split_chars: int | None = None,
    val_fraction: float = DEFAULT_VAL_FRACTION,
    require_held_out: bool = False,
) -> list[Path]:
    """Extract, clean and write the domain documents; return the files written, in order.

    Args:
        inputs: files and/or directories to read. Directories are searched for
            ``.txt``/``.md``/``.html``/``.htm``/``.pdf``; anything else is ignored.
        out_dir: directory to write into; created if absent. Existing files are never
            overwritten -- a name collision gets a ``_1``, ``_2``, ... suffix.
        min_length: drop an input shorter than this many characters *after cleaning*. It is
            applied to the whole input, before any splitting.
        combine: write one ``combined_domain_data.txt`` (documents separated by a rule and headed
            by ``# Source: <filename>``) instead of one file per input. It is suffixed like any
            other output when that name is taken, so a second ``--combine`` run over different
            inputs writes ``combined_domain_data_1.txt`` rather than replacing the first corpus.
        recursive: search directories' subdirectories.
        split_chars: cut every cleaned input into documents of about this many characters at
            paragraph boundaries (:func:`split_into_documents`), written as ``<stem>-0001.txt``,
            ``<stem>-0002.txt``, ... ``None``, the default, writes one document per input.
            :data:`SUGGESTED_SPLIT_CHARS` (3,500) is the walkthrough's size; below
            :data:`MIN_SPLIT_CHARS` (500) is refused.
        val_fraction: the held-out share the corpus will be trained under, used only to check
            that it can be split: a corpus (what ``out_dir`` already holds, plus what this call
            adds) with fewer documents than that split needs is logged at WARNING, naming
            ``split_chars`` as the fix. Default: the bundled recipes' 0.1, which needs 2.
        require_held_out: refuse such a corpus instead, before anything is written -- and,
            when there is no split and the input files alone cannot make enough documents,
            before any file is read. The CLI sets it under ``--supplement``, so the pairs are
            never written for a corpus the trainer cannot hold anything out of.

    Raises:
        ValueError: no supported file is found under ``inputs``; ``split_chars`` together with
            ``combine``, or below :data:`MIN_SPLIT_CHARS`; an input of which
            :data:`REPREPARATION_SHARE` or more of the sentence text is already in ``out_dir``
            or in an earlier input (always: the same text twice lets the held-out split score
            text the run trains on -- a smaller share is warned about); or, under
            ``require_held_out``, too few documents.
        MissingExtra: if a found file needs the ``[html]`` or ``[pdf]`` extra and it is missing.
    """
    out_dir = Path(out_dir)
    if split_chars is not None:
        if combine:
            raise ValueError("--split-chars and --combine do not go together: --combine writes "
                             "one file, which the loader reads as one document whatever was "
                             "split inside it.")
        if split_chars < MIN_SPLIT_CHARS:
            raise ValueError(
                f"--split-chars {split_chars} is below the floor of {MIN_SPLIT_CHARS} characters: "
                f"a split document can be about half the size, and the trainer drops a whole "
                f"document under {MIN_CHUNK_TOKENS} tokens without a word, so a smaller size "
                f"could lose text silently. {SUGGESTED_SPLIT_CHARS} is the walkthrough's size.")

    files = find_input_files(inputs, recursive=recursive)
    if not files:
        raise ValueError(
            f"No supported files ({', '.join(sorted(SUPPORTED_EXTENSIONS))}) found in {inputs}")

    # What the trainer will read is whatever `out_dir` already holds plus what this call adds.
    # Without a split, every input is at most one document, so a refusal that is already
    # certain is made here -- before a PDF loads marker's models or anything is extracted.
    n_existing = len(load_texts(out_dir)) if out_dir.is_dir() else 0
    if require_held_out and split_chars is None:
        at_most = n_existing + (1 if combine else len(files))
        if held_out_shortfall(at_most, val_fraction):
            raise ValueError(_too_few_documents(out_dir, at_most, val_fraction, combine=combine,
                                                written=False, at_most=True)
                             + " Nothing was read or written.")

    # Load the marker models once, and only if a PDF is actually in the batch.
    pdf_converter = None
    if any(p.suffix.lower() in PDF_EXTENSIONS for p in files):
        logger.info("Loading marker models for %d PDF(s)...",
                    sum(1 for p in files if p.suffix.lower() in PDF_EXTENSIONS))
        pdf_converter = make_pdf_converter()

    # Everything is extracted, cleaned and split before anything is written, so a corpus that
    # is refused for its document count leaves nothing behind to be doubled by the re-run.
    documents: list[tuple[str, str, str]] = []           # (output stem, source name, text)
    whole_texts: list[tuple[str, str]] = []              # (input path, cleaned text)
    n_kept, n_skipped, n_characters = 0, 0, 0

    for file_path in files:
        text = clean_text(extract(file_path, pdf_converter))
        if len(text) < min_length:
            logger.info("Skipping %s: %d characters after cleaning (min_length=%d)",
                        file_path.name, len(text), min_length)
            n_skipped += 1
            continue

        n_kept += 1
        n_characters += len(text)
        whole_texts.append((str(file_path), text))
        if split_chars is None:
            documents.append((file_path.stem, file_path.name, text))
        else:
            pieces = split_into_documents(text, split_chars)
            documents += [(f"{file_path.stem}-{index:04d}", file_path.name, piece)
                          for index, piece in enumerate(pieces, start=1)]

    # Before the count and before any write: a text already in `out_dir` (or given twice) is
    # refused whatever the count, and shared boilerplate is warned about here, once.
    for note in _check_text_already_there(out_dir, whole_texts):
        logger.warning(note)
    combined = (_SEPARATOR.join(f"# Source: {name}\n\n{text}" for _, name, text in documents)
                if combine and documents else None)

    n_documents = n_existing + (min(1, len(documents)) if combine else len(documents))
    shortfall = held_out_shortfall(n_documents, val_fraction)
    if shortfall and require_held_out:
        raise ValueError(_too_few_documents(out_dir, n_documents, val_fraction, combine=combine,
                                            written=False) + " Nothing was written.")

    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    if combine:
        if combined is not None:
            # Through `_unique_output` like every per-file write: under `combine` this one file
            # IS the corpus, so an overwrite costs the whole of a previous preparation rather
            # than one document of it -- and the docstring above promises it does not happen.
            out_path = _unique_output(out_dir, Path(COMBINED_NAME).stem)
            out_path.write_text(combined, encoding="utf-8")
            written.append(out_path)
    else:
        for stem, _, text in documents:
            out_path = _unique_output(out_dir, stem)
            out_path.write_text(text, encoding="utf-8")
            written.append(out_path)

    logger.info("Prepared %d document(s) from %d file(s) into %s (%d skipped, %d characters, "
                "~%d tokens)", len(written), n_kept, out_dir,
                n_skipped, n_characters, n_characters // 4)
    if shortfall:
        logger.warning(_too_few_documents(out_dir, n_documents, val_fraction, combine=combine,
                                          written=True))
    return written
