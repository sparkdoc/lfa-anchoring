# Preparing your data

What `train` reads is a **corpus**: a directory of documents, one per file. This page covers
getting your files into that shape, the three corpus shapes that train badly, and the
question-and-answer **supplement** the model writes over the corpus before training.

## From your files to a corpus

```bash
lfa prepare-domain ~/history ~/notes.md --out data/world_history
```

`prepare-domain` takes files and directories (a directory is searched, subdirectories included,
for the formats below; anything else in it is ignored) and writes one cleaned `.txt` per input
document into `--out`:

* `.txt` and `.md` pass straight through;
* `.html` / `.htm` need the `[html]` extra (from the checkout, `pip install -c
  constraints-tested.txt -e '.[html]'`);
* `.pdf` needs the `[pdf]` extra (`pip install -c constraints-tested.txt -e '.[pdf]'`).

A missing extra is refused, naming the file and the extra, rather than skipped: a corpus is never
quietly half-prepared.

**Cleaning.** Every document goes through the same pass, and it is deliberately minimal: only
what every text shares, whatever its source.

* Line ends: CRLF, a lone CR and the other line breaks (a form feed at a page end, a vertical tab,
  a Unicode line or paragraph separator) become a newline.
* Invisible characters removed: the byte-order mark, zero-width spaces and joiners, soft hyphens,
  and every other control character but the newline.
* Space variants — a tab, a no-break space, the other Unicode spaces — become a plain space, and a
  run of spaces becomes one.
* Last, the characters are put in Unicode normal form NFC, so an accent typed as a letter plus a
  combining mark and the same accent as one character are the same text.
* Markup the extractors leave behind (footnote wrappers, link targets, image references, stray
  tags) is removed, and mid-paragraph line breaks are unwrapped, so a paragraph is one line.

What it does not touch is content: a licence, a table of contents, an index, page headers, a
transcriber's note are text like any other, and cutting them is yours to do (below). A document
that cleans down to less than `--min-length` characters (default 1000) is dropped — a page that
extracted to a nav bar and a cookie notice is not training data.

**One long file — a book, a report — split it.** A file is one document, and training holds out
whole documents (a tenth of them, at the bundled recipes' `val_fraction` 0.1), so a corpus needs at
least two documents before anything is held out. One downloaded book prepared as it stands is one
document: nothing is held out, so there is no per-epoch held-out curve to choose the number of
epochs by (the trainer says so when the run ends), and the domain number is a fit rather than a
measurement. `--split-chars` cuts every
file into documents of about that many characters at paragraph boundaries:

```bash
lfa prepare-domain book.txt --out data/world_history --split-chars 3500
```

The documents are written as `book-0001.txt`, `book-0002.txt`, …. 3,500 characters (about 850
tokens, so most documents are one or two of the trainer's 512-token chunks) is the size the
two-domain walkthrough splits its books at. A document closes at the first paragraph end past
the size; a single paragraph longer than twice the size is cut inside it, at a sentence end where
there is one and at whitespace where there is not; a remainder at the end of a file shorter than
half the size joins the document before it, so splitting drops no text and every document is at
least half the size (a file shorter than that stays one document). The size has a floor of 500
characters: the trainer drops a whole document under ten tokens without a word, and at 250
characters or more no prose document is that short. (Like any document, each one can still lose
a final piece under ten tokens where the chunker's cut leaves one.)
`--min-length` applies to the whole file, before it is split. A corpus with fewer documents than a
held-out split needs is warned about, with this fix named, and refused wherever a supplement would
be written for it: `prepare-domain --supplement` refuses it before anything is written (below),
and `train`, `prepare-supplement` and their Python counterparts refuse it before writing any pairs
(`train --no-supplement` trains it as it is, with no held-out curve). When the input files cannot
make enough documents however they extract, `prepare-domain --supplement` refuses before any of
them is read.

**Strip the boilerplate first.** A Project Gutenberg licence, a table of contents or an index
left in the file is trained on like the text around it, and the supplement writes questions about
it. Cut them out of the file before preparing it.

**The layout.** The output is a flat directory of `.txt` files, and the loader reads each file as
one document. A directory of `.txt` or `.md` files you already have is a corpus as it stands.
`prepare-domain` never overwrites: a name that is already taken gets a `_1`, `_2`, … suffix, so a
second run into the same directory adds its documents beside the first run's. Write into a fresh
directory. Each input is compared, sentence by sentence, with what the directory already holds
and with the inputs before it in the same command; only sentences of 60 characters or more count,
since headings and short lines recur across unrelated texts. Sentences survive splitting at any
size — `--split-chars` cuts at paragraph breaks and sentence ends — and a text of short paragraphs
(an FAQ, a play) still has them. An input with half or more of its sentence text already there is
refused before anything is written, naming the files with the largest shares: the same text twice in
a corpus lets the held-out split score text the run trains on. That is what re-running a one-file
preparation into the same directory with `--split-chars` would do — the whole book beside its own
pieces — and it is refused even when the text was edited in between (a header stripped, a word
changed) and whatever size either preparation was split at. Prepare into a fresh directory, or
delete the earlier preparation of that text first. An input with less than half shared is
prepared with a warning naming the files it shares text with: typically a licence or front matter
that two sources both carry, which is the boilerplate above to strip.

**`--combine`** writes one combined file instead of one per input, headed and separated by
source. Use it to read the cleaned text, not to train on: the loader would read the whole corpus as
a single document, which nothing can be held out of. `prepare-domain` warns that it is one
document, and everything that would write a supplement for it refuses it (above); it does not go
with `--split-chars`.

**Point `--corpus` at the corpus directory itself** (`data/world_history`), never at its parent. A
supplement prepared with the data sits beside the corpus in `data/world_history.supplement/`, and a
`--corpus data/` would read its `.jsonl` and manifest as documents.

**Host RAM.** The whole corpus is tokenized eagerly and held in host RAM at **about eight times
its size on disk**, for the whole run. A corpus too large for the machine is refused before it is
tokenized rather than killed part-way through. [faq.md](faq.md#how-large-a-corpus-can-i-train-on)
has the measurements and the override.

## Three shapes that train badly

Three shapes train badly without failing — the losses fall, the counts look ordinary, and the
model that comes out is quietly worse than the corpus could have made it. The trainer reads them
off the corpus **before the first step** and names the numbers it read them from. None is a
refusal.

* **Too few chunks for the batch.** Under eight optimizer steps an epoch, each step's gradient
  comes from more than an eighth of the corpus, so consecutive steps see nearly the same examples
  and the shuffle buys almost nothing; at the recipe's 50-step warmup such a run also spends its
  first six epochs or more below the learning rate the operating point was tuned at. Add documents,
  lower `batch_size`, or lower `sequence_length` so each document yields more chunks.
* **One document dominating.** A single document past half the domain's chunks contributes more
  gradient than the rest of the domain's documents together, so the run is at least as much a
  fine-tune on that one document. Supplement pairs are not counted as documents here — they are
  written from the domain's documents — so a single book with its pairs beside it is one document,
  and is warned about. The warning names the document and its share, and reports the domain's
  documents and the supplement pairs separately. Split it at its own section boundaries, or with
  `prepare-domain --split-chars 3500` into a fresh directory.
* **More epochs than the text can carry.** Under 500,000 training tokens (~2 MB of English) at more
  than five epochs. At the bundled recipes' λ (1,000,000, both models) the two-domain walkthrough's
  Darwin text (~189 k training tokens an epoch) turned late and shallowly over 15 epochs — its
  held-out perplexity lowest at epoch 10 and ending 2.6 % above it (Qwen3-0.6B), lowest at epoch
  11 and ending 0.2 % above it (Qwen3-1.7B); no run at 10 or 11 epochs was measured — while at
  λ = 100,000 the same text turned by epoch 4–5. If λ is lowered or the corpus is smaller still,
  keep `val_fraction` above 0 and let the held-out curve pick the dose — at the end of every run
  the trainer reports the epoch it bottomed at, the final value and the gap between them, and,
  when the run did not end at its best, advises a re-run at that epoch (1 % or more above the
  lowest) or calls one optional (under 1 %)
  ([recipes.md](recipes.md#run-length-and-checkpointing) has the rule).

A sound corpus produces none of them. To read them without starting a run,
`ChunkedCorpus.shape_warnings(batch_size=…, epochs=…)` returns them as a list of strings.

## The supplement

**What it is.** The model reads each training-side passage (up to 4,000 characters) and writes
six question-and-answer pairs about it from a fixed template. The domain content comes from the
passage; only the question-forming, the answer construction and the assistant's voice come from
the model. The writer is the model the stage starts from — the one you will train.

**What it is for.** Reaching the domain's knowledge when the model is asked about it: the pairs
make what the corpus says answerable in question-and-answer form. The recipe mixes them in at its
`supplement_fraction`, 0.13 of training tokens, which is the frame the recipe's λ was tuned at.

**What it is not for.** It does not protect skills. A supplement written in a skill's mode left
that skill no better, measured on instruction following and reasoning — one model (Qwen3-0.6B),
one seed, one domain. The anchor is what keeps skills.

The held-out tenth is split off before anything is mixed, and stays raw text, so the domain number
is a raw-text measurement whether or not a supplement was mixed in.

### Preparing it with the data

```bash
lfa prepare-domain ~/history --out data/world_history --supplement --model Qwen/Qwen3-0.6B
lfa prepare-supplement --corpus data/world_history --model Qwen/Qwen3-0.6B    # a corpus you already have
```

Both have the model write the supplement for the corpus and print the path of the file, with
its manifest beside it:

```
data/world_history.supplement/<hash>/supplement.jsonl
data/world_history.supplement/<hash>/supplement.jsonl.manifest.json
```

`<hash>` is the first twelve hex digits of the training side's sha256. The pairs are written from
the training side only — the recipe's held-out fraction and seed decide which documents that is,
so held-out text never reaches the supplement. The recipe is `--recipe` when you pass one, and
otherwise the bundled recipe that names `--model`; with neither, the command is refused before
anything is written. `--supplement` without `--model` is refused the same way, and so is a corpus
with too few documents for the recipe's held-out split (one file without `--split-chars`): the
supplement is minutes of generation, and the pairs would be written from text the trainer cannot
hold anything out of. The refusal names the fix and writes nothing, so the same `prepare-domain`
command re-run with `--split-chars 3500` starts clean.

**Reading it.** `supplement.jsonl` holds one pair per line, `{"prompt": …, "response": …}`, in
the order they were written. The manifest says who wrote it and from what: the writer's model id
and checkpoint sha256, the training side's sha256, the domain description and the template it was
rendered into, the decoding settings, and how many passages and pairs there were, with what it
rejected counted by reason: a passage whose output held no parseable pair (`unparseable_passage`),
a pair whose question or answer carries the writer's own JSON field syntax — `"answer": …` or
`{"question"` inside the text, where the writer botched an object's quoting and the next field ran
into this one (`leaked_json`; braces and quotes alone are kept) — an answer too short or too long
(`short_answer`, `long_answer`), and a question already written (`duplicate`). The same counts end
the log of a write; no pair is printed.

**When `train` reuses it.** `train` looks for a supplement first in the workspace's
`supplements/`, then beside the corpus, and reuses one only when five things match: the training
side's hash, the writer checkpoint's hash, the template's hash, the pair filters' hash (which
reasons drop a pair, and how leaked JSON is matched) and the domain description. A supplement
written under other filters is written again rather than trained on. So a
supplement prepared with the model a workspace starts from is used by that workspace's first stage
as it is. A chain's later stage starts from a fused model, a different writer, so it writes its
own into the workspace rather than training on pairs the base model wrote.

**`--domain-description`** is what the template says the text is on. The default is the corpus
directory's name with `_` and `-` read as spaces (`data/world_history` → `world history`). A generic
container name — `train`, `test`, `val`, `data`, `corpus`, `texts`, `raw` and the like — says
nothing about the domain, so the nearest enclosing directory with a real name is used instead
(`data/darwin/train` → `darwin`), looking no higher than your home directory, whose name is
yours rather than the domain's; `the domain` if there is none. If you pass
one when preparing, pass the same one to `train`, or `train` sees a different description and
writes its own. `--force` rewrites a supplement that already exists.

Inside a workspace, `lfa prepare-supplement --corpus data/world_history` (without `--model`) has the
workspace's current model write the file into the workspace's `supplements/` instead —
`--workspace` defaults to the current directory. That is the file `train` would write, written
ahead of time to inspect, and it is refused for a corpus with nothing to hold out, as above.

### Bringing your own

```bash
lfa train --workspace runs/world_history --corpus data/world_history --supplement my_pairs.jsonl
```

A file of your own is one JSON object per line with a `prompt` and a `response`; a line without a
`prompt` is skipped. Each pair is rendered through the model's chat template as a user turn and an
assistant turn (joined plainly when the model has no template), and mixed in at the recipe's
fraction: the pairs are taken as a prefix of the file, the one whose achieved share is closest to
the target, and a file too small to reach it trains at what it has and warns with the achieved
share. Keep the file outside the corpus directory — anything under `--corpus` is read as a
document.

### Training without one

```bash
lfa train --workspace runs/world_history --corpus data/world_history --no-supplement
```

The run trains on the raw corpus alone and warns that it is off the frame λ was tuned at (it is
also how a corpus with nothing to hold out trains at all):
*"Training on the raw corpus alone: this recipe's lambda was calibrated at supplement_fraction
0.13 and this run mixes none."* A recipe that sets `supplement_fraction: 0.0` has opted out at the
recipe level and is not warned.

### What it costs

One generation per 4,000-character passage of the training side, in batches of 16 passages, once
per corpus and writer. [faq.md](faq.md#how-long-does-self-generation-take) has what the timed
pieces cost on an 8 GB card.
