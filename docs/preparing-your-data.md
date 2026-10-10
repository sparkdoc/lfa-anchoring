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
* Markup the extractors leave behind is removed: `<sup>`, `<sub>` and `<span>` tags (their text
  kept), Markdown footnote wrappers and `[[…]]` editorial brackets, link targets (the link text
  kept), Markdown image references (`![alt](path)`), horizontal rules, and a line that is bold
  Markdown and nothing else (an affiliation, an e-mail address). Mid-paragraph line breaks are
  unwrapped, so a paragraph is one line.

What it does not touch is content: a licence, a table of contents, an index, page headers, a
transcriber's note are text like any other, and so is a caption in plain text such as Project
Gutenberg's `[Illustration: …]` — it is not an image reference in the sense above, and it is kept.
Cutting them is yours to do ([a Project Gutenberg book](#a-project-gutenberg-book), below). A
document that cleans down to less than `--min-length` characters (default 1000) is dropped — a
page that extracted to a nav bar and a cookie notice is not training data.

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
it. Cut them out of the file before preparing it;
[a Project Gutenberg book](#a-project-gutenberg-book) below lists what one carries and how to cut
each part.

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

### A Project Gutenberg book

A Gutenberg plain-text book carries more than its licence, and cleaning keeps all of it. What to
look for, with what each step did to *A Short History of Astronomy* (Arthur Berry, 1898; eBook
#59212, `pg59212.txt`, 18,695 lines):

* **The START and END markers, and the licence around them.** Everything up to the line
  `*** START OF THE PROJECT GUTENBERG EBOOK … ***` and everything from `*** END OF THE PROJECT
  GUTENBERG EBOOK … ***` on is Gutenberg's, not the book's. One line cuts both, the marker lines
  included:

  ```bash
  sed '1,/\*\*\* START OF/d; /\*\*\* END OF/,$d' pg59212.txt > book.txt
  ```

  On that book it kept 18,315 lines. The walkthrough's `strip_gutenberg()` is the same cut in
  Python ([`examples/two_domain_walkthrough.ipynb`](../examples/two_domain_walkthrough.ipynb),
  section 2, the cell that downloads the two books). Two lines of Gutenberg's survive it there: a
  producer's credit ("Produced by … Distributed Proofreading Team …") just after the START marker
  and an "End of Project Gutenberg's …" line just before the END one; `grep -n -i gutenberg
  book.txt` finds the second.
* **Front matter.** A transcriber's note, a frontispiece caption, the title page, the table of
  contents and a LIST OF ILLUSTRATIONS: notes and lists, not the book's prose. Cut them by hand.
* **Illustration captions in the text**, such as `[Illustration: FIG. 1.—The celestial sphere.]`.
  They are short descriptions of figures the text cannot show. To drop them, cut from a line that
  starts `[Illustration` through the first line holding a `]`, which covers a caption wrapped over
  several lines:

  ```bash
  awk '/^\[Illustration/ {skip = 1} skip {if (/\]/) skip = 0; next} {print}' book.txt > book_cut.txt
  ```

  On that book it removed all 114 captions, 37 of them wrapped over two to five lines (159 lines
  in all), and nothing else. `sed '/^\[Illustration/,/\]/d'` is not the same: sed looks for the
  closing `]` only from the line after the first, so it runs a one-line caption on into the text
  below. A caption holding a `]` of its own before its end stops the cut early, so check that
  `grep -c Illustration book_cut.txt` prints 0.
* **Footnotes collected at the end** (a FOOTNOTES section, with markers `[1]`, `[2]`, … in the
  text): your call. Kept, they are trained on as text of their own, away from the passage each one
  annotates; after `--split-chars` they make documents of their own, and the held-out split may
  take some. Cut, the book loses what they say and the markers stay in the prose, where
  `sed 's/\[[0-9]\+\]//g'` removes them.
* **Indexes** (that book has an INDEX OF NAMES and a GENERAL INDEX): names and page numbers. Cut
  them.

To cut a section by hand, find where it starts with `grep -n` — `grep -n -E
'^(CONTENTS|LIST OF ILLUSTRATIONS|CHAPTER I\.|INDEX|GENERAL INDEX|FOOTNOTES)' book.txt` on that
book printed lines 196, 1127, 1361, 14970, 15809 and 17454 — and delete the range with `sed -i
'14970,17453d' book.txt`, working from the end of the file up so the earlier numbers stay right.
Gutenberg's CRLF line ends make no difference to these commands, and cleaning turns them into
newlines.

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
  than five epochs. The warning opens with your run's own numbers — its epochs and its training
  tokens an epoch — and says to read the held-out line at the end of the run. At the bundled
  recipes' λ (160,000, both models) the two-domain walkthrough's Darwin text (~189 k training
  tokens an epoch) turned early in the recipe's 15 epochs — its held-out perplexity lowest at
  epoch 4 and ending 80.0 % above it (Qwen3-0.6B), lowest at epoch 5 and ending 47.4 % above it
  (Qwen3-1.7B) — and the re-runs at those epochs were below the base model on both axes. On a
  corpus this size, or smaller, keep `val_fraction` above 0 and let the held-out curve pick the
  dose — at the end of every run the trainer reports the epoch it
  bottomed at, the final value and the gap between them, and, when the run did not end at its best,
  advises a re-run at that epoch (1 % or more above the lowest) or calls one optional (under 1 %)
  ([recipes.md](recipes.md#run-length-and-checkpointing) has the rule).

A sound corpus produces none of them. To read them without starting a run,
`ChunkedCorpus.shape_warnings(batch_size=…, epochs=…)` returns them as a list of strings.

## The supplement

**What it is.** The model writes it in two passes over each training-side passage (up to 4,000
characters). First it writes six questions about the passage from a fixed template, as a JSON
array of strings. Then it answers each question in one ordinary chat turn: "Here is a passage from
a text on <domain>:", the passage in triple quotes, and "Using the passage, answer: <question>".
There is no length, sentence, format or voice instruction, so the answer comes in the model's own
default chat format (bold lead-ins, lists and all, for Qwen3). The training pair is the question
and that answer, without the passage, so a question or answer that cites "the passage" or "the
text" is dropped — and under this prompt many answers do. The domain content comes from the
passage; only the question-forming and the answer's construction come from the model. The writer
is the model the stage starts from — the one you will train.

**Why this answer turn.** The one-pass writer of earlier versions asked for the pairs as JSON,
answers in the third person in 2–5 sentences. Its answers were short plain prose (191 characters on
average on the Darwin text, none with markdown, 52.7 % speaking of "the passage", "the text" or "the
author"), and a model trained on them answered in that style everywhere, under the anchor: 0 of 40
general prompts with any markdown against 39 of 40 for the base model, 0 of 20 Darwin prompts
against 19 of 20, 0 of the judge's 80 Darwin questions against 78 of 80, general answers of 223
characters against 1,095, and 6 of the 40 general prompts declined as not in "the text". Trained on
this writer's supplement, the model answered 36 of 40, 15 of 20 and 66 of 80 with markdown, wrote
625-character general answers, declined none, and moved less from the base model's own answers (a
base-answer NLL shift of +0.106 against the one-pass writer's +0.222). It read held-out Darwin
perplexity 19.36 against the one-pass writer's 19.26, and WikiText-2 16.06 against 16.18. On
correctness and completeness (`gpt-5.6-luna` at medium effort, 80 Darwin questions, blinded, a
rubric frozen before judging, 0–4) it scored 1.337 against the one-pass writer's 1.575 (−0.237,
paired p = 0.071: not separable at this n), with zero-rates 0.350 and 0.338; the base model's level
was 1.050 (zero-rate 0.525). Qwen3-0.6B, the walkthrough's Darwin data, four epochs, one seed, one
greedy decode per prompt (the format-drift diagnostic, 2026-10-07/08). A variant that gave the
passage as a system message and the question alone kept more of the format, but was behind the
one-pass writer on the judged score (correctness and completeness, p = 0.010; correctness alone p =
0.007), which is why this turn ships. Telling the writer not to mention the passage suppressed the
format (markdown in 8 % and 14 % of the base model's answers under two such wordings, against 72 %
under this prompt), which is why the turn holds no such instruction and the filter does that job.
The measured runs mixed their supplement's unshuffled prefix (below), under the `refers_to_source`
filter as it stood before the "of"/"from" guards and the extended "the text" patterns. The
research's judged reachability result (below) was measured on the one-pass template, not on this
one.

**What it is for.** Reaching the domain's knowledge when the model is asked about it: the pairs
make what the corpus says answerable in question-and-answer form. The recipe mixes them in at its
`supplement_fraction`, 0.13 of training tokens, which is the frame the recipe's λ was tuned at.

**What it buys, against no supplement.** Qwen3-0.6B at the recipe's λ (160,000), 4 epochs on the
walkthrough's Darwin data, this writer's pairs shuffled, two training seeds; the same 80 Darwin
questions, answers decoded greedily up to 1,024 tokens and judged blinded by `gpt-5.6-luna` at
medium effort on a rubric frozen before judging (correctness plus completeness, 0–4). With the
supplement the model scored 1.350 and 1.375 (the two seeds; zero-rates 0.36 and 0.38), without it
0.938 and 1.087 (zero-rates 0.53 and 0.48), and the base model 1.100 (zero-rate 0.48). Paired over
the 80 questions and both seeds, the supplement is ahead by +0.350 (95 % CI +0.142 to +0.558,
paired t p = 0.0013), and ahead in each seed (+0.412, p = 0.0014; +0.287, p = 0.052). What that
gain is made of matters. Without the supplement answers ran long and some looped: 6 and 11 of the
80 still hit the 1,024-token cap, about 5,300 characters each, with 44–73 % of the words in their
last 200 repeats. On the 66 questions where no arm's answer was cut off, the gap is +0.106 (p =
0.27), not separable. So the supplement's measured value is that the model ends its answers; it
is not shown to add correct content.

**What it costs: the model's format.** On 40 general prompts (one greedy decode each, the same two
seeds), the model trained with the supplement used headings in 0.00–0.03 of its answers, against
0.45 for the base model and 0.55–0.57 trained without it; its answers averaged 667–720 characters
against the base model's 1,095; it followed 7 and 8 of 10 explicit format instructions, against
10 of 10 for the base model and for the runs without it; and its base-answer NLL shift (the
behaviour reading of [tuning.md §4](tuning.md#4-λ-when-to-try-another)) was +0.13, against
+0.075–0.077 without it. Markdown of some kind stayed (0.90–0.95 of answers, base 0.97). Measured
2026-10-09, on this release's code.

**How many pairs a run uses.** The writer writes every pair it can, and `train` mixes in only as
many as the fraction asks for. It shuffles the file's pairs under the run's `seed` (the recipe's,
42), as the research mixer does, then takes pairs from the start of that order until they are 0.13
of the training tokens; the rest are not used by that run. The file is written passage by
passage, so without the shuffle a run would mix only the first passages' pairs — which is what
earlier builds did, the walkthrough's recorded runs and the format-drift diagnostic's arms
included. The fraction is of tokens, so longer answers mean fewer pairs: on the Darwin text the
one-pass writer's answers (191 characters on average) put 384 pairs in the mix, and this writer's
(657 characters on average, 67 % with markdown) 162 of the 840 it wrote, all from the file's
first 51 passages in 32 documents (the diagnostic above, unshuffled). Shuffled under seed 42, the
same file gives 158 pairs from 125 of the 265 passages, in 98 of the 173 training documents. With the
one-pass writer, and before the shuffle, *A Short History of Astronomy*'s supplement held 1,355
pairs and each run mixed in 389 of them. The
stage's `history.json` entry records both counts (`n_pairs_available`, `n_pairs_used`) and the
seed the pairs were shuffled under (`shuffle_seed`), and a
corpus-shape warning ([above](#three-shapes-that-train-badly)), when one fires, counts the pairs
mixed in (`189 document(s) + 389 supplement pair(s)`).

**What it is not for.** It does not protect skills. A supplement written in a skill's mode left
that skill no better, measured on instruction following and reasoning — one model (Qwen3-0.6B),
one seed, one domain, on the research's one-pass supplement. The anchor is what keeps skills.

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

**Reading it.** `supplement.jsonl` holds one pair per line, `{"prompt": …, "response": …,
"source_index": …, "passage_index": …}`, in the order they were written. `source_index` is the
training-side *document* the pair came from and `passage_index` the passage within that document,
counted as the writer cut it (`chunk_passages` at the manifest's `passage_chars` and
`min_passage_chars`). The manifest says who wrote it and from what: the writer's model id and
checkpoint sha256, the training side's sha256, the domain description, the question template and
the answer template it was rendered into with their hashes, the decoding settings (thinking off),
and how many passages, questions and pairs there were, with what it rejected counted by reason:

* `unparseable_passage` — a passage whose first-pass output held no question;
* `leaked_json` — a question carrying the writer's own JSON field syntax, `"answer": …` or
  `{"question"` inside the text, where the writer botched an object's quoting and the next field
  ran into this one; or an answer that is itself in that shape, opening on such an object or
  ending on such a field (braces, quotes and those field names in the middle of an answer — an
  example in a text about a JSON API — are kept; an answer whose closing code block ends on an
  `"answer"` field, such as `{"id": 1, "answer": "42"}`, reads as that shape and is dropped);
* `refers_to_source` — a question or answer that speaks of its source, which the training pair
  does not hold: "according to the passage", "in the passage", "the text states", "the text does
  not mention", and the like ("the passage of time", "in the text of the treaty" and "in the
  passage from Genesis 1" are kept), or an answer reporting what "the author states" when the
  question did not ask about the author; "the text also states", "the text references", "the
  text's argument" and "in the context of the text" count too ("in the context of the text
  messaging era" does not). A question caught here is never answered. Some domain phrasings can
  still be dropped, since no pattern tells them from a reference to the writer's passage or text:
  a passage of music or scripture spoken of as "the passage in bars 12–16", "the text is not
  considered canonical", "within the text, Hamlet says", a philologist's "the text's earliest
  surviving manuscript", "the text shows a chiastic structure";
* `duplicate` — a question already written (it is answered once);
* `truncated_answer` — an answer that had not ended its turn when it reached the writer's
  1,024-token cap;
* `short_answer` — an answer under 40 characters, a bare name or date, say.

The same counts end the log of a write; no pair is printed. How many are dropped depends on the
text. On the walkthrough's Darwin text (Qwen3-0.6B, the format-drift diagnostic's run of this
writer) `refers_to_source` dropped 750: 355 of the 1,591 questions parsed (22.3 %), then 395 of
the 1,235 answers (32.0 %) — answers to this prompt often cite the passage; one duplicate
went, nothing else, and 840 pairs were written. That run used the filter before its "of"/"from"
guards and its extended "the text" patterns; the guards change no verdict on the diagnostic's
recorded answers, and the extended patterns would drop 2 more of those 840 pairs (two questions
ending "in the context of the text?").

**When `train` reuses it.** `train` looks for a supplement first in the workspace's
`supplements/`, then beside the corpus, and reuses one only when six things match: the training
side's hash, the writer checkpoint's hash, the question template's hash, the answer template's
hash, the filters' hash (which reasons drop a question or answer, and how leaked JSON and a
reference to the source are matched) and the domain description. A supplement written under other
templates or filters — every supplement an earlier version wrote — is written again rather than
trained on. So a
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
fraction: the pairs are shuffled under the run's seed and taken as a prefix of that order, the
one whose achieved share is closest to the target, and a file too small to reach it trains at
what it has and warns with the achieved share. Keep the file outside the corpus directory — anything under `--corpus` is read as a
document.

### Training without one

```bash
lfa train --workspace runs/world_history --corpus data/world_history --no-supplement
```

The run trains on the raw corpus alone and warns that it is off the frame λ was tuned at (it is
also how a corpus with nothing to hold out trains at all):
*"Training on the raw corpus alone: this recipe's lambda was calibrated at supplement_fraction
0.13 and this run mixes none."* A recipe that sets `supplement_fraction: 0.0` has opted out at the
recipe level and is not warned. On the Darwin data, the runs without one kept the base model's format
and its compliance with format instructions, and wrote longer answers, some of which ran on to the
token cap ([above](#the-supplement)).

### What it costs

Two passes of generation, once per corpus and writer: one for each 4,000-character passage of the
training side (its questions), then one for each question kept (its answer), in batches of 16
prompts. The answers are the larger part: about six per passage, each up to 1,024 tokens. On the
walkthrough's Darwin text (265 passages, 1,235 questions, 840 pairs) it took 13 min 47 s on one
RTX 3090 with Qwen3-0.6B (2026-10-07; timed from the first logged batch of questions to the
closing `supplement:` line, so the model load and that first batch are outside it, with a second
such run on the host's other card): about 1.5 min of questions and 12 min of answers. [faq.md](faq.md#how-long-does-self-generation-take) has
the other timed pieces.
