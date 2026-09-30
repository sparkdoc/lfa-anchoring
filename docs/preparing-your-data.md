# Preparing your data

What `train` reads is a **corpus**: a directory of documents, one per file. This page covers
getting your files into that shape, the three corpus shapes that train badly, and the
question-and-answer **supplement** the model writes over the corpus before training.

## From your files to a corpus

```bash
lfa prepare-domain ~/papers ~/notes.md --out data/my_domain
```

`prepare-domain` takes files and directories (a directory is searched, subdirectories included,
for the formats below; anything else in it is ignored) and writes one cleaned `.txt` per input
document into `--out`:

* `.txt` and `.md` pass straight through;
* `.html` / `.htm` need the `[html]` extra (`pip install 'lfa-anchoring[html]'`);
* `.pdf` needs the `[pdf]` extra (`pip install 'lfa-anchoring[pdf]'`).

A missing extra is refused, naming the file and the extra, rather than skipped: a corpus is never
quietly half-prepared.

**Cleaning.** Every document goes through the same pass: the markup the extractors leave behind
(footnote wrappers, link targets, image references, stray tags) is removed and mid-paragraph line
breaks are unwrapped, so a paragraph is one line. A document that cleans down to less than
`--min-length` characters (default 1000) is dropped — a page that extracted to a nav bar and a
cookie notice is not training data.

**The layout.** The output is a flat directory of `.txt` files, and the loader reads each file as
one document. A directory of `.txt` or `.md` files you already have is a corpus as it stands.
`prepare-domain` never overwrites: a name that is already taken gets a `_1`, `_2`, … suffix, so a
second run into the same directory adds its documents beside the first run's. Write into a fresh
directory.

**`--combine`** writes one combined file instead of one per input, headed and separated by
source. Use it to read the cleaned text, not to train on: the loader would read the whole corpus as
a single document, and training would warn that one document dominates.

**Point `--corpus` at the corpus directory itself** (`data/my_domain`), never at its parent. A
supplement prepared with the data sits beside the corpus in `data/my_domain.supplement/`, and a
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
* **One document dominating.** A single document past half the chunks contributes more gradient
  than the whole of the rest of the corpus, so the run is at least as much a fine-tune on that one
  document. The warning names the document and its share. Split it at its own section boundaries.
* **More epochs than the text can carry.** Under 500,000 training tokens (~2 MB of English) at more
  than five epochs. The shipped 15 were tuned on about 6.6 MB; the two-domain walkthrough, at 164 k
  tokens a stage, reached its held-out minimum at epoch 4 and was worse by epoch 8. Start nearer
  five, keep `val_fraction` above 0, and let the held-out curve pick the dose — the trainer names
  the epoch it bottomed at when the run ends.

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
lfa prepare-domain ~/papers --out data/my_domain --supplement --model Qwen/Qwen3-0.6B
lfa prepare-supplement --corpus data/my_domain --model Qwen/Qwen3-0.6B    # a corpus you already have
```

Both have the model write the supplement for the corpus and print the path of the file, with
its manifest beside it:

```
data/my_domain.supplement/<hash>/supplement.jsonl
data/my_domain.supplement/<hash>/supplement.jsonl.manifest.json
```

`<hash>` is the first twelve hex digits of the training side's sha256. The pairs are written from
the training side only — the recipe's held-out fraction and seed decide which documents that is,
so held-out text never reaches the supplement. The recipe is `--recipe` when you pass one, and
otherwise the bundled recipe that names `--model`; with neither, the command is refused before
anything is written. `--supplement` without `--model` is refused the same way.

**Reading it.** `supplement.jsonl` holds one pair per line, `{"prompt": …, "response": …}`, in
the order they were written. The manifest says who wrote it and from what: the writer's model id
and checkpoint sha256, the training side's sha256, the domain description and the template it was
rendered into, the decoding settings, and how many passages and pairs there were, with what it
rejected counted by reason (an answer too short or too long, a duplicate, an unparseable passage).

**When `train` reuses it.** `train` looks for a supplement first in the workspace's
`supplements/`, then beside the corpus, and reuses one only when four things match: the training
side's hash, the writer checkpoint's hash, the template's hash and the domain description. So a
supplement prepared with the model a workspace starts from is used by that workspace's first stage
as it is. A chain's later stage starts from a fused model, a different writer, so it writes its
own into the workspace rather than training on pairs the base model wrote.

**`--domain-description`** is what the template says the text is on. The default is the corpus
directory's name with `_` and `-` read as spaces (`data/my_domain` → `my domain`). If you pass
one when preparing, pass the same one to `train`, or `train` sees a different description and
writes its own. `--force` rewrites a supplement that already exists.

Inside a workspace, `lfa prepare-supplement --corpus data/my_domain` (without `--model`) has the
workspace's current model write the file into the workspace's `supplements/` instead —
`--workspace` defaults to the current directory. That is the file `train` would write, written
ahead of time to inspect.

### Bringing your own

```bash
lfa train --workspace runs/my_domain --corpus data/my_domain --supplement my_pairs.jsonl
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
lfa train --workspace runs/my_domain --corpus data/my_domain --no-supplement
```

The run trains on the raw corpus alone and warns that it is off the frame λ was tuned at:
*"Training on the raw corpus alone: this recipe's lambda was calibrated at supplement_fraction
0.13 and this run mixes none."* A recipe that sets `supplement_fraction: 0.0` has opted out at the
recipe level and is not warned.

### What it costs

One generation per 4,000-character passage of the training side, in batches of 16 passages, once
per corpus and writer. [faq.md](faq.md#how-long-does-self-generation-take) has what the timed
pieces cost on an 8 GB card.
