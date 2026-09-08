# Terraria EN: curated SFT corpus

English Terraria corpus for experimenting with QLoRA/SFT. The canonical source
contains **200 conversations from 50 Official Terraria Wiki articles**:

- **160 training conversations** from 40 source articles;
- **24 validation conversations** from 6 different source articles;
- **16 final-evaluation conversations** from 4 additional source articles.

The split is performed by source article, not by randomly moving individual
questions. This avoids evaluating on paraphrases of the same article that the
model already saw during training.

## Files

The versioned, editable source of truth is under `data/`:

- `data/metadata.json`: scope, system prompt, attribution and split policy;
- `data/sources-train-core.json`: translated core training topics;
- `data/sources-train-extra.json`: additional English training topics;
- `data/sources-train-more.json`: further NPC/progression/material topics;
- `data/sources-validation.json`: development/validation topics;
- `data/sources-evaluation.json`: final held-out topics.

`build.py` materializes the JSONL files locally:

- `../../examples/terraria-training.jsonl`: selectable SFT training dataset;
- `validation.jsonl`: assistant-target validation set used for checkpoint
  selection; it is intentionally outside the GUI's selectable SFT folders;
- `evaluation.jsonl`: final held-out evaluation without assistant targets.

These JSONL files are generated artifacts and are gitignored. Opening
`./fine-tune-gui` rebuilds them automatically from the versioned English corpus.

## Why validation was added

The previous training loop only reported training-side metrics such as loss,
mean token accuracy, entropy and gradient norm. Those are useful diagnostics but
cannot tell whether a later checkpoint generalizes better: they are measured on
examples the optimizer already sees.

When `terraria-training.jsonl` is trained through the normal script, its separate
`validation.jsonl` is detected automatically. `trainer/train.py` then evaluates
once before training and once per epoch, selects the checkpoint with the **lowest
`eval_loss`**, reloads that checkpoint and saves it as the final adapter.

The manifest records:

- baseline validation loss and perplexity;
- selected validation loss and perplexity;
- best checkpoint and best metric;
- relative validation-loss improvement.

Perplexity is `exp(eval_loss)` and is included as an easier-to-read transform of
the same token-level objective. The primary selection criterion remains
`eval_loss`: lower is better.

The **16 final evaluation questions are not used for checkpoint selection**. Do
not repeatedly tune against them; they are intended as a final regression and
generalization check.

## Scope and data quality

The corpus covers unmodded Terraria on modern PC versions and ordinary worlds
unless a question says otherwise. Topics include early progression, health and
mana, crafting, housing, NPCs, mobility, Hardmode, bosses, fishing, pylons,
accessories, the Jungle Temple, Meteorite and several crafting/progression
systems.

Questions and answers are synthetic English paraphrases based on reviewed
Official Terraria Wiki pages or indexed excerpts. This is not a complete wiki
dump, a frozen snapshot of one patch, or a substitute for RAG when current,
source-cited factual coverage is required.

Moving the experimental corpus to English reduces the mixed-language burden for
the base model and makes the training/evaluation language consistent. It does
not by itself guarantee higher quality; that is why the validation split and
final held-out evaluation remain necessary.

## Rebuild and validate

From the repository root:

```bash
python trainer/corpora/terraria/build.py
python trainer/corpora/terraria/build.py --check
python trainer/validate_dataset.py trainer/examples/terraria-training.jsonl
python trainer/validate_dataset.py trainer/corpora/terraria/validation.jsonl
```

## License and attribution

The corpus is derived from contributors to Official Terraria Wiki (wiki.gg).
Terraria and its materials belong to Re-Logic. Changes include selection of
facts, English paraphrasing, question authoring and source-level splitting.
Images, audio and game-dialogue dumps are not included.

Wiki-derived text is handled under **CC BY-NC-SA 4.0**, following the wiki's
copyright page. Keep attribution, links and share-alike/non-commercial terms
when redistributing the dataset. The surrounding project license does not turn
this corpus into MIT-licensed data. Review both the wiki terms and the base-model
license before distributing trained weights or using them commercially.
