# Next steps: country-only model

Branch `country-only`. Models: `xseedo/tinyllm-countries-base` (pretrained on country snippets only,
748 steps) and `xseedo/tinyllm-countries-closed` (closed-book Q&A fine-tune).

## Baseline (2026-11-01)

```
python src/eval.py --model xseedo/tinyllm-countries-closed --data xseedo/tinyllm-countries-closed --split test
```

Held-out countries (19 countries the model never saw as Q&A; their facts were in pretraining):

| Attribute | Score |
|---|---|
| **overall** | **229/292 = 78.4%** |
| capital | 62/76 = 81.6% |
| continent | 75/76 = 98.7% |
| currency | 58/76 = 76.3% |
| language | 34/64 = 53.1% |

Treat this as ~19 data points, not 292: errors cluster per country (Madagascar alone is ~10 of the
63 misses), so the score can move by ±10 points between runs.

## What the errors show

The base model knows every missed fact. Prompted in pretraining wording, it completes all 11
checked misses correctly:

| Base model (sentence completion) | Closed model (question) |
|---|---|
| The capital of Burundi is → **Gitega** ✓ | → Rabat ✗ |
| The money in Türkiye is the → **Turkish lira** ✓ | → Turkish đồng / rufiyaa ✗ |
| The money in Madagascar is the → **Malagasy ariary** ✓ | → Malagasy shilling ✗ |
| People in Lithuania speak → **Lithuanian** ✓ | → English ✗ |

Also correct in the base model: Riga, Ciudad de la Paz, Jordanian dinar, Lebanese pound, Korean,
French and Malagasy, Greek and Turkish.

So the knowledge is there. The step from a question to the stored fact is what fails. The Q&A model
often builds a plausible answer instead of recalling one (country adjective + a common currency
word: "Malagasy shilling", "Lebanese shilling"), or pulls another country's whole record (Burundi →
Morocco: Rabat, Arabic and French, Moroccan dirham).

It isn't a scoring problem: the language misses are wrong languages, not the right two in a
different order.

## Experiments

Each one changes only the closed-book fine-tune (~5 min on HF Jobs). Compare on the same `--split test`.

1. **Less fine-tuning.** Currently 4 epochs at lr 1e-4 on ~2,000 rows. That's likely enough to
   memorize question → answer for the trained countries instead of learning question → look it up.
   Try 1–2 epochs. Needs `--epochs` and `--lr` flags on `train.py`. Start here: it's one setting.
2. **Mix pretraining sentences into the fine-tuning data** (e.g. 50% country snippets), so the
   fact-sentence pathway stays active while the model learns the question format.
3. **Final model with `--holdout 0`.** Every country gets Q&A training. First check the trained
   countries:
   ```
   python src/eval.py --model xseedo/tinyllm-countries-closed --data xseedo/tinyllm-countries-closed --split train --limit 300
   ```
   If that's close to 100%, retrain with `--holdout 0` for a model that answers all 194 countries.
   You lose the held-out test, so do it after experiments 1–2.
