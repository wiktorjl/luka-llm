# Tiny LLM — Preamble (tools path)

*Download pre-trained model from [here](https://huggingface.co/xseedo).*

Train a ~30M-parameter language model from random weights with Hugging Face tools, then fine-tune it
twice: closed-book Q&A (answers from memory) and open-book Q&A (answers from a fact card).
The full guide, with explanations and expected output for every step, is the "Tiny LLM — Preamble" doc.

## Files

| File | What it does |
| --- | --- |
| `src/build_country_data.py` | countries.json → cleaned facts + 5% country prose for pretraining |
| `src/build_mix.py` | clean, dedupe, shuffle TinyStories + country prose → train/val text |
| `src/train_tokenizer.py` | 8,192-token byte-level BPE with special tokens reserved |
| `src/export_for_trl.py` | tokenizer + text → Hugging Face datasets (optionally pushed to the Hub) |
| `src/build_sft_data.py` | closed-book Q&A and open-book fact cards (chapters 2–3) |
| `src/train.py` | one training script for all three stages; runs locally or on HF Jobs |
| `src/eval.py` | exact-match accuracy on a Q&A split |
| `src/demo.py` | demo prompts across the three models; `--progression` for chapter 1 |
| `src/app.py` | Gradio web demo |

## Commands, in order

Replace `NAME` with your Hugging Face username. Each block is safe to paste into zsh.

### 0. setup

Use Python 3.12. PyTorch has no build for Python 3.15 yet, and HF Jobs runs 3.12 too. `uv` installs 3.12 for you
(no Homebrew? `curl -LsSf https://astral.sh/uv/install.sh | sh`).

```bash
brew install uv
uv venv --python 3.12 .venv
source .venv/bin/activate
uv pip install -r requirements.txt
python -c "import sys, torch; print(sys.version.split()[0], torch.__version__, 'MPS:', torch.backends.mps.is_available())"
hf auth login
```

### 1. download

```bash
mkdir -p data
hf download roneneldan/TinyStories TinyStoriesV2-GPT4-train.txt TinyStoriesV2-GPT4-valid.txt \
  --repo-type dataset --local-dir data/tinystories
curl -L -o data/countries.json https://raw.githubusercontent.com/mledoze/countries/master/countries.json
```

### 2-4. pretraining data and tokenizer

```bash
python src/build_country_data.py --countries data/countries.json --out-dir data/ \
  --stories data/tinystories/TinyStoriesV2-GPT4-train.txt --fraction 0.05
python src/build_mix.py --stories data/tinystories/TinyStoriesV2-GPT4-train.txt \
  --val-stories data/tinystories/TinyStoriesV2-GPT4-valid.txt \
  --countries data/countries_prose.txt --out-dir data/
python src/train_tokenizer.py --train data/train_mix.txt --val data/val_mix.txt --out data/tokenizer.json
```

### 5-6. package for TRL and push to the Hub (private repos)

```bash
python src/export_for_trl.py --tokenizer data/tokenizer.json \
  --train data/train_mix.txt --val data/val_mix.txt --out-dir data/hf --hf-user NAME
python src/build_sft_data.py --countries data/countries_clean.jsonl --out-dir data/sft \
  --hf-out data/hf --hf-user NAME
```

### 7. smoke test on the laptop

```bash
python src/train.py --stage pretrain --local data/hf --smoke
python src/train.py --stage closed --local data/hf --smoke
python src/train.py --stage open --local data/hf --smoke
python src/demo.py --local out
```

### 8-9. train on HF Jobs

```bash
hf jobs uv run --flavor a100-large --timeout 2h --secrets HF_TOKEN src/train.py --stage pretrain --hf-user NAME
hf jobs uv run --flavor a100-large --timeout 30m --secrets HF_TOKEN src/train.py --stage closed --hf-user NAME
hf jobs uv run --flavor a100-large --timeout 30m --secrets HF_TOKEN src/train.py --stage open --hf-user NAME
```

### 10. evaluate and demo

```bash
python src/eval.py --model NAME/tinyllm-closed --data NAME/tinyllm-closed --split test
python src/eval.py --model NAME/tinyllm-closed --data NAME/tinyllm-closed --split train --limit 300
python src/eval.py --model NAME/tinyllm-open --data NAME/tinyllm-open --split test
python src/demo.py --hf-user NAME
python src/demo.py --hf-user NAME --progression
python src/app.py --hf-user NAME
```
