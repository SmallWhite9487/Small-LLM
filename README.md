# Small-LLM

Small-LLM is an educational project for building a compact, GPT-style language model with PyTorch. It includes a pretraining pipeline, optional instruction fine-tuning, a text-collection crawler, and a local web chat app. Text is tokenized with `tiktoken`; training uses CUDA when available and otherwise runs on the CPU.

## Getting started

Run commands from the project root. Install the dependencies for the workflow you plan to use:

```bash
python -m pip install torch tiktoken
```

For the crawler and web app, install their additional dependencies:

```bash
python -m pip install -r "Web Crawler/requirements.txt"
python -m pip install -r "Deployment/requirements.txt"
```

## Pretraining

Add UTF-8 `.txt` files to `raw_text/`. The loader searches this folder and its subfolders. Then start training:

```bash
python main.py
```

The script trains the model, asks whether to save a checkpoint as `gpt_checkpoint.pt`, and generates sample text from the prompt `One day,`. Training settings are defined near the bottom of `main.py`. The default run is substantial and may take a long time, especially on CPU.

## Optional workflows

### Collecting training text

The Wikisource crawler stages downloaded texts in `temp_raw_text/`. To enable its optional Gemini filtering, copy `.env.example` to `.env` and set `GEMINI_API_KEY` and `GEMINI_MODEL`. Run the crawler with:

```bash
python "Web Crawler/wikisource_crawler.py"
```

When you stop the crawler, it offers to filter the staged files. Filtered files are saved under `raw_text/`; if filtering is skipped, staged files remain in `temp_raw_text/` and can be moved to `raw_text/` manually.

### Instruction fine-tuning

After pretraining, prepare instruction examples in `SFT/instructions.jsonl` and run:

```bash
python "SFT/train_sft.py"
```

By default, this loads `gpt_checkpoint.pt` and saves the fine-tuned model as `SFT/sft_checkpoint.pt`.

### Local chat app

Start the Flask app with:

```bash
python "Deployment/app.py"
```

By default, it serves the chat UI at `http://127.0.0.1:5002` and loads `SFT/sft_checkpoint.pt`.

## Project files

- `main.py` — tokenizer, text loader, dataset and data loader, GPT model, training loop, checkpoint saving, and text generation.
- `raw_text/` — pretraining corpus; the crawler's filtered output is also saved here.
- `temp_raw_text/` — crawler staging area for downloaded texts.
- `SFT/train_sft.py` — instruction fine-tuning script.
- `SFT/instructions.jsonl` — instruction-tuning examples.
- `SFT/sft_checkpoint.pt` — fine-tuned model checkpoint.
- `Web Crawler/wikisource_crawler.py` — Wikisource crawler and optional Gemini-based text filtering.
- `Deployment/app.py` — local Flask chat interface.
- `gpt_checkpoint.pt` — pretrained model checkpoint used by the fine-tuning workflow.
- `.env.example` — example configuration for the crawler; keep API keys in `.env` and do not commit them.
