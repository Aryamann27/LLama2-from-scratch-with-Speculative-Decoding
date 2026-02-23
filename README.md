# LLaMA 2 From Scratch

An implementation of **LLaMA 2** in PyTorch—transformer architecture, training loop, inference, and **speculative decoding**—without relying on Hugging Face model code.

## Features

- **Model**: LLaMA 2–style decoder (RMSNorm, RoPE, SwiGLU FFN, grouped-query attention)
- **Training**: Training script on [RedPajama-Data-V2](https://huggingface.co/datasets/togethercomputer/RedPajama-Data-V2) sample with checkpointing and resume
- **Inference**: Load official LLaMA 2 checkpoints (from Meta) and run text completion
- **Speculative decoding**: Draft model (early layers) + verification for faster generation

## Requirements

- Python 3.10+
- PyTorch 2.x (CUDA or MPS recommended for speed)
- ~16GB+ RAM for 7B inference; GPU VRAM recommended

## Setup

```bash
# Clone the repo
git clone https://github.com/Aryamann27/LLama2-from-scratch-with-Speculative-Decoding.git
cd llama2_from_scratch

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## LLaMA 2 Weights

This repo does **not** include LLaMA 2 weights. You must obtain them from Meta:

1. Accept the [LLaMA 2 license](https://ai.meta.com/resources/models-and-libraries/llama-downloads/) and get the presigned download URL from Meta.
2. Run the download script (it will prompt for the URL and model size):

```bash
bash download.sh
```

By default, weights are written to the current directory (e.g. `llama-2-7b/`, `llama-2-7b-chat/`). The tokenizer is saved as `tokenizer.model` in the same folder as the model (e.g. `llama-2-7b/tokenizer.model`).

**Note:** Use of LLaMA 2 weights is subject to Meta’s [LLaMA 2 Community License](https://ai.meta.com/resources/models-and-libraries/llama-downloads/).

## Usage

### Inference (with downloaded 7B weights)

Standard completion:

```bash
python inference.py
```

By default this loads from `llama-2-7b/` and `llama-2-7b/tokenizer.model`. Edit the `checkpoints_dir` and `tokenizer_path` in `inference.py` if your paths differ.

### Speculative decoding

The script uses speculative decoding by default (draft from early layers, verify with full model). You can switch to standard decoding by changing the call to `model.text_completion(...)` in the `if __name__ == '__main__'` block.

### Benchmark: standard vs speculative

```bash
python benchmark_inference.py
```

Uses the same paths as above (`llama-2-7b/`, `llama-2-7b/tokenizer.model`). Adjust inside the script if needed.

### Training

Training uses the RedPajama-Data-V2 sample and the LLaMA 2 tokenizer. Set the tokenizer path in `train.py` (default: `llama-2-7b/tokenizer.model`). Checkpoints are saved under `./checkpoints_train/`.

```bash
python train.py
```

Hyperparameters (batch size, sequence length, learning rate, etc.) are defined at the top of `train.py`.

## Project layout

| File / folder        | Description                                  |
|----------------------|----------------------------------------------|
| `model.py`           | LLaMA 2–style `Transformer`, attention, RoPE |
| `inference.py`       | Load weights, run completion + speculative  |
| `speculative.py`     | Speculative decoding (draft + verify)       |
| `train.py`           | Training loop, RedPajama, checkpointing      |
| `benchmark_inference.py` | Compare standard vs speculative latency  |
| `download.sh`        | Download LLaMA 2 weights from Meta (URL)      |
