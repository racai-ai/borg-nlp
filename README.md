# borg-cube

## 1. Introduction

`borg-cube` is a Python NLP pipeline for tokenization, multiword-token
expansion, part-of-speech tagging, dependency parsing, and lemmatization.
Components can be used from a local model directory or loaded as a
Hugging Face pipeline.

## 2. Quick start guide

Install the package and its runtime dependencies:

```sh
python -m pip install borg-cube
```

### Local models

Load component folders from a local directory:

```python
from borg import cube

nlp = cube("en", model_path="/path/to/models/en")
sentences = nlp("The cat sat on the mat.")
```

The model directory contains one folder per component. Without `model_path`,
`cube("en")` looks under `~/.borg_cube/en/`.

### Hugging Face models

The base pipeline repo lists its components. By default, the pipeline loads
unsuffixed adapter repos such as `racai/borg-tokenizer`:

```python
from transformers import pipeline

nlp = pipeline(model="racai/borg", trust_remote_code=True)
result = nlp("The cat sat on the mat.")
```

For a named variant, pass `adapter_variant`. For example, `"combined"` loads
`racai/borg-tokenizer-combined`, `racai/borg-parser-combined`, and the other
listed component repos with the same suffix:

```python
nlp = pipeline(
    model="racai/borg",
    trust_remote_code=True,
    adapter_variant="combined",
)
result = nlp("The cat sat on the mat.")
```

Install `borg-cube` in the inference environment before loading the pipeline.
`trust_remote_code=True` runs code from the Hub repository; review it before
enabling this option. Runtime device selection uses the local hardware by
default; pass `device=-1` to force CPU.

Predictions use batches of up to `eval_batch_size` sentences (default: 32).
Reduce it if inference exceeds available memory. Inputs are truncated at
`max_seq_length`. Component `predict()` methods show progress bars by default;
pass `show_progress=False` to disable them.

## 3. Training your own model

### Install from GitHub

Install the latest source directly:

```sh
python -m pip install "git+https://github.com/tiberiu44/borg-cube.git"
```

For development, clone the repository and install it in editable mode:

```sh
git clone https://github.com/tiberiu44/borg-cube.git
cd borg-cube
python -m pip install -e .
```

### Train components

Train one component using CoNLL-U training and development files:

```sh
borg train --component tokenizer train.conllu dev.conllu data/models/mixed
```

Repeat with `mwt`, `tagger`, `parser`, or `lemmatizer` as needed. Training
saves component checkpoints under
`<model-directory>/<component>/{best,last}`. The CLI also accepts options such
as `--epochs`, `--batch-size`, `--device`, and `--lang`.

### Publish Hugging Face adapters and pipeline

Log in to Hugging Face first:

```sh
huggingface-cli login
```

Upload each component directory using the base repo name, component name, and
optional variant suffix. For example:

```sh
borg upload data/models/mixed/tokenizer/ racai/borg-tokenizer-combined
borg upload data/models/mixed/mwt/ racai/borg-mwt-combined
borg upload data/models/mixed/tagger/ racai/borg-tagger-combined
borg upload data/models/mixed/parser/ racai/borg-parser-combined
borg upload data/models/mixed/lemmatizer/ racai/borg-lemmatizer-combined
```

Then publish the base pipeline repo:

```sh
borg upload racai/borg --pipeline
```

This publishes the pipeline configuration and remote code, not adapter
weights. By default it lists all supported components; use
`--components tokenizer parser` to list only selected ones. Adapters may have
`borg_config.json` at the repository root; `best/` or `last/` subfolders are
not required.

Runtime configuration is loaded from the adapters. Training checkpoints save
`BorgConfig.dtype` as a string and restore it as a PyTorch dtype; if the local
device does not support BF16 autocast, inference warns and uses float32.

## 4. Paper to cite

Citation information will be added here.
