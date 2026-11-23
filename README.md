# borg-cube
Repository for the borg-cube nlp processing framework

## Checkpoints

Training saves the latest and best models in the `last/` and `best/`
subdirectories of the component's model directory. Checkpoint configuration
stores `BorgConfig.dtype` as a string (for example, `"torch.bfloat16"`) and
restores it to a PyTorch dtype when loading. Checkpoints without a dtype field
use the default `torch.bfloat16`; invalid dtype names raise an error.

`BorgConfig.dtype` controls mixed-precision computation during training and
inference. Pretrained encoder weights are loaded in their standard dtype, then
the configured autocast dtype is used on CPU, CUDA/ROCm, and MPS backends. If
the runtime device cannot execute bfloat16 autocast, the run warns and uses
float32 without changing the saved `BorgConfig.dtype`.

## Inference

All four models use `BorgConfig.eval_batch_size` (default: 32) for prediction.
The tagger, parser, and lemmatizer group sentences by estimated length, tokenize
them in batches, and right-pad only to the longest sequence in each batch.
Returned sentences remain in input order, and empty sentences are preserved.
The tokenizer batches overlapping windows without changing whitespace
reconstruction or overlap scoring.

Each model's `predict()` displays a tqdm progress bar measured in batches.
Pass `show_progress=False` to disable it, for example
`tagger.predict(sentences, show_progress=False)` or
`tokenizer.predict(text, show_progress=False)`. Empty input produces no bar.

Prediction uses PyTorch inference mode and the configured autocast dtype.
The parser scores dependency relations only for decoded arcs instead of
materializing a relation tensor for every subword pair. Training is unchanged.
Reduce `eval_batch_size` if inference exceeds available device memory; set it
to 1 for single-example inference. It must be positive.

Sentence models still truncate at `max_seq_length`; batching does not add
long-sentence recovery. Annotation preserves token metadata, including
`space_after`, multiword tokens, and empty nodes.

## Hugging Face pipeline

Export a Hugging Face Hub pipeline bundle that references separately uploaded
adapters.
The packaged template at `src/pipeline/hf/config.json` is populated by the
exporter into the Hub folder's root `config.json`, including the custom task,
remote-code references, and component names. The export includes the
model-loading marker weights and runtime source, but not component checkpoints;
runtime configuration is read from the downloaded adapters unless explicitly
provided:

```python
from src.pipeline.hf import export_hf_repo

export_hf_repo(
    output_dir="./your-nlp-repo",
    checkpoint="best",
)
```

Upload each component directory to a repository named by appending its component
name to the base repo ID. For example:

```sh
borg upload data/models/mixed/tokenizer/ racai/borg-tokenizer-combined
borg upload data/models/mixed/mwt/ racai/borg-mwt-combined
borg upload data/models/mixed/tagger/ racai/borg-tagger-combined
borg upload data/models/mixed/parser/ racai/borg-parser-combined
borg upload data/models/mixed/lemmatizer/ racai/borg-lemmatizer-combined
```

For a named variant, append it after the component name, for example
`racai/borg-tokenizer-combined`.
The uploaded component may have its `borg_config.json` at the repository root;
no `best` or `last` subfolder is required.

Then publish the base pipeline repo. This does not require local model folders
or upload adapter weights; by default it lists all supported components:

```sh
borg upload racai/borg --pipeline
```

Log in first with `huggingface-cli login` (or set `HF_TOKEN`). Omit
`adapter_variant` to load default repos such as `racai/borg-tokenizer`, or set
it to select a variant such as `racai/borg-tokenizer-combined`:

```python
from transformers import pipeline

nlp = pipeline(model="racai/borg", trust_remote_code=True)
combined_nlp = pipeline(
    model="racai/borg",
    trust_remote_code=True,
    adapter_variant="combined",
)
result = nlp("Your input text here")
```

Install the base repo's `requirements.txt` in the inference environment first.
The call returns a list of sentence dictionaries, each containing comments and
token dictionaries. A list of input strings returns one such sentence list per
input. `trust_remote_code=True` executes the pipeline/model code stored in the
Hub repository; review it before enabling this option.
