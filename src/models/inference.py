"""Shared, right-padded batches for sentence-level inference."""
from __future__ import annotations

from bisect import bisect_left, bisect_right
from dataclasses import dataclass
from typing import Dict, Iterator, List, Optional, Set, Tuple

import torch
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm

from src.data.conllu import Sentence, Token
from src.models.base import BorgBaseModel


@dataclass
class SentenceBatch:
    indices: List[int]
    tokens: List[List[Token]]
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    word_ids: List[List[int | None]]
    word_positions: List[Dict[int, int]]
    core_word_ids: List[Set[int]]
    windowed_rows: List[bool]
    input_char_ids: torch.Tensor | None = None


@dataclass
class _Window:
    sentence_index: int
    tokens: List[Token]
    word_offset: int
    core_word_ids: Optional[Set[int]]
    subword_count: int
    windowed: bool = False


def _window_ranges(
    word_piece_counts: List[int],
    content_limit: int,
) -> List[Tuple[int, int, int, int]]:
    if not word_piece_counts:
        return []
    if content_limit < 1:
        raise ValueError("max_seq_length must leave room for input tokens")
    if any(count > content_limit for count in word_piece_counts):
        raise ValueError("A single word exceeds max_seq_length")

    prefix = [0]
    for count in word_piece_counts:
        prefix.append(prefix[-1] + count)

    ranges = []
    start = 0
    overlap = content_limit // 4
    while start < len(word_piece_counts):
        end = bisect_right(
            prefix, prefix[start] + content_limit, lo=start + 1
        ) - 1
        if end <= start:
            raise ValueError("Unable to fit a word within max_seq_length")
        ranges.append((start, end))
        if end == len(word_piece_counts):
            break

        if overlap:
            next_start = bisect_left(
                prefix, prefix[end] - overlap, lo=start + 1, hi=end + 1
            )
            if next_start >= end:
                next_start = end - 1
            start = max(start + 1, next_start)
        else:
            start = end

    boundaries = [0]
    for (_, previous_end), (next_start, _) in zip(ranges, ranges[1:]):
        midpoint = (prefix[previous_end] + prefix[next_start]) / 2
        boundary = min(
            range(next_start, previous_end + 1),
            key=lambda index: abs(prefix[index] - midpoint),
        )
        boundaries.append(boundary)
    boundaries.append(len(word_piece_counts))

    return [
        (start, end, boundaries[index], boundaries[index + 1])
        for index, (start, end) in enumerate(ranges)
    ]


def _window_sentence(
    model: BorgBaseModel,
    sentence_index: int,
    tokens: List[Token],
) -> List[_Window]:
    encoding = model.hf_tokenizer(
        [[token.form for token in tokens]],
        is_split_into_words=True,
        truncation=False,
        padding=False,
    )
    word_piece_counts = [0] * len(tokens)
    for word_id in encoding.word_ids(batch_index=0):
        if word_id is not None:
            word_piece_counts[word_id] += 1
    if any(count == 0 for count in word_piece_counts):
        raise ValueError(f"Tokenizer omitted words in sentence {sentence_index}")

    special_tokens = model.hf_tokenizer.num_special_tokens_to_add(pair=False)
    content_limit = model.config.max_seq_length - special_tokens
    ranges = _window_ranges(word_piece_counts, content_limit)
    return [
        _Window(
            sentence_index=sentence_index,
            tokens=tokens[start:end],
            word_offset=start,
            core_word_ids=set(range(core_start, core_end)),
            subword_count=sum(word_piece_counts[start:end]),
            windowed=True,
        )
        for start, end, core_start, core_end in ranges
    ]


def _batch_from_encoding(
    model: BorgBaseModel,
    windows: List[_Window],
    encoding,
    *,
    check_window_length: bool = False,
) -> SentenceBatch:
    word_ids = [
        encoding.word_ids(batch_index=index)
        for index in range(len(windows))
    ]
    word_positions = []
    core_word_ids = []
    for window, ids in zip(windows, word_ids):
        if check_window_length and len(encoding["input_ids"][len(word_positions)]) > model.config.max_seq_length:
            raise RuntimeError(
                f"Window for sentence {window.sentence_index} exceeds "
                f"max_seq_length={model.config.max_seq_length}"
            )
        positions: Dict[int, int] = {}
        for position, word_id in enumerate(ids):
            if word_id is not None:
                positions.setdefault(window.word_offset + word_id, position)
        word_positions.append(positions)

        core = window.core_word_ids
        if core is None:
            core = set(positions)
        if not core.issubset(positions):
            raise RuntimeError(
                f"Window for sentence {window.sentence_index} omitted "
                "core words during tokenization"
            )
        core_word_ids.append(core)

    input_char_ids = None
    if model.component in ("lemmatizer", "mwt"):
        vocab = model.input_vocab
        pad_id = vocab["<PAD>"]
        char_sequences = []
        max_char_length = max(
            len(token.form) + 2
            for window in windows
            for token in window.tokens
        )
        for window, ids in zip(windows, word_ids):
            seen_words = set()
            char_rows = []
            for word_id in ids:
                if word_id is None:
                    continue
                if word_id not in seen_words:
                    token = window.tokens[word_id]
                    char_ids = [vocab["<SOS>"]]
                    char_ids.extend(
                        vocab.get(char, vocab["<UNK>"])
                        for char in token.form
                    )
                    char_ids.append(vocab["<EOS>"])
                    seen_words.add(word_id)
                else:
                    char_ids = [pad_id]
                char_rows.append(
                    char_ids + [pad_id] * (max_char_length - len(char_ids))
                )
            char_sequences.append(torch.tensor(char_rows, dtype=torch.long))
        input_char_ids = pad_sequence(
            char_sequences,
            batch_first=True,
            padding_value=pad_id,
        )

    return SentenceBatch(
        indices=[window.sentence_index for window in windows],
        tokens=[window.tokens for window in windows],
        input_ids=pad_sequence(
            [torch.tensor(ids, dtype=torch.long) for ids in encoding["input_ids"]],
            batch_first=True,
            padding_value=model.hf_tokenizer.pad_token_id,
        ),
        attention_mask=pad_sequence(
            [torch.tensor(mask, dtype=torch.long) for mask in encoding["attention_mask"]],
            batch_first=True,
            padding_value=0,
        ),
        word_ids=word_ids,
        word_positions=word_positions,
        core_word_ids=core_word_ids,
        windowed_rows=[window.windowed for window in windows],
        input_char_ids=input_char_ids,
    )


def sentence_batches(
    model: BorgBaseModel,
    sentences: List[Sentence],
    *,
    show_progress: bool = False,
    window_long_sentences: bool = False,
) -> Iterator[SentenceBatch]:
    batch_size = model.config.eval_batch_size
    if batch_size < 1:
        raise ValueError("eval_batch_size must be positive")

    token_getter = (
        Sentence.surface_tokens
        if model.component == "mwt"
        else Sentence.regular_tokens
    )
    examples = [
        (index, token_getter(sentence))
        for index, sentence in enumerate(sentences)
    ]
    examples = sorted(
        (example for example in examples if example[1]),
        key=lambda example: sum(len(token.form) for token in example[1]),
    )
    for start in tqdm(
        range(0, len(examples), batch_size),
        desc=f"[{type(model).__name__}] Predict",
        unit="batch",
        disable=not show_progress or not examples,
    ):
        chunk = examples[start:start + batch_size]
        encoding = model.hf_tokenizer(
            [[token.form for token in tokens] for _, tokens in chunk],
            is_split_into_words=True,
            max_length=model.config.max_seq_length,
            truncation=True,
            padding=False,
        )
        initial_word_ids = [
            encoding.word_ids(batch_index=index)
            for index in range(len(chunk))
        ]
        truncated_rows = []
        initial_windows = []
        for row, ((sentence_index, tokens), ids) in enumerate(zip(chunk, initial_word_ids)):
            covered = {word_id for word_id in ids if word_id is not None}
            truncated = covered != set(range(len(tokens)))
            if truncated:
                truncated_rows.append(row)
            initial_windows.append(
                _Window(
                    sentence_index=sentence_index,
                    tokens=tokens,
                    word_offset=0,
                    core_word_ids=(
                        set() if truncated else set(range(len(tokens)))
                    ),
                    subword_count=0,
                )
            )

        yield _batch_from_encoding(model, initial_windows, encoding)

        if not window_long_sentences or not truncated_rows:
            continue

        windows = []
        for row in truncated_rows:
            sentence_index, tokens = chunk[row]
            windows.extend(_window_sentence(model, sentence_index, tokens))
        windows.sort(key=lambda window: window.subword_count)

        for window_start in range(0, len(windows), batch_size):
            window_chunk = windows[window_start:window_start + batch_size]
            window_encoding = model.hf_tokenizer(
                [
                    [token.form for token in window.tokens]
                    for window in window_chunk
                ],
                is_split_into_words=True,
                max_length=model.config.max_seq_length,
                truncation=False,
                padding=False,
            )
            yield _batch_from_encoding(
                model,
                window_chunk,
                window_encoding,
                check_window_length=True,
            )
