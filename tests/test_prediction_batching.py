"""Offline prediction regressions and measurable batching checks."""
from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch

from src.config import BorgConfig
from src.data.conllu import Sentence, Token
from src.models.inference import sentence_batches
from src.models.lemmatizer import (
    LemmatizerModel,
    _prepare_training_sentences,
)
from src.models.parser import ParserModel, greedy_decode
from src.models.tagger import TaggerModel
from src.models.tokenizer import TokenizerModel


class Encoding(dict):
    def __init__(self, input_ids, word_ids):
        super().__init__(
            input_ids=input_ids,
            attention_mask=[[1] * len(ids) for ids in input_ids],
        )
        self.ids = word_ids

    def word_ids(self, batch_index=0):
        return self.ids[batch_index]


class WordTokenizer:
    pad_token_id = 0
    padding_side = "left"

    def __call__(self, forms, max_length=None, **kwargs):
        sequences, mappings = [], []
        if forms and isinstance(forms[0], str):
            for word in forms:
                token_id = sum(map(ord, word)) % 30 + 3
                pieces = [token_id, token_id + 1] if len(word) > 2 else [token_id]
                sequences.append([1, *pieces, 2])
                mappings.append([None, *([0] * len(pieces)), None])
            return Encoding(sequences, mappings)
        for words in forms:
            ids, word_ids = [], []
            for wid, word in enumerate(words):
                token_id = sum(map(ord, word)) % 30 + 3
                pieces = [token_id, token_id + 1] if len(word) > 2 else [token_id]
                ids.extend(pieces)
                word_ids.extend([wid] * len(pieces))
            if kwargs.get("truncation", True) and max_length is not None:
                ids = ids[:max_length - 2]
                word_ids = word_ids[:max_length - 2]
            sequences.append([1, *ids, 2])
            mappings.append([None, *word_ids, None])
        return Encoding(sequences, mappings)

    def num_special_tokens_to_add(self, pair=False):
        return 2


def make_model(model_type, batch_size):
    torch.manual_seed(42)
    model = model_type.__new__(model_type)
    torch.nn.Module.__init__(model)
    model.component = {
        TaggerModel: "tagger",
        LemmatizerModel: "lemmatizer",
        ParserModel: "parser",
    }[model_type]
    model.config = BorgConfig(device="cpu", max_seq_length=10, eval_batch_size=batch_size)
    model.hf_tokenizer = Mock(wraps=WordTokenizer())
    model.hf_tokenizer.pad_token_id = 0
    model.hidden_size = 4
    if model_type is TaggerModel:
        model.upos_vocab = {"<PAD>": 0, "<UNK>": 1, "NOUN": 2}
        model.xpos_vocab = {"<PAD>": 0, "<UNK>": 1, "NN": 2}
        model.feats_vocab = {"<PAD>": 0, "<UNK>": 1, "Number=Sing": 2}
    elif model_type is LemmatizerModel:
        model.upos_vocab = {"<PAD>": 0, "<UNK>": 1, "NOUN": 2}
        model.input_vocab = {"<PAD>": 0, "<UNK>": 1, "<SOS>": 2, "<EOS>": 3, "L": 4}
        model.output_vocab = {"<PAD>": 0, "<UNK>": 1, "<SOS>": 2, "<EOS>": 3, "l": 4}
        model.encoder = SimpleNamespace(config=SimpleNamespace(hidden_size=4))
    else:
        model.deprel_vocab = {"<PAD>": 0, "<UNK>": 1, "root": 2, "dep": 3}
        model.ARC_HIDDEN = 8
        model.REL_HIDDEN = 4
    model._build_heads()

    def encode(input_ids, attention_mask):
        assert torch.is_inference_mode_enabled()
        ids = input_ids.float()
        return torch.stack([ids / 30, ids.sin(), ids.cos(), ids / 60], dim=-1)

    model.encode = Mock(side_effect=encode)
    return model


def sentences():
    return [
        Sentence(
            comments=["# long"],
            tokens=[
                Token(id=1, form="Long", upos="NOUN", space_after=False, misc="X=Y"),
                Token(id="1-2", form="Longword"),
                Token(id=2, form="word", upos="UNKNOWN"),
                Token(id="2.1", form="empty"),
                *[Token(id=i, form="more", upos="NOUN") for i in range(3, 8)],
            ],
        ),
        Sentence(comments=["# empty"]),
        Sentence(tokens=[Token(id=1, form="A", upos="NOUN")]),
        Sentence(tokens=[Token(id=1, form="B"), Token(id=2, form="C")]),
        Sentence(tokens=[Token(id="1.1", form="node")]),
        Sentence(tokens=[Token(id=1, form="Last", upos="NOUN")]),
    ]


@pytest.mark.parametrize("model_type", [TaggerModel, LemmatizerModel, ParserModel])
def test_batched_predictions_match_single_and_preserve_metadata(model_type):
    inputs = sentences()
    original = deepcopy(inputs)
    single = make_model(model_type, 1)
    batched = make_model(model_type, 3)
    expected = single.predict(inputs)
    actual = batched.predict(inputs)
    assert actual == expected
    assert inputs == original
    assert actual[1] is inputs[1]
    assert actual[4] is inputs[4]
    assert actual[0].tokens[0].space_after is False
    assert actual[0].tokens[1] is inputs[0].tokens[1]
    assert actual[0].tokens[3] is inputs[0].tokens[3]
    if model_type is TaggerModel:
        assert all(
            (token.upos, token.xpos, token.feats) != ("_", "_", "_")
            for token in actual[0].regular_tokens()
        )
    elif model_type is LemmatizerModel:
        assert all(token.lemma != "_" for token in actual[0].regular_tokens())
    else:
        parsed_tokens = actual[0].regular_tokens()
        assert all(token.head is not None and token.deprel != "_" for token in parsed_tokens)
        assert sum(token.head == 0 for token in parsed_tokens) == 1


def test_long_sentence_windows_cover_every_word_once_with_overlap():
    model = make_model(TaggerModel, 3)
    sentence = Sentence(
        tokens=[Token(id=index, form="more") for index in range(1, 13)]
    )

    batches = list(
        sentence_batches(model, [sentence], window_long_sentences=True)
    )
    ownership = [
        word_id
        for batch in batches
        for core_ids in batch.core_word_ids
        for word_id in core_ids
    ]
    window_word_count = sum(
        len(batch.word_positions[row])
        for batch in batches
        for row in range(len(batch.indices))
        if batch.core_word_ids[row]
    )

    assert sorted(ownership) == list(range(12))
    assert window_word_count > len(sentence.regular_tokens())
    assert all(
        len(batch.input_ids[row]) <= model.config.max_seq_length
        for batch in batches
        for row in range(len(batch.indices))
    )


@pytest.mark.parametrize("model_type", [TaggerModel, LemmatizerModel, ParserModel])
def test_short_sentence_keeps_single_existing_inference_batch(model_type):
    model = make_model(model_type, 3)
    sentence = Sentence(tokens=[
        Token(id=1, form="A"),
        Token(id=2, form="B"),
    ])

    model.predict([sentence])

    assert model.hf_tokenizer.call_count == 1
    assert model.encode.call_count == 1


def test_parser_selected_relation_scores_match_dense_forward():
    model = make_model(ParserModel, 3)
    inputs = [
        Sentence(tokens=[
            Token(id=1, form="A"),
            Token(id=2, form="B"),
        ]),
        Sentence(tokens=[
            Token(id=1, form="C"),
        ]),
    ]
    expected = {}
    with torch.inference_mode():
        for batch in sentence_batches(model, inputs):
            arcs, relations = model(batch.input_ids, batch.attention_mask)
            for row, index in enumerate(batch.indices):
                positions = batch.word_positions[row]
                heads = greedy_decode(arcs[row], positions)
                inverse = {pos: wid for wid, pos in positions.items()}
                expected[index] = {
                    wid: (
                        0 if head == 0 else inverse[head] + 1,
                        int(relations[row, positions[wid], head].argmax(-1)),
                    )
                    for wid, head in heads.items()
                }
    shapes = []
    hook = model.rel_biaffine.register_forward_pre_hook(
        lambda _, args: shapes.append(args[0].shape)
    )
    actual = model.predict(inputs)
    hook.remove()
    inverse_relations = {value: key for key, value in model.deprel_vocab.items()}
    for index, predictions in expected.items():
        for wid, (head, relation) in predictions.items():
            token = actual[index].regular_tokens()[wid]
            assert (token.head, token.deprel) == (head, inverse_relations[relation])
    assert all(shape[1] == 1 for shape in shapes)


def test_tagger_uses_first_subword_predictions():
    model = make_model(TaggerModel, 3)
    inputs = [Sentence(tokens=[
        Token(id=1, form="Long"),
        Token(id=2, form="word"),
    ])]

    def forward(input_ids, attention_mask):
        assert torch.is_inference_mode_enabled()
        logits = torch.zeros((*input_ids.shape, 3))
        ids = input_ids.remainder(3)
        logits.scatter_(-1, ids.unsqueeze(-1), 10)
        return logits, logits, logits

    model.forward = Mock(side_effect=forward)
    actual = model.predict(inputs)
    inverse = {value: key for key, value in model.upos_vocab.items()}
    for batch in sentence_batches(model, inputs):
        for row, index in enumerate(batch.indices):
            for wid, pos in batch.word_positions[row].items():
                expected = inverse[int(batch.input_ids[row, pos] % 3)]
                assert actual[index].regular_tokens()[wid].upos == expected
    assert model.forward.call_count == 1


def test_lemmatizer_predicts_in_token_batches_and_preserves_alignment():
    model = make_model(LemmatizerModel, 3)
    inputs = sentences()
    observed_context_batches = []

    def predict_word_batch(hidden_states, upos_ids, input_char_ids):
        assert hidden_states.shape[0] <= 32
        assert input_char_ids.shape[0] == hidden_states.shape[0]
        observed_context_batches.append(hidden_states.shape)
        return [f"lemma-{upos}" for upos in upos_ids.tolist()]

    model._predict_word_batch = Mock(side_effect=predict_word_batch)
    actual = model.predict(inputs)
    assert actual[0].tokens[0].lemma == "lemma-2"
    assert actual[0].tokens[2].lemma == "lemma-1"
    assert actual[2].tokens[0].lemma == "lemma-2"
    assert actual[0].tokens[1] is inputs[0].tokens[1]
    assert actual[0].tokens[3] is inputs[0].tokens[3]
    assert model._predict_word_batch.call_count > 0
    assert all(shape[0] <= 32 for shape in observed_context_batches)


def test_lemmatizer_batches_words_in_groups_of_at_most_32():
    model = make_model(LemmatizerModel, 64)
    sentence = Sentence(
        tokens=[Token(id=index, form="more", upos="NOUN") for index in range(1, 41)]
    )
    observed_batch_sizes = []

    def predict_word_batch(hidden_states, upos_ids, input_char_ids):
        observed_batch_sizes.append(hidden_states.shape[0])
        assert input_char_ids.shape[0] == hidden_states.shape[0]
        return ["lemma"] * hidden_states.shape[0]

    model._predict_word_batch = Mock(side_effect=predict_word_batch)
    actual = model.predict([sentence])

    assert sum(observed_batch_sizes) == 40
    assert max(observed_batch_sizes) <= 32
    assert all(token.lemma == "lemma" for token in actual[0].regular_tokens())
    assert model.encode.call_count > 1


def test_lemmatizer_truncates_long_runtime_forms_and_restores_originals():
    model = make_model(LemmatizerModel, 2)
    long_form = "x" * 63
    inputs = [Sentence(tokens=[
        Token(id=1, form="short", lemma="old"),
        Token(id=2, form=long_form, lemma="old-long"),
    ])]
    original = deepcopy(inputs)

    def predict_word_batch(hidden_states, upos_ids, input_char_ids):
        assert input_char_ids.shape[-1] <= 52
        return ["predicted"] * hidden_states.shape[0]

    model._predict_word_batch = Mock(side_effect=predict_word_batch)
    actual = model.predict(inputs)

    assert actual[0].tokens[0].lemma == "predicted"
    assert actual[0].tokens[1].form == long_form
    assert actual[0].tokens[1].lemma == long_form
    assert model.hf_tokenizer.call_args.args[0] == [["short", long_form[:50]]]
    assert inputs == original


def test_lemmatizer_training_truncates_words_lemmas_and_sentence_length():
    long_form = "f" * 60
    long_lemma = "l" * 60
    sentence = Sentence(
        comments=["# training"],
        tokens=[
            Token(id=index, form=long_form if index == 1 else "word",
                  lemma=long_lemma if index == 1 else "lemma")
            for index in range(1, 131)
        ],
    )

    prepared = _prepare_training_sentences([sentence])

    assert len(prepared[0].regular_tokens()) == 128
    assert prepared[0].tokens[0].form == long_form[:50]
    assert prepared[0].tokens[0].lemma == long_lemma[:50]
    assert prepared[0].comments == sentence.comments
    assert len(sentence.regular_tokens()) == 130
    assert sentence.tokens[0].form == long_form
    assert sentence.tokens[0].lemma == long_lemma


def test_lemmatizer_original_decoder_shapes():
    model = make_model(LemmatizerModel, 2)
    decoder_inputs = []
    hook = model.output_decoder.register_forward_pre_hook(
        lambda _, args: decoder_inputs.append(args[0].detach().clone())
    )

    with torch.inference_mode():
        result = model(
            torch.tensor([[1, 2, 3, 4], [1, 3, 2, 4]]),
            torch.ones((2, 4), dtype=torch.long),
            torch.tensor([[0, 2, 2, 0], [0, 2, 2, 0]]),
            torch.tensor([[[2, 4, 3], [2, 3, 0]], [[2, 4, 3], [2, 3, 0]]]),
            torch.tensor([[[2, 4, 3], [2, 3, -100]], [[2, 4, 3], [2, 3, -100]]]),
        )
    hook.remove()

    assert torch.count_nonzero(decoder_inputs[0]) == 0
    assert result["logits"].shape == (4, 3, len(model.output_vocab))


def test_lemmatizer_lstm_runs_outside_bfloat16_autocast():
    model = make_model(LemmatizerModel, 1)
    autocast_states = []
    hooks = [
        lstm.register_forward_pre_hook(
            lambda _, __: autocast_states.append(torch.is_autocast_enabled("cpu"))
        )
        for lstm in (model.input_encoder, model.output_decoder)
    ]

    with patch("src.config._supports_bfloat16", return_value=True):
        with model.config.autocast_context("cpu"):
            model(
                torch.tensor([[1, 2, 3, 4]]),
                torch.ones((1, 4), dtype=torch.long),
                torch.tensor([[0, 2, 2, 0]]),
                torch.tensor([[[2, 4, 3], [2, 3, 0]]]),
                max_output_length=1,
            )

    for hook in hooks:
        hook.remove()

    assert autocast_states == [False, False]


@pytest.mark.parametrize("model_type", [TaggerModel, LemmatizerModel, ParserModel])
def test_empty_input_and_invalid_batch_size(model_type):
    model = make_model(model_type, 3)
    assert model.predict([]) == []
    assert model.encode.call_count == 0
    model.config.eval_batch_size = 0
    with pytest.raises(ValueError, match="eval_batch_size"):
        model.predict(sentences())


class WindowTokenizer:
    pad_token_id = 0

    def __call__(self, text, **kwargs):
        return {
            "input_ids": [ord(char) for char in text],
            "offset_mapping": [(i, i + 1) for i in range(len(text))],
        }

    def num_special_tokens_to_add(self, pair=False):
        return 2

    def prepare_for_model(self, token_ids, **kwargs):
        ids = [1, *token_ids, 2]
        return {"input_ids": ids, "attention_mask": [1] * len(ids)}

    def get_special_tokens_mask(self, token_ids, **kwargs):
        return [1, *[0 for _ in token_ids], 1]


def make_tokenizer(batch_size):
    model = TokenizerModel.__new__(TokenizerModel)
    torch.nn.Module.__init__(model)
    model.config = BorgConfig(device="cpu", max_seq_length=6, eval_batch_size=batch_size)
    model.encoder = torch.nn.Module()
    model.encoder.config = model.config
    model.encoder.config.max_position_embeddings = 6
    model.hf_tokenizer = WindowTokenizer()

    def forward(ids, mask):
        assert torch.is_inference_mode_enabled()
        logits = torch.zeros((*ids.shape, 3))
        # Depend on window position to exercise overlap score accumulation.
        logits[..., 0] = 1
        logits[:, 1, 2] = 2
        logits[..., 1] = (ids % 3 == 0).float() * 3
        return logits, torch.zeros((*ids.shape, 2))

    model.forward = Mock(side_effect=forward)
    return model


def test_tokenizer_batches_windows_and_preserves_overlap_predictions():
    single, batched = make_tokenizer(1), make_tokenizer(3)
    text = "A B C D E F G H I J"
    assert batched.predict(text) == single.predict(text)
    assert single.forward.call_count == 7
    assert batched.forward.call_count == 3
    assert batched.forward.call_args_list[0].args[0].shape == (3, 6)
    assert batched.predict("") == []
    batched.config.eval_batch_size = 0
    with pytest.raises(ValueError, match="eval_batch_size"):
        batched.predict(text)


def test_tokenizer_limits_mps_prediction_to_one_window(monkeypatch):
    model = make_tokenizer(3)
    model.config.resolve_device = lambda: "mps"
    model.to = Mock(return_value=model)
    monkeypatch.setattr(torch.Tensor, "to", lambda tensor, *args, **kwargs: tensor)

    model.predict("A B C D E F G H I J", show_progress=False)

    assert model.forward.call_count == 7
    assert all(call.args[0].size(0) == 1 for call in model.forward.call_args_list)


@pytest.mark.parametrize("model_type", [TaggerModel, LemmatizerModel, ParserModel])
def test_sentence_prediction_progress_is_batch_based_and_optional(monkeypatch, model_type):
    progress = Mock(side_effect=lambda iterable, **kwargs: iterable)
    monkeypatch.setattr("src.models.inference.tqdm", progress)
    model = make_model(model_type, 3)
    inputs = sentences()
    visible = model.predict(inputs)
    assert len(progress.call_args.args[0]) == 2
    assert progress.call_args.kwargs["unit"] == "batch"
    assert progress.call_args.kwargs["disable"] is False
    assert progress.call_args.kwargs["desc"] == f"[{model_type.__name__}] Predict"
    assert model.predict(inputs, show_progress=False) == visible
    assert progress.call_args.kwargs["disable"] is True
    assert model.predict([]) == []
    assert progress.call_args.kwargs["disable"] is True


def test_tokenizer_prediction_progress_is_batch_based_and_optional(monkeypatch):
    progress = Mock(side_effect=lambda iterable, **kwargs: iterable)
    monkeypatch.setattr("src.models.tokenizer.tqdm", progress)
    model = make_tokenizer(3)
    text = "A B C D E F G H I J"
    visible = model.predict(text)
    assert len(progress.call_args.args[0]) == 3
    assert progress.call_args.kwargs["unit"] == "batch"
    assert progress.call_args.kwargs["disable"] is False
    assert model.predict(text, show_progress=False) == visible
    assert progress.call_args.kwargs["disable"] is True
    assert model.predict("") == []
    assert progress.call_args.kwargs["disable"] is True
