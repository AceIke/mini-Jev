import numpy as np

from minijev.text import Vocab, normalize_state, tokenize


def test_normalize_state_accepts_string_object_array():
    assert normalize_state("hello") == "hello"
    assert '"a": 1' in normalize_state({"a": 1})
    assert normalize_state([{"a": 1}]).startswith("[")


def test_tokenize_is_lowercase_and_splits_punctuation():
    assert tokenize("Charged twice, ASAP!") == ["charged", "twice", "asap"]
    assert tokenize("don't stop") == ["don't", "stop"]


def test_tokenize_handles_cjk_one_char_per_token():
    assert tokenize("退款") == ["退", "款"]


def test_vocab_known_and_unknown_ids():
    vocab = Vocab(["charge", "refund"], n_unk_buckets=16)
    assert vocab.size == 18
    assert vocab.encode("charge") == 16
    assert vocab.encode("refund") == 17
    # unknown tokens land inside the reserved bucket range
    unseen_a = vocab.encode("zzz")
    unseen_b = vocab.encode("zzz")
    assert 0 <= unseen_a < 16
    assert unseen_a == unseen_b  # stable hashing


def test_vocab_bag_counts_and_sums():
    vocab = Vocab(["charge", "refund"], n_unk_buckets=8)
    bag = vocab.bag(["charge", "charge", "refund", "not-in-vocab"])
    assert bag.shape == (vocab.size,)
    assert bag.sum() == 4
    assert bag[vocab.encode("charge")] == 2


def test_vocab_round_trip():
    vocab = Vocab(["a", "b", "c", "d"], n_unk_buckets=4)
    restored = Vocab.from_dict(vocab.to_dict())
    assert restored.size == vocab.size
    for token in ("a", "b", "c", "d", "e"):
        assert restored.encode(token) == vocab.encode(token)
