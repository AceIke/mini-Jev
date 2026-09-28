"""State serialization, tokenization, and the vocabulary.

mini-jev is a bag-of-words model, so this module is where the "text front end"
lives. Token order is deliberately ignored: that keeps the synthetic Bayes
posterior exactly computable (see ``tasks.py``) at the cost of throwing away
word order. Replacing this encoder with a small transformer is the obvious
"level 2" exercise; see ``docs/DESIGN.md``.
"""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any, Iterable, Sequence

import numpy as np

# Lowercase words (optionally with an internal apostrophe) or a single CJK
# character. Jev's docs call out CJK as accepted-but-weaker; we keep the same
# behaviour of "it tokenizes, but we make no accuracy promise".
_TOKEN_RE = re.compile(r"[a-z0-9]+(?:'[a-z]+)?|[\u3400-\u4dbf\u4e00-\u9fff]")


def normalize_state(state: Any) -> str:
    """Render any accepted *state* value as text.

    Jev accepts a string, a JSON object, or an array of text values. Objects and
    arrays are canonicalised to pretty-printed JSON so that key names stay
    visible to the model.
    """

    if isinstance(state, str):
        return state
    if isinstance(state, (bytes, bytearray)):
        return state.decode("utf-8", errors="replace")
    return json.dumps(state, ensure_ascii=False, indent=2, default=str)


def normalize_instruction(value: Any) -> str:
    """Render ``instructions`` / ``criteria`` values, which may be str/obj/array."""

    if value is None:
        return ""
    return normalize_state(value)


def tokenize(text: str) -> list[str]:
    """Split text into lowercase tokens. Order is preserved here, then discarded."""

    return _TOKEN_RE.findall(text.lower())


class Vocab:
    """Token -> id map with hashed buckets for out-of-vocabulary tokens.

    Ids ``[0, n_unk_buckets)`` are reserved for unknown tokens, so a word the
    model never saw during training still lands somewhere stable (the hashing
    trick) instead of blowing up the embedding table.
    """

    def __init__(self, tokens: Iterable[str] = (), n_unk_buckets: int = 32):
        self.n_unk_buckets = int(n_unk_buckets)
        known: dict[str, int] = {}
        for tok in tokens:
            if tok not in known:
                known[tok] = self.n_unk_buckets + len(known)
        self.known = known

    # -- lookup ----------------------------------------------------------
    @property
    def size(self) -> int:
        return self.n_unk_buckets + len(self.known)

    def encode(self, token: str) -> int:
        idx = self.known.get(token)
        if idx is not None:
            return idx
        digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
        return int.from_bytes(digest, "big") % self.n_unk_buckets

    def bag(self, tokens: Sequence[str]) -> np.ndarray:
        """Bag-of-words count vector (float32) of length ``size``."""

        out = np.zeros(self.size, dtype=np.float64)
        for tok in tokens:
            out[self.encode(tok)] += 1.0
        return out

    # -- persistence -----------------------------------------------------
    def to_dict(self) -> dict:
        return {"n_unk_buckets": self.n_unk_buckets, "known": self.known}

    @classmethod
    def from_dict(cls, data: dict) -> "Vocab":
        v = cls((), n_unk_buckets=int(data["n_unk_buckets"]))
        v.known = {str(k): int(i) for k, i in data["known"].items()}
        return v
