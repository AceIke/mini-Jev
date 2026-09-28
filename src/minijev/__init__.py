"""mini-jev: a tiny, educational System One decision model.

Jev (TypeSafe AI) turns a *state* plus typed *questions* into typed *answers*
with calibrated probabilities. mini-jev is a from-scratch, NumPy-only
reimplementation of that *contract* with a deliberately small model, so that
every claim in this repository can be checked by reading the code and running
the tests.

It is not affiliated with TypeSafe AI, and it is not a reproduction of Jev.
See ``docs/DESIGN.md`` for what is and is not faithful to the original.
"""

from .types import (
    Answer,
    Choice,
    ChoiceAnswer,
    Noul,
    NoulAnswer,
    Question,
    Score,
    ScoreAnswer,
    SystemOneResult,
    choice,
    noul,
    score,
)
from .confidence import (
    choice_confidence,
    entropy_confidence,
    normalize_probs,
)
from .runtime import MiniSystemOne
from .model import MiniJevModel
from .text import Vocab, normalize_state, tokenize
from .temperature import TemperatureScaler, fit_temperature, scale_probs

__version__ = "0.2.0"

__all__ = [
    "Answer",
    "Choice",
    "ChoiceAnswer",
    "MiniJevModel",
    "MiniSystemOne",
    "Noul",
    "NoulAnswer",
    "Question",
    "Score",
    "ScoreAnswer",
    "SystemOneResult",
    "TemperatureScaler",
    "Vocab",
    "choice",
    "choice_confidence",
    "entropy_confidence",
    "fit_temperature",
    "normalize_probs",
    "normalize_state",
    "noul",
    "scale_probs",
    "score",
    "tokenize",
]
