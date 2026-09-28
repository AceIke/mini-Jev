"""The hand-written backward pass must match finite differences."""

import numpy as np
import pytest

from minijev.model import Item, MiniJevModel
from minijev.text import Vocab

VOCAB_WORDS = [
    "charge", "refund", "invoice", "error", "crash", "timeout",
    "pricing", "quote", "asap", "urgent", "minor", "blocking", "usual",
]


def _fixture(seed: int = 7):
    vocab = Vocab(VOCAB_WORDS, n_unk_buckets=8)
    rng = np.random.default_rng(seed)
    model = MiniJevModel(vocab.size, dim=6, seed=seed)
    items = []
    for _ in range(3):
        state = vocab.bag(rng.choice(VOCAB_WORDS, size=7))
        instr = vocab.bag(rng.choice(VOCAB_WORDS, size=3))
        options = np.stack(
            [vocab.bag(rng.choice(VOCAB_WORDS, size=2)) for _ in range(3)]
        )
        items.append(
            Item("choice", state, instr, options, rng.dirichlet(np.ones(3)))
        )
    # a score question is scored exactly like a choice, plus an ordinal target
    state = vocab.bag(rng.choice(VOCAB_WORDS, size=6))
    instr = vocab.bag(rng.choice(VOCAB_WORDS, size=2))
    options = np.stack(
        [vocab.bag(rng.choice(VOCAB_WORDS, size=2)) for _ in range(3)]
    )
    items.append(Item("score", state, instr, options, np.array([0.2, 0.3, 0.5])))
    items.append(Item("noul", state, instr, None, 0.65))
    return model, items


def test_analytic_gradients_match_finite_differences():
    model, items = _fixture()
    _directional_check(model, items)


@pytest.mark.parametrize("kind", ["choice", "score", "noul"])
def test_gradients_match_through_each_head(kind):
    """Check each head in isolation so a failure points at one code path."""

    model, items = _fixture()
    subset = [item for item in items if item.kind == kind]
    assert subset, f"fixture has no {kind} item"
    _directional_check(model, subset)


def test_loss_decreases_after_one_adam_step():
    from minijev.model import Adam

    model, items = _fixture()
    before, grads = model.loss_and_grads(items)
    Adam(model, lr=0.1).step(grads)
    after, _ = model.loss_and_grads(items)
    assert after < before


def test_gradients_vanish_when_prediction_equals_target():
    """If p == t the soft-target gradient p - t is zero, everywhere."""

    vocab = Vocab(["a", "b"], n_unk_buckets=0)
    model = MiniJevModel(vocab.size, dim=4, seed=0)
    model.P_o[:] *= 0.0
    model.w_n[:] = 0.0
    state = vocab.bag(["a"])
    instr = vocab.bag(["b"])
    options = np.stack([vocab.bag(["a"]), vocab.bag(["b"])])
    item = Item("choice", state, instr, options, np.array([0.5, 0.5]))
    # Zero option projections -> all logits equal -> p is uniform, which is
    # exactly the target here, so the gradient is zero.
    loss, grads = model.loss_and_grads([item])
    assert loss == pytest.approx(np.log(2))
    for value in grads.as_dict().values():
        assert np.allclose(value, 0.0)


def _directional_check(model, items, seed: int = 1234) -> None:
    """Compare sum(grad * direction) with a central finite difference."""

    _, grads = model.loss_and_grads(items)
    grad_dict = grads.as_dict()
    rng = np.random.default_rng(seed)
    eps = 1e-6
    for name, param in model.parameters().items():
        direction = rng.normal(size=param.shape)
        direction /= np.linalg.norm(direction)
        analytic = float(np.sum(grad_dict[name] * direction))

        original = param.copy()
        param[...] = original + eps * direction
        loss_plus, _ = model.loss_and_grads(items)
        param[...] = original - eps * direction
        loss_minus, _ = model.loss_and_grads(items)
        param[...] = original

        numeric = (loss_plus - loss_minus) / (2 * eps)
        assert analytic == pytest.approx(numeric, rel=1e-4, abs=1e-7), (
            f"{name}: analytic {analytic} vs numeric {numeric}"
        )
