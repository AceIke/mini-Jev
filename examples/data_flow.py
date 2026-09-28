"""Print the numbers at every step of one request.

    python examples/data_flow.py

This is the shortest way to see what the model actually does. Text becomes a
bag of word ids, the state and the question become two vectors that add to a
single context vector, every option is scored against that vector in one
matmul, and a softmax turns the scores into a distribution. Nothing is
generated, and no question can see another question's answer.
"""

import numpy as np

from minijev.runtime import plan_questions
from minijev.tasks import NoisyTriageTask
from minijev.text import normalize_state, tokenize
from minijev.train import TrainConfig, train

STATE = "my card was charged twice, please refund asap"


def vec(v: np.ndarray) -> str:
    head = " ".join(f"{x:+.3f}" for x in np.asarray(v).ravel()[:4])
    return f"[{head} ...]  d={np.size(v)}  |v|={np.linalg.norm(v):.3f}"


def section(title: str) -> None:
    print()
    print(title)
    print("-" * len(title))


def main() -> None:
    print("training a small model (dim=16 so the vectors stay printable)")
    mini = train(
        TrainConfig(n_train=1500, n_eval=400, epochs=6, batch_size=256,
                    dim=16, seed=0),
        verbose=False,
    ).mini
    model, vocab = mini.model, mini.vocab
    task = NoisyTriageTask()
    plans = plan_questions(vocab, task.questions())

    section("1. state -> text -> tokens")
    print("raw state :", STATE)
    print("normalised:", normalize_state(STATE))
    tokens = tokenize(normalize_state(STATE))
    print("tokens    :", tokens)

    section("2. tokens -> bag of word ids (order is discarded here)")
    state_bag = vocab.bag(tokens)
    inverse = {i: w for w, i in vocab.known.items()}
    seen, unknown = {}, 0
    for i, count in enumerate(state_bag):
        if not count:
            continue
        if i < vocab.n_unk_buckets:
            unknown += int(count)
        else:
            seen[inverse[i]] = int(count)
    print("word -> count:", seen)
    print("bag vector  :", vec(state_bag), f"  sum={state_bag.sum():.0f}")
    missing = [t for t in tokens if t not in vocab.known]
    print(f"out of vocabulary ({unknown} tokens): {missing}")
    print("those hash into a fixed unknown bucket. They do not crash, and they")
    print("carry no evidence either. This is the honest limit of a 32-word vocab.")

    section("3. state + question -> one context vector")
    plan = plans["department"]
    print("instruction:", task.questions()["department"].instructions)
    logits, cache = model.score_options(state_bag, plan.instr_bag, plan.option_bags)
    print("s (state) :", vec(cache["e_s"]), "-- embedded, before P_s")
    print("q (quest) :", vec(cache["e_q"]), "-- embedded, before P_q")
    print("u = s + q :", vec(cache["u"]))
    print("u is the only thing any option sees. It does not depend on the options.")

    section("4. all options scored in one matmul:  logits = O @ u / sqrt(d)")
    print("O shape (k x d):", cache["o"].shape, " u shape:", cache["u"].shape)
    probs = model.predict_options(state_bag, plan.instr_bag, plan.option_bags)
    for name, logit, p in zip(plan.option_names, logits, probs):
        print(f"  {name:<10} logit {logit:+.4f}   P {p:.4f}")
    peak = float(probs.max())
    n = len(probs)
    print(f"  confidence = (n*peak - 1)/(n - 1) = ({n}*{peak:.4f} - 1)/{n - 1}"
          f" = {(n * peak - 1) / (n - 1):.4f}")

    section("5. noul is a separate head on the same u")
    noul_plan = plans["is_urgent"]
    logit, _ = model.score_noul(state_bag, noul_plan.instr_bag)
    print(f"  logit = w_n . u + b_n = {logit:+.4f}")
    print(f"  noul  = sigmoid(logit) = {model.predict_noul(state_bag, noul_plan.instr_bag):.4f}")

    section("6. questions cannot interfere with each other")
    only_department = mini.system_one(state=STATE, questions={"department": task.questions()["department"]})
    all_three = mini.system_one(state=STATE, questions=task.questions())
    a = np.array(list(only_department.probabilities("department").values()))
    b = np.array(list(all_three.probabilities("department").values()))
    print("  asked alone :", np.array2string(a, precision=6))
    print("  asked with 2 other questions:", np.array2string(b, precision=6))
    print("  identical   :", bool(np.array_equal(a, b)))
    print("  max abs diff:", float(np.abs(a - b).max()))

    section("7. what comes out of the runtime")
    result = all_three
    print("  model      :", result.model)
    print("  department :", result["department"].to_wire())
    print("  is_urgent  :", result["is_urgent"].to_wire())
    print("  severity   :", result["severity"].to_wire())
    print("  usage      :", result.usage, " (output tokens are free in Jev, hence 0)")


if __name__ == "__main__":
    main()
