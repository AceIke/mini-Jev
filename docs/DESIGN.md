# Design notes

Why the code is shaped this way, what it deliberately gives up, and the numbers
behind each decision. The README is the summary; this is the reasoning.

## The claim being tested

One sentence, and everything else is in service of it or absent:

> A non-autoregressive model that reads a state once and scores every option in
> parallel can be trained, on a task whose Bayes posterior is known in closed
> form, to reproduce that posterior, and its probabilities can then be shown to
> be empirically calibrated with a confidence value that is useful for routing.

There is no second mechanism layered on top. No retrieval, no chain of thought,
no tool use, no prompt scaffolding. If the sentence above were false, you would
find out from `python -m minijev train` and `python -m minijev experiments`.

## Why the task is synthetic

The usual reason you cannot check whether a model's probabilities are right is
that you do not know the true conditional distribution. You have labels, so you
can measure accuracy and empirical calibration, but you cannot measure the
distance from the optimum because you do not know the optimum.

`NoisyTriageTask` removes that by construction:

* Three latent variables: `department` with 3 values, `is_urgent` binary,
  `severity` with 3 ordered levels.
* Each latent drives a disjoint vocabulary segment: topic cues, urgency markers,
  severity markers. Disjointness is what makes the posteriors factorise.
* Within a segment each token is drawn i.i.d. from a class-conditional
  categorical distribution with fixed constants, so

  ```
  log P(latent = j | tokens) = log P(j) + SUM_over_words count(word) * log P(word | j)
  ```

  is exact. It is computed, not estimated, in `posteriors_from_tokens`.

Two consequences follow.

The Bayes-optimal predictor is linear in token counts, so it lies inside
`MiniJevModel`'s hypothesis class. Matching the posterior is therefore an
achievable outcome and not a hope. The measured residual,
`mean_posterior_l1 ~ 5e-08`, is numerical noise.

The task is still ambiguous. A task where every posterior is one-hot would leave
every reliability bin empty and every ECE trivially zero, which would make the
whole exercise vacuous. The generator parameters (`cue_signal=0.10`,
`cue_confuse=0.05`, 12 topic slots) were picked by grid search over the generator
so that the Bayes-optimal top-1 accuracy is about 0.74 and the max-posterior
quartiles are roughly 0.58, 0.75 and 0.90.
`tests/test_tasks.py::test_the_task_is_neither_trivial_nor_impossible` keeps that
property from drifting.

The strongest check available is
`test_bayes_posterior_is_calibrated_against_realised_labels`, which samples 6000
states, computes the closed-form posterior for each, and verifies that it is
calibrated against the realised latent labels. That single test ties the
likelihood tables, the sampler and the inference function together; an error in
any of the three breaks it.

## Why this architecture

```
logits_j = (P_o . bag(option_j)) . (P_s . bag(state) + P_q . bag(instructions)) / sqrt(d)
```

That form gives four properties at once.

| property | how it follows |
| --- | --- |
| the state is read once | the state contributes exactly one vector, `u` |
| non-autoregressive | all `k` logits come from a single `O @ u` matmul |
| questions are independent | each question is a separate pass over the same `u`, so no question's encoding can reach another's |
| linear in token counts | sum pooling plus linear projections, which is what the Bayes rule needs here |

At `dim=48` the whole model is 9,265 parameters.

## Deliberate limits

These are choices, not oversights. Each is a place the code can be extended, and
each is stated in the README rather than only here.

Bag of words means no order and no syntax. That is the price of an exact Bayes
posterior and a short codebase. Without it, distance from the optimum stops being
measurable.

States are fixed length. Every generated state has the same number of slots, so
the model never faces the question of how much extra evidence should count. A
model that pools over a variable-length state has to answer it. TypeSafe's own
jaggedness notes describe the same issue in a much larger model as "large state
full of irrelevant detail".

Text only, one vocabulary. Unknown tokens hash into fixed buckets, so nothing
crashes and nothing is understood either.

No context management. Jev documents a 64k request budget with a 32k sub-budget
for state plus the longest question. This implementation never needs one, and a
serious version would need to say what happens at the limit.

Soft targets require a known posterior, which only exists because the task is
synthetic. On real data you have outcomes. That gap is the `hard_targets`
ablation, and it is in the repository specifically to price it.

There is no serving layer. The wire format is implemented in `types.py` as
`to_wire` and `from_wire`, and that is enough to show the contract without
maintaining a server that nobody is going to run in production.

## Confidence is a choice

TypeSafe's docs describe confidence as a convenience derived from the
distribution you already receive and say you are not locked into their
definition. This repository takes that at face value and ships two:

* `choice_confidence` generalises the formula their confidence explorer prints
  for the three-option case, `(3 * peak - 1) / 2`, to `n` options:
  `(n * peak - 1) / (n - 1)`, clamped to `[0, 1]`.
* `entropy_confidence` computes `1 - H(p) / log(n)`, the same endpoints with a
  different shape.

`tests/test_confidence.py` pins the documented value, 0.85 for
`[0.90, 0.06, 0.04]`, so the generalisation cannot drift away from its source.

## What the ablations showed

### Hard targets versus soft targets, at three training budgets

| budget | posterior-target dept ECE | hard-target dept ECE | posterior-target L1 | hard-target L1 |
| --- | --- | --- | --- | --- |
| 800 samples, 6 epochs, dim 32 | 0.0399 | 0.0328 | 0.0098 | 0.278 |
| 1500 samples, 8 epochs, dim 48 | 0.0216 | 0.0314 | 0.0003 | 0.285 |
| 4000 samples, 12 epochs, dim 48 | 0.0138 | 0.0149 | 0.0000 | 0.203 |

Distance to the posterior and Brier move the same way at every budget. The Choice
top-label ECE does not: at the smallest budget the hard-target run comes out
ahead on that one metric because the posterior-target run is itself underfit, and
underfit models are miscalibrated in their own direction. It is left in the
report. Removing the metric that disagreed would be the exact failure this
repository is about.

### Underfitting produces over-confidence

The `underfit` variant is 1500 samples for a single pass, which at batch size 256
is about six optimiser steps. Its fitted temperature is 2.37, so the model is
sharply over-confident, and a single post-hoc parameter drops its ECE from 0.214
to 0.045. Underfitting is not only an accuracy problem.

### Temperature scaling has a boundary

| model | fitted T | ECE before | ECE after | L1 to Bayes before | L1 after |
| --- | --- | --- | --- | --- | --- |
| `posterior_targets` | 1.01 | 0.014 | 0.013 | 0.000 | 0.006 |
| `hard_targets` | 1.06 | 0.016 | 0.025 | 0.204 | 0.210 |
| `underfit` | 2.37 | 0.214 | 0.045 | 0.512 | 0.429 |
| `evidence_shift` | 1.69 | 0.114 | 0.021 | 0.222 | 0.093 |

The model that already matches the posterior fits `T = 1.01` and is left alone,
which is the sanity check that the fitter is not chasing noise. The two models
that are over-confident, for different reasons, both improve substantially. The
hard-target model fits `T = 1.06` and its distance to the posterior does not
move, because its probabilities are the wrong shape rather than the wrong scale.
Rescaling is monotone, so it cannot reorder options, and it can never change an
answer. That is the whole of what post-hoc scaling can do, and it is why both the
ablations and the scaler are in the repository.

Fitting is always done on the first half of a held-out fold and reported on the
second half. Fitting on the same samples you report on would guarantee an
improvement on paper and mean nothing.

## Reading order

1. `src/minijev/types.py`, for what the contract is.
2. `src/minijev/tasks.py`, for the task and why the target is knowable.
3. `src/minijev/model.py`, for the forward and backward pass, a couple hundred
   lines.
4. `src/minijev/calibration.py`, for how you would know if it were wrong.
5. `src/minijev/temperature.py`, for what is fixable after the fact.
6. `src/minijev/experiments.py`, for what breaks it.
