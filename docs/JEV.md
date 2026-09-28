# The System One model, explained

This is the knowledge base for the ideas mini-jev implements: what Jev is, how a
decision is shaped, where the sharp edges are, and which file in this repository
carries each idea.

Everything about the product comes from the official documentation at
<https://docs.typesafe.ai>, read on 2026-09-28. Anything that can drift is dated
because it will drift.

Contents

1. [The problem: text out versus decisions out](#1-the-problem-text-out-versus-decisions-out)
2. [The contract](#2-the-contract)
3. [The three primitives](#3-the-three-primitives)
4. [State](#4-state)
5. [The evaluation model](#5-the-evaluation-model)
6. [Confidence](#6-confidence)
7. [Calibration](#7-calibration)
8. [Designing with it](#8-designing-with-it)
9. [Known failure modes](#9-known-failure-modes)
10. [Where each idea lives in this repository](#10-where-each-idea-lives-in-this-repository)
11. [Glossary](#11-glossary)

---

## 1. The problem: text out versus decisions out

A language model is trained to produce text a person will read. When the
consumer of that output is a program instead of a person, the mismatch shows up
immediately. You end up coercing a text generator into emitting structured data,
then writing a parser, then writing retry logic for when the parser fails.

Jev starts from the other end. It is trained to return a decision and a
probability distribution over the possible answers, and it does not write prose
at all. There is no parser, because there is no text to parse.

TypeSafe calls this class of model a System One model, borrowing the term from
the fast, intuitive half of Kahneman's *Thinking, Fast and Slow*. The framing is
deliberate: these are meant to be fast gut-check judgments that sit inside a
larger workflow, not deliberative reasoning.

The stated design bet is that most AI usage in production will be machine to
machine, so the machine interface matters more than the chat interface. Whether
that bet pays off is a separate question. What follows from it is concrete: typed
output, calibrated probabilities, low latency, and answers a program can branch
on without a model in the loop.

## 2. The contract

One HTTP endpoint handles everything:

```
POST https://api.typesafe.ai/v1/systemone
Authorization: Bearer <API_KEY>
Content-Type: application/json
```

The request body has three fields.

| field | type | meaning |
| --- | --- | --- |
| `state` | string, object, or array | the material to judge |
| `model` | string | which model answers, for example `jev-latest` |
| `questions` | map of id to question | the judgments you want back |

The response mirrors the question ids.

| field | meaning |
| --- | --- |
| `model` | the versioned id that actually answered |
| `answers` | one answer per question id |
| `usage` | input and output token counts |

Two details worth knowing early. Question ids are yours and never reach the
model, so they cannot influence the answer. And the response's `model` field
reports the resolved version rather than the alias you sent, which is what lets
you log which build produced a result.

## 3. The three primitives

There are exactly three question types. Each returns a different shape, and all
three can be mixed in one request.

### 3.1 Choice

Pick one option from a set you define.

```json
{
  "department": {
    "type": "choice",
    "instructions": "Which team should handle this?",
    "criteria": {
      "billing": "Charges, invoices, refunds and payment problems.",
      "technical": "Bugs, crashes and integration failures.",
      "sales": "Pricing, quotes and upgrade requests."
    }
  }
}
```

The answer carries the winning option, a probability for every option, and a
confidence.

```json
{
  "type": "choice",
  "choice": "billing",
  "probabilities": { "billing": 0.88, "technical": 0.12, "sales": 0.0 },
  "confidence": 0.81
}
```

Notes that matter in practice:

* the option names and their descriptions are both sent to the model, so the
  descriptions are doing real work. Write them to separate the options from each
  other, not to restate the name;
* up to 255 options are allowed. Adding options costs a few tokens each, so a
  full taxonomy is usually better than a shortlist;
* add an `other` or `none of the above` option when the list might not cover
  everything, otherwise the model is forced to choose the least bad fit.

### 3.2 Score

Rate the state against ordered levels.

```json
{
  "severity": {
    "type": "score",
    "instructions": "How severe is the problem?",
    "criteria": ["Minor", "Moderate", "Blocking"]
  }
}
```

The answer gives a probability for each level, a legend mapping level index back
to its description, and an expected value that can land between levels.

```json
{
  "type": "score",
  "score": 1.05,
  "legend": { "0": "Minor", "1": "Moderate", "2": "Blocking" },
  "probabilities": { "0": 0.0, "1": 0.95, "2": 0.05 },
  "confidence": 0.92
}
```

Between two and ten levels. The docs are explicit that levels are not
arithmetically calibrated: `score: 1.05` is a position between "Moderate" and
"Blocking", not a measurement. Use it for thresholding, not for interpolation.

### 3.3 Noul

A yes or no question, answered with a probability.

```json
{
  "is_urgent": {
    "type": "noul",
    "instructions": "Does this convey urgency?",
    "criteria": { "true": "Explicitly time-sensitive", "false": "No urgency" }
  }
}
```

```json
{ "type": "noul", "noul": 0.95 }
```

One number, no distribution and no confidence. The docs are direct about why the
name is odd: it is a Jev-specific term, and worth treating as a label for "the
boolean primitive" rather than as an acronym.

### 3.4 Choosing between them

| if the answer is | use |
| --- | --- |
| one of a fixed set | `Choice` |
| a position on an ordered scale | `Score` |
| true or false | `Noul` |

One trap worth naming, because it is documented and easy to fall into: a `Choice`
with two options `yes`/`no` is **not** the same as a `Noul`. The Choice is
relative, so it answers "which of these two fits better" and its probabilities
are forced to sum to one. The Noul is absolute, so it can be low for both.
Designs that assume the two agree will eventually see them disagree.

## 4. State

State is everything you want judged. It can be a plain string, a JSON object, or
an array of text values.

| shape | good for |
| --- | --- |
| string | a message, a passage, an article |
| object | several named things that have to be compared, such as a message plus an order plus a policy |
| array | a sequence of turns or records |

Use an object when the material has parts, because field names are how the model
keeps the parts apart. The docs describe this as presenting the material to a
panel of experts before asking for a judgment.

Two constraints are easy to forget:

* **Text only.** Images, audio and video are not accepted. If you have them,
  convert to text or to named fields first.
* **State and questions compete for the same budget.** The request budget covers
  the state plus all questions together, and there is a tighter sub-budget for
  the state plus the single longest question.

## 5. The evaluation model

This is the part that most changes how you write code against it.

The state is read once. Every question is then evaluated in parallel and in
isolation. Three consequences follow, and they are all load-bearing.

**Adding questions is cheap in time and cheap in money.** Every question costs
its own tokens, but they are input tokens and they are processed together.
TypeSafe's own cookbook measures batching 13 questions into one call as roughly
an order of magnitude faster and cheaper than 13 separate calls, with no change
in the answers.

**Questions cannot contaminate each other.** Because each is evaluated against
the same input in isolation, one question's answer never becomes another's
context. There is no context rot between questions.

**Asking a question you might not need is close to free.** This gives rise to the
speculative fan-out pattern: ask every question the code could conceivably want,
including ones that only matter for some inputs, then let the code decide which
answers to use. If a ticket turns out not to be a bug report, ignore the severity
answer.

The corollary is a rule about dependencies. If a later judgment needs an earlier
answer, make a second request. The test is whether you can build the second
request without the first answer. If you need the answer to fetch more material,
or to decide what the next question's options are, the dependency is real. If
you could have asked both against the original state, ask them together and
combine them in code.

## 6. Confidence

`Choice` and `Score` return a confidence between 0 and 1. `Noul` does not.

Confidence is derived from the probability distribution that the answer already
carries. It is a summary of the shape of that distribution, not a second opinion
about the answer. A distribution concentrated on one option gives a high
confidence; a flat one gives a low confidence.

The docs state the three-option case as `(3 * peak - 1) / 2` and note that you
are not locked into their definition, which is the honest way to put it. Any
monotone function of the distribution will do, and different ones suit different
stakes. This repository implements the general-`n` form of that same statistic
and also ships an entropy-based alternative so the choice is visible:

```
choice_confidence  = (n * peak - 1) / (n - 1)        # this repository's default
entropy_confidence = 1 - H(p) / log(n)               # same endpoints, other shape
```

The architectural point is that confidence gives the model a way to say "I am
not sure about this one". That is what lets you build three behaviours out of two
numbers:

| confidence | behaviour |
| --- | --- |
| high | act automatically |
| medium | act cautiously, confirm, or gather more information |
| low | do not act, route to a person or fall back |

The thresholds are not one number. They scale with the cost of being wrong: a
read-only lookup can be gated at 0.5 while a destructive action should demand
0.9 or more. The code owns the risk tolerance; the model only reports its
uncertainty.

## 7. Calibration

Calibration is the property that makes the probabilities worth reading.

> Across many predictions, outcomes assigned probability 0.2 should happen about
> 20% of the time, and outcomes assigned 0.8 should happen about 80% of the time.

Three things about that definition are worth stating plainly, because they are
where most misunderstandings live.

**It is a statement about groups, not about single answers.** A well-calibrated
model can still be wrong on the specific case in front of you. Calibration tells
you how much to trust the numbers in aggregate.

**It is measurable and therefore falsifiable.** You bin predictions by stated
probability, compare mean confidence against observed frequency, and report the
gap (expected calibration error). That is what `calibration.py` does, and what
`docs/DESIGN.md` uses to hold this repository to its claims.

**It is a per-question property.** One question can be well calibrated while
another is not, and a distribution shift can wreck one while leaving the others
untouched. Reporting a single global number hides exactly the failure you need
to see.

### How Jev is trained for it

Three post-training approaches are contrasted in the docs:

| approach | what it optimises | what it produces |
| --- | --- | --- |
| RLHF | responses people prefer | chat assistants; can reward sycophancy and confident hallucinations |
| RLVR | verifiable rewards | strong at tasks like mathematics; slower and more expensive |
| RLCD | calibrated decisions | decisions plus probabilities that mean something |

Jev uses RLCD, reinforcement learning for calibrated decisions. The stated
failure mode it avoids is **mode dropping**: preference optimisation narrows the
output distribution towards whichever style scored well, which is the opposite of
what you want when the task is to report honest uncertainty across several
plausible answers.

## 8. Designing with it

The guidance is consistent across the docs, and it is mostly about
decomposition.

**Ask atomic questions.** Each question should be the kind of judgment a
knowledgeable person could make in a few seconds. If a question needs extended
reasoning or weighs several independent factors, split it.

**Recombine in code.** Ask about market size, technical feasibility and
differentiation separately, then combine the three with weights you control. When
priorities change you edit a coefficient instead of rewriting a prompt.

**Keep deterministic work out of the model.** Arithmetic, date handling,
counting, and set membership belong in code. The model's job is the part that is
genuinely a judgment.

**Put your domain rules in the criteria.** Since there is no fine-tuning per
customer, the request is the only place to encode your rules. Boundary cases
belong in the option descriptions, where they affect the answer.

## 9. Known failure modes

TypeSafe publishes a jaggedness list for the current model, which is unusually
forthcoming and worth reading in full. The shape of it:

| failure | why | what to do instead |
| --- | --- | --- |
| Literal reading | it answers the question you wrote, not the one you meant. Scope words and negations are taken at face value. | Write the exact condition. Put boundary cases in the criteria. Where interpretation is unavoidable, split into two literal questions and combine in code. |
| Counting | It recognises the shape of a count rather than tallying, and the error grows with size. | Count in code. To count items matching a criterion, ask one question per item and add the answers yourself. |
| Numeric representations | Semantic values beat numeric ones. Hex colours and raw assembly are weaker than names. | Convert in code, then let the model judge something it can actually read, such as whether a colour reads as a warning. |
| Date and time comparison | Dates are read as text, not as ordered quantities. | Treat extraction as a judgment over a closed set of parts (month, day, year, or an explicit "not stated"), and do all arithmetic in code. |
| Indirection | Double negatives and multi-hop questions cost accuracy. | Reduce the number of hops and name the relevant parts of the state directly. |
| Large state | Unrelated material acts as a distractor and accuracy falls. | Filter in code first. Ask for relevance as its own question if you cannot filter. |
| Adversarial content | The state is data, and it is not treated as hostile by default, so injected instructions can move the answer. | Be explicit in the criteria. Test edge cases before deployment. |
| Contradictory instructions and criteria | If the instruction and the criteria ask for different things, the model gets confused. | Treat the criteria as an extension of the instruction and align them. |
| Structural invariants | A question and its negation need not sum to one, and a Noul need not agree with a two-option Choice. | Do not rely on expected identities. Word each question to mean exactly what you want, and do not carry a threshold between question types. |
| Generation | It is not trained to produce text. | If the answer space is bounded, use Choice. If you need generated text, use a generative model. |

Two of these deserve emphasis because they are counterintuitive. First, a model
described as highly deterministic can still fail structural invariants: similar
inputs give quantitatively similar outputs, but `P(x)` and `1 - P(not x)` are not
guaranteed to agree. Second, the model is honest about uncertainty but not
sceptical about its input, so prompt-injection style content in the state is a
real risk and belongs in your threat model.

## 10. Where each idea lives in this repository

| idea from the docs | file | what it does here | what is different |
| --- | --- | --- | --- |
| the three primitives | `src/minijev/types.py` | `Choice`, `Score` and `Noul` as frozen dataclasses, with `to_wire` that matches the documented JSON | validation limits match the docs: 255 options, 2 to 10 levels |
| state as string, object or array | `src/minijev/text.py` | `normalize_state` renders any of the three into text, then tokenises and hashes out-of-vocabulary words into buckets | bag of words, so order and syntax are discarded |
| state read once, one context vector | `src/minijev/model.py` | `_context_vector` produces a single `u`; every option is scored against it by one matmul | a real model has a transformer in front, this has a linear map |
| questions evaluated in isolation | `src/minijev/runtime.py` | each question is a separate pass over the same vector, and `tests/test_runtime.py` asserts adding a question cannot change another answer | the property is tested here, and demonstrated live in `examples/data_flow.py` |
| confidence derived from the distribution | `src/minijev/confidence.py` | the general-`n` form of the documented three-option formula, plus an entropy alternative | same statistic, different `n` |
| calibration as a measurable property | `src/minijev/calibration.py` | ECE, MCE, Brier, NLL, reliability bins, selective accuracy | |
| calibrated decisions rather than preferences | `src/minijev/tasks.py` | a synthetic task whose Bayes posterior is closed-form, so the optimum is known and the gap is measurable | RLCD needs outcomes at web scale; this needs a generative process you wrote |
| decomposed judgments combined in code | `src/minijev/tasks.py`, `src/minijev/experiments.py` | three atomic questions per state, combined by code, with ablations that change one mechanism at a time | |
| uncertainty must be usable, not just present | `src/minijev/temperature.py` | post-hoc temperature scaling, plus the boundary where it stops helping | |
| model answers are versioned | `src/minijev/runtime.py` | every result carries `model`, and `usage` reports input tokens with outputs free, matching Jev's billing | |

## 11. Glossary

| term | meaning |
| --- | --- |
| System One model | a model trained to return typed decisions and calibrated probabilities rather than generated text |
| primitive | one of the three question types: Choice, Score, Noul |
| state | the material being judged, sent once per request |
| instructions | the question text |
| criteria | the option descriptions for Choice, the level descriptions for Score, the true/false meaning for Noul |
| noul | the boolean primitive, returning `P(yes)` with no confidence |
| confidence | a number in `[0, 1]` derived from the answer's probability distribution |
| calibration | agreement between stated probabilities and observed frequencies, measured across groups |
| ECE | expected calibration error, the sample-weighted mean gap between confidence and accuracy |
| RLCD | reinforcement learning for calibrated decisions, the training approach Jev uses |
| mode dropping | a preference-tuning failure where the output distribution narrows towards one style |
| speculative fan-out | asking questions you may not need in the same request, because extra questions are cheap |
| jaggedness | documented, accepted failure modes of a specific model version |
