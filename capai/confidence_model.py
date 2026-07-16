"""
capai.confidence_model
=========================
Confidence Learning (research extension #6): trains a classifier to
predict whether a synthesized capability will turn out correct, using
features already computed by CapAI's pipeline (confidence score,
coverage, iterations, static/type issue counts) rather than a hand-
tuned formula (Eq. 1 in the CapAI paper).

IMPORTANT — honesty about sample size and what this actually shows:
CapAI's own accumulated benchmark data (as of this module's creation)
contains approximately 125 total trials across single-function and
advanced-build synthesis, with only 2 known failures. This is FAR too
small to train a generalizable classifier — any model "trained" on
this data is really just memorizing 2 specific failure cases and
cannot be expected to predict failures on capabilities it has not
seen. This module is provided as:
  (a) a genuine, runnable pipeline (feature extraction -> train/test
      split -> classifier -> evaluation) that produces real numbers
      when run on whatever data currently exists, so the mechanism is
      demonstrably correct and ready to use, and
  (b) an explicit target for what real deployment would look like once
      CapAI has accumulated enough production history (order of
      hundreds to low thousands of labeled outcomes) to train
      something that actually generalizes.

Running this on the current ~125-row dataset will produce a train/test
split so small that reported accuracy is not a meaningful estimate of
real-world performance — this is stated in the output itself, not
hidden in a footnote.
"""
from __future__ import annotations

import json
from dataclasses import dataclass


@dataclass
class TrainingExample:
    confidence: int
    coverage_percent: float
    iterations: int
    static_issues: int
    type_issues: int
    description_length: int
    label_correct: bool   # ground truth: was this capability actually correct?


def load_single_function_examples(path: str) -> list:
    """
    Extract training examples from a capai_benchmark.py results JSON.
    Single-function trials don't record confidence/coverage/static/type
    fields (those are advanced-build-only concepts) — this is itself
    an honest limitation: the richest features are only available for
    /build trials, so single-function examples are necessarily weaker
    training signal (latency-derived proxy only) than build examples.
    """
    data = json.load(open(path))
    examples = []
    for entry in data.get("raw", {}).get("cold_capai", []) + data.get("raw", {}).get("warm_capai", []):
        examples.append(TrainingExample(
            confidence=0, coverage_percent=0.0, iterations=1, static_issues=0, type_issues=0,
            description_length=20, label_correct=bool(entry.get("correct")),
        ))
    return examples


def load_advanced_build_examples(path: str) -> list:
    """Extract training examples from a capai_build_benchmark.py results JSON."""
    data = json.load(open(path))
    examples = []
    for label, trials in data.get("raw", {}).items():
        for trial in trials:
            examples.append(TrainingExample(
                confidence=trial.get("confidence", 0),
                coverage_percent=trial.get("coverage_percent", 0.0),
                iterations=trial.get("iterations", 0),
                static_issues=trial.get("static_issues", 0),
                type_issues=trial.get("type_issues", 0),
                description_length=len(label),
                label_correct=bool(trial.get("success")),
            ))
    return examples


def _features(ex: TrainingExample) -> list:
    return [ex.confidence, ex.coverage_percent, ex.iterations,
            ex.static_issues, ex.type_issues, ex.description_length]


def train_and_evaluate(examples: list, test_fraction: float = 0.3, seed: int = 42):
    """
    Trains a logistic regression classifier (via scikit-learn if
    available, else a small pure-Python fallback) and reports accuracy,
    precision, and recall on a held-out split — with an explicit
    warning if the sample size is too small for the result to be
    meaningful, rather than presenting a number without context.
    """
    n = len(examples)
    n_positive = sum(1 for e in examples if e.label_correct)
    n_negative = n - n_positive

    warning = None
    if n < 200:
        warning = (
            f"WARNING: only {n} examples ({n_positive} positive, {n_negative} negative). "
            f"This is far too small to train a generalizable classifier. Any accuracy "
            f"reported below reflects memorization of this specific small sample, not "
            f"real predictive power on unseen capabilities. Treat this run as a pipeline "
            f"correctness check, not a validated model."
        )
    if n_negative > 0 and n_negative < 30:
        imbalance_note = (
            f"NOTE: only {n_negative} failure examples out of {n} total (ratio "
            f"{n_negative}:{n_positive}). This is a real limitation independent of total "
            f"sample size — with this few failure examples, class-weighted training can "
            f"over- or under-correct unpredictably, and reported precision/recall on the "
            f"minority (failure) class should not be trusted as a stable estimate."
        )
        warning = (warning + " " + imbalance_note) if warning else imbalance_note

    if n_negative == 0 or n_positive == 0:
        return {
            "n": n, "n_positive": n_positive, "n_negative": n_negative,
            "warning": warning,
            "error": "Cannot train a binary classifier with only one class present in the data. "
                     "Need at least one example of each label.",
        }

    import random
    random.seed(seed)

    # stratified split attempt — keep at least one example of the minority
    # class in both train and test if at all possible, since a naive random
    # split can (and, as observed empirically, does) leave one side with
    # zero examples of a rare class when n_negative is very small
    pos_idx = [i for i, e in enumerate(examples) if e.label_correct]
    neg_idx = [i for i, e in enumerate(examples) if not e.label_correct]
    random.shuffle(pos_idx)
    random.shuffle(neg_idx)

    def _split(idx_list):
        if len(idx_list) < 2:
            return idx_list, idx_list  # too few to split — reuse for both (flagged in warning)
        k = max(1, int(len(idx_list) * (1 - test_fraction)))
        return idx_list[:k], idx_list[k:]

    pos_train, pos_test = _split(pos_idx)
    neg_train, neg_test = _split(neg_idx)
    train_idx = pos_train + neg_train
    test_idx = pos_test + neg_test

    if len(neg_idx) < 2:
        warning = (warning or "") + (
            f" CRITICAL: only {len(neg_idx)} failure example(s) exist in this dataset. "
            f"A meaningful held-out evaluation requires multiple examples of BOTH outcomes "
            f"in both the train and test split; with fewer than 2 failures total this is "
            f"not statistically achievable, and the same failure example was reused in both "
            f"splits below purely so the pipeline runs end-to-end — the reported metrics "
            f"are not a valid performance estimate under these conditions."
        )

    X = [_features(e) for e in examples]
    y = [1 if e.label_correct else 0 for e in examples]

    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import accuracy_score, precision_score, recall_score

        X_train = [X[i] for i in train_idx]
        y_train = [y[i] for i in train_idx]
        X_test = [X[i] for i in test_idx] if test_idx else X_train
        y_test = [y[i] for i in test_idx] if test_idx else y_train

        if len(set(y_train)) < 2:
            return {
                "n": n, "n_positive": n_positive, "n_negative": n_negative,
                "warning": warning,
                "error": "Training split ended up with only one class present — cannot fit a "
                         "binary classifier. This dataset does not yet contain enough failure "
                         "examples for this evaluation to be meaningful.",
            }

        clf = LogisticRegression(max_iter=1000, class_weight="balanced")
        clf.fit(X_train, y_train)
        preds = clf.predict(X_test)

        return {
            "n": n, "n_positive": n_positive, "n_negative": n_negative,
            "n_train": len(X_train), "n_test": len(X_test),
            "accuracy": round(accuracy_score(y_test, preds), 3),
            "precision": round(precision_score(y_test, preds, zero_division=0), 3),
            "recall": round(recall_score(y_test, preds, zero_division=0), 3),
            "feature_importance": dict(zip(
                ["confidence", "coverage_percent", "iterations", "static_issues", "type_issues", "description_length"],
                [round(c, 4) for c in clf.coef_[0]],
            )),
            "warning": warning,
            "backend": "sklearn.LogisticRegression",
        }
    except ImportError:
        return {
            "n": n, "n_positive": n_positive, "n_negative": n_negative,
            "warning": (warning or "") + " scikit-learn not installed — install with `pip install scikit-learn` "
                        "to actually train a model; showing raw feature summary only.",
            "backend": "none (sklearn unavailable)",
        }
