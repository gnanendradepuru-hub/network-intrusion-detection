"""Train and compare intrusion-detection models on NSL-KDD.

Protocol (so the test numbers stay honest):
  1. Deduplicate KDDTrain+, split off a stratified validation set.
  2. Fit every model/imbalance variant on the train split.
  3. Pick the winner by validation macro-F1 only.
  4. Refit the winner on the full training data.
  5. Evaluate once on KDDTest+ (which contains attack types never seen in training).
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix, f1_score, precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

sys.path.insert(0, str(Path(__file__).resolve().parent))
from data import CLASSES, FlowPreprocessor, clean, load_raw  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "artifacts"
OUT.mkdir(exist_ok=True)
SEED = 42
CLASS_IDX = {c: i for i, c in enumerate(CLASSES)}


def sqrt_balanced_weights(y: np.ndarray) -> np.ndarray:
    """Per-sample weights ~ 1/sqrt(class frequency), normalised to mean 1.

    Full 'balanced' weighting gives the 52 U2R rows a weight in the hundreds and
    makes trees chase noise; the square root keeps rare classes audible without that.
    """
    counts = np.bincount(y, minlength=len(CLASSES)).astype(float)
    w_class = (counts.sum() / (len(CLASSES) * np.maximum(counts, 1))) ** 0.5
    w = w_class[y]
    return w / w.mean()


def make_models() -> dict:
    return {
        "RandomForest": RandomForestClassifier(
            n_estimators=150, n_jobs=-1, random_state=SEED, min_samples_leaf=1),
        "XGBoost": XGBClassifier(
            n_estimators=150, max_depth=6, learning_rate=0.1, subsample=0.8,
            colsample_bytree=0.8, tree_method="hist", n_jobs=-1, random_state=SEED,
            objective="multi:softprob", eval_metric="mlogloss"),
        "LightGBM": LGBMClassifier(
            n_estimators=150, learning_rate=0.1, num_leaves=63, subsample=0.8,
            subsample_freq=1, colsample_bytree=0.8, n_jobs=-1, random_state=SEED,
            verbose=-1),
    }


def fit(model, X, y, weighted: bool):
    sw = sqrt_balanced_weights(y) if weighted else None
    model.fit(X, y, sample_weight=sw)
    return model


def binary_metrics(y_true_idx, proba, normal_idx=0) -> dict:
    """Attack-vs-normal view: this is what an alerting system actually does."""
    is_attack = (y_true_idx != normal_idx).astype(int)
    score = 1.0 - proba[:, normal_idx]
    pred = (score >= 0.5).astype(int)
    p, r, f, _ = precision_recall_fscore_support(is_attack, pred, average="binary", zero_division=0)
    return {
        "precision": float(p), "recall": float(r), "f1": float(f),
        "roc_auc": float(roc_auc_score(is_attack, score)),
        "false_alarm_rate": float(((pred == 1) & (is_attack == 0)).sum() / max((is_attack == 0).sum(), 1)),
    }


def plot_confusion(cm, title, path):
    cmn = cm / cm.sum(axis=1, keepdims=True)
    fig, ax = plt.subplots(figsize=(5.6, 4.8))
    im = ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(CLASSES)), CLASSES)
    ax.set_yticks(range(len(CLASSES)), CLASSES)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(title)
    for i in range(len(CLASSES)):
        for j in range(len(CLASSES)):
            ax.text(j, i, f"{cm[i, j]}\n({cmn[i, j]:.0%})", ha="center", va="center",
                    fontsize=7, color="white" if cmn[i, j] > 0.5 else "black")
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def write_replay_sample(test_raw, per_attack_family: int = 60, n_normal: int = 500) -> None:
    """Held-out traffic the dashboard replays: mostly normal, with every attack family present."""
    parts = [g.sample(min(len(g), n_normal if fam == "normal" else per_attack_family),
                      random_state=SEED)
             for fam, g in test_raw.groupby("family")]
    import pandas as pd
    sample = pd.concat(parts).sample(frac=1, random_state=SEED)
    out = sample.drop(columns=["difficulty", "label"]).rename(
        columns={"attack": "true_attack", "family": "true_family"})
    out.to_json(OUT / "replay_sample.json", orient="records")


def main() -> None:
    t0 = time.time()
    print("Loading data")
    train_raw = clean(load_raw("KDDTrain+.txt"))
    test_raw = load_raw("KDDTest+.txt")

    print("\nClass counts (train after dedupe):")
    print(train_raw["family"].value_counts().to_string())
    print("\nClass counts (test):")
    print(test_raw["family"].value_counts().to_string())

    y_all = train_raw["family"].map(CLASS_IDX).to_numpy()
    idx_tr, idx_val = train_test_split(
        np.arange(len(train_raw)), test_size=0.2, stratify=y_all, random_state=SEED)

    # Preprocessor schema is fit on the training split only.
    pre = FlowPreprocessor().fit(train_raw.iloc[idx_tr])
    X_tr = pre.transform(train_raw.iloc[idx_tr])
    X_val = pre.transform(train_raw.iloc[idx_val])
    y_tr, y_val = y_all[idx_tr], y_all[idx_val]

    # ---- model selection on validation ------------------------------------
    print("\n== Validation (model + imbalance handling comparison) ==")
    val_rows = []
    for name, _ in make_models().items():
        for weighted in (False, True):
            model = fit(make_models()[name], X_tr, y_tr, weighted)
            pred = model.predict(X_val)
            per_class = f1_score(y_val, pred, average=None, labels=range(len(CLASSES)), zero_division=0)
            row = {
                "model": name, "weighted": weighted,
                "macro_f1": float(f1_score(y_val, pred, average="macro")),
                "per_class_f1": {c: float(v) for c, v in zip(CLASSES, per_class)},
            }
            val_rows.append(row)
            tag = "sqrt-balanced" if weighted else "plain"
            print(f"  {name:13s} {tag:14s} macro-F1={row['macro_f1']:.4f}  "
                  + "  ".join(f"{c}={v:.3f}" for c, v in zip(CLASSES, per_class)))

    best = max(val_rows, key=lambda r: r["macro_f1"])
    print(f"\nSelected on validation: {best['model']} (weighted={best['weighted']}) "
          f"macro-F1={best['macro_f1']:.4f}")

    # ---- refit every variant on full train, evaluate on test ---------------
    print("\n== Test (KDDTest+, includes unseen attack types) ==")
    pre = FlowPreprocessor().fit(train_raw)
    X_full = pre.transform(train_raw)
    X_test = pre.transform(test_raw)
    y_test = test_raw["family"].map(CLASS_IDX).to_numpy()
    seen_attacks = set(train_raw["attack"])
    novel_mask = (~test_raw["attack"].isin(seen_attacks) & (test_raw["family"] != "normal")).to_numpy()
    known_attack_mask = (test_raw["attack"].isin(seen_attacks) & (test_raw["family"] != "normal")).to_numpy()

    test_rows = []
    fitted = {}
    for r in val_rows:
        model = fit(make_models()[r["model"]], X_full, y_all, r["weighted"])
        proba = model.predict_proba(X_test)
        pred = proba.argmax(axis=1)
        p, rc, f, _ = precision_recall_fscore_support(
            y_test, pred, labels=range(len(CLASSES)), zero_division=0)
        score = 1.0 - proba[:, 0]
        det = (score >= 0.5)
        entry = {
            "model": r["model"], "weighted": r["weighted"],
            "accuracy": float((pred == y_test).mean()),
            "macro_f1": float(f1_score(y_test, pred, average="macro")),
            "per_class": {c: {"precision": float(p[i]), "recall": float(rc[i]), "f1": float(f[i]),
                              "support": int((y_test == i).sum())} for i, c in enumerate(CLASSES)},
            "binary": binary_metrics(y_test, proba),
            "detection_rate_known_attacks": float(det[known_attack_mask].mean()),
            "detection_rate_novel_attacks": float(det[novel_mask].mean()),
        }
        test_rows.append(entry)
        fitted[(r["model"], r["weighted"])] = (model, proba, pred)
        tag = "sqrt-balanced" if r["weighted"] else "plain"
        print(f"  {r['model']:13s} {tag:14s} acc={entry['accuracy']:.4f} macroF1={entry['macro_f1']:.4f} "
              f"attack-recall known={entry['detection_rate_known_attacks']:.3f} "
              f"novel={entry['detection_rate_novel_attacks']:.3f} "
              f"false-alarm={entry['binary']['false_alarm_rate']:.3f}")

    # ---- report on the selected model --------------------------------------
    key = (best["model"], best["weighted"])
    model, proba, pred = fitted[key]
    print(f"\n== Selected model on test: {key} ==")
    print(classification_report(y_test, pred, target_names=CLASSES, digits=3, zero_division=0))
    cm = confusion_matrix(y_test, pred, labels=range(len(CLASSES)))
    plot_confusion(cm, f"{best['model']} on KDDTest+", OUT / "confusion_matrix.png")

    # Comparison chart: macro-F1 + novel-attack detection per variant
    labels = [f"{r['model']}\n{'weighted' if r['weighted'] else 'plain'}" for r in test_rows]
    fig, ax = plt.subplots(figsize=(8, 3.8))
    x = np.arange(len(labels))
    ax.bar(x - 0.2, [r["macro_f1"] for r in test_rows], 0.4, label="Macro-F1 (5 classes)")
    ax.bar(x + 0.2, [r["detection_rate_novel_attacks"] for r in test_rows], 0.4,
           label="Detection rate, unseen attack types")
    ax.set_xticks(x, labels, fontsize=8)
    ax.set_ylim(0, 1)
    ax.legend(fontsize=8)
    ax.set_title("KDDTest+ results by model and imbalance handling")
    fig.tight_layout()
    fig.savefig(OUT / "model_comparison.png", dpi=140)
    plt.close(fig)

    # Novel attack breakdown for the selected model
    det = (1.0 - proba[:, 0]) >= 0.5
    novel_break = {}
    for a in sorted(set(test_raw.loc[novel_mask, "attack"])):
        m = (test_raw["attack"] == a).to_numpy()
        novel_break[a] = {"family": test_raw.loc[m, "family"].iloc[0], "n": int(m.sum()),
                          "detected": float(det[m].mean())}

    joblib.dump({"model": model, "preprocessor": pre, "classes": CLASSES,
                 "name": f"{best['model']}{' (sqrt-balanced weights)' if best['weighted'] else ''}"},
                OUT / "model.joblib", compress=3)
    results = {"selected": best, "validation": val_rows, "test": test_rows,
               "selected_test": next(r for r in test_rows
                                     if (r["model"], r["weighted"]) == key),
               "novel_attack_breakdown": novel_break,
               "n_train": int(len(train_raw)), "n_test": int(len(test_raw)),
               "classes": CLASSES}
    (OUT / "results.json").write_text(json.dumps(results, indent=2))

    # A small replay sample for the dashboard (stratified so rare classes appear)
    write_replay_sample(test_raw)

    print(f"\nDone in {time.time() - t0:.0f}s. Artifacts in {OUT}")


if __name__ == "__main__":
    main()
