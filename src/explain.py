"""SHAP explainability: per-alert reasons and a global importance chart."""
from __future__ import annotations

import sys
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import shap

sys.path.insert(0, str(Path(__file__).resolve().parent))
from data import CLASSES, load_raw  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "artifacts"


def shap_array(explainer: shap.TreeExplainer, X) -> np.ndarray:
    """Return SHAP values as (n_samples, n_features, n_classes) for any shap version."""
    sv = explainer.shap_values(X)
    if isinstance(sv, list):  # older shap: list of (n, f) per class
        sv = np.stack(sv, axis=-1)
    return np.asarray(sv)


def top_reasons(explainer, row_df, class_idx: int, k: int = 4) -> list[dict]:
    """Features pushing the model toward `class_idx` for one flow, strongest first."""
    sv = shap_array(explainer, row_df)[0, :, class_idx]
    order = np.argsort(-np.abs(sv))[:k]
    cols = list(row_df.columns)
    return [{"feature": cols[i], "value": float(row_df.iloc[0, i]),
             "impact": float(sv[i])} for i in order]


def main() -> None:
    bundle = joblib.load(OUT / "model.joblib")
    model, pre = bundle["model"], bundle["preprocessor"]
    test = load_raw("KDDTest+.txt")
    sample = test.sample(1500, random_state=0)
    X = pre.transform(sample)

    explainer = shap.TreeExplainer(model)
    sv = shap_array(explainer, X)  # (n, f, c)
    cols = np.array(X.columns)

    # Global importance: mean |SHAP| per feature, summed over attack classes.
    attack_idx = [i for i, c in enumerate(CLASSES) if c != "normal"]
    imp_by_class = np.abs(sv).mean(axis=0)  # (f, c)
    total = imp_by_class[:, attack_idx].sum(axis=1)
    top = np.argsort(-total)[:15][::-1]

    fig, ax = plt.subplots(figsize=(7.5, 5.2))
    left = np.zeros(len(top))
    palette = {"dos": "#c8553d", "probe": "#2a6f97", "r2l": "#7a9e7e", "u2r": "#6d597a"}
    for ci in attack_idx:
        vals = imp_by_class[top, ci]
        ax.barh(cols[top], vals, left=left, label=CLASSES[ci], color=palette[CLASSES[ci]])
        left += vals
    ax.set_xlabel("Mean |SHAP value| (stacked by attack class)")
    ax.set_title("What drives the model's attack calls")
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(OUT / "shap_global.png", dpi=140)
    plt.close(fig)
    print("Saved", OUT / "shap_global.png")
    for i in top[::-1][:10]:
        print(f"  {cols[i]:34s} {total[i]:.4f}")


if __name__ == "__main__":
    main()
