import json
import joblib
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
from sklearn.pipeline import Pipeline

warnings.filterwarnings("ignore")
print("Libraries ready")

# ── PATHS ──────────────────────────────────────────────────────
OUTPUT_DIR = Path("data/processed")
MODEL_DIR  = Path("models")
MODEL_DIR.mkdir(exist_ok=True)

# ── LOAD DATA ──────────────────────────────────────────────────
with open(OUTPUT_DIR / "metadata.json") as f:
    meta = json.load(f)

FEATURE_NAMES = meta["feature_names"]
LABEL_MAP     = meta["label_map"]
LABEL_NAMES   = {v: k for k, v in LABEL_MAP.items()}
ASANAS        = meta["asanas"]

df = pd.read_csv(OUTPUT_DIR / "master_dataset.csv")

print("Loaded rows    :", len(df))
print("Total features :", len(FEATURE_NAMES))
print("Asanas         :", ASANAS)
print("\nSample counts:")
print(df.groupby(["asana", "label"]).size().unstack(fill_value=0))

# ── MODEL SETTINGS ─────────────────────────────────────────────
RF_PARAMS = {
    "n_estimators":     300,
    "max_depth":        12,
    "min_samples_leaf": 3,
    "class_weight":     "balanced",
    "random_state":     42,
    "n_jobs":           -1,
}

# ── TRAIN ONE MODEL PER ASANA ──────────────────────────────────
results = {}

for asana in ASANAS:
    print("\n" + "=" * 55)
    print("  Training:", asana.upper())
    print("=" * 55)

    adf = df[df["asana"] == asana].copy()

    print("  Total   :", len(adf))
    print("  Good    :", len(adf[adf["label"] == "good"]))
    print("  Average :", len(adf[adf["label"] == "avg"]))
    print("  poor     :", len(adf[adf["label"] == "poor"]))

    if adf["label"].nunique() < 2:
        print("  Skipping — only one class found")
        continue

    if len(adf) < 30:
        print("  Skipping — too few samples")
        continue

    X = adf[FEATURE_NAMES].values
    y = adf["label_int"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("clf",    RandomForestClassifier(**RF_PARAMS))
    ])

    cv = cross_val_score(
        model, X_train, y_train,
        cv=StratifiedKFold(5, shuffle=True, random_state=42),
        scoring="f1_weighted"
    )

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc    = accuracy_score(y_test, y_pred)
    f1     = f1_score(y_test, y_pred, average="weighted", zero_division=0)

    print("\n  CV F1    :", round(cv.mean(), 3), "+-", round(cv.std(), 3))
    print("  Test Acc :", round(acc, 3))
    print("  Test F1  :", round(f1, 3))

    classes      = sorted(list(set(y)))
    target_names = [LABEL_NAMES[c] for c in classes]
    print(classification_report(
        y_test, y_pred,
        labels=classes,
        target_names=target_names,
        zero_division=0
    ))

    joblib.dump(model, MODEL_DIR / (asana + "_model.pkl"))
    print("  Saved: models/" + asana + "_model.pkl")

    results[asana] = {
        "model":    model,
        "accuracy": acc,
        "f1":       f1,
        "y_test":   y_test,
        "y_pred":   y_pred,
        "classes":  classes,
    }

# ── CONFUSION MATRICES ─────────────────────────────────────────
n    = len(results)
cols = min(n, 3)
rows = (n + cols - 1) // cols

fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 5 * rows))
axes = np.array(axes).flatten() if n > 1 else [axes]

for ax, (asana, res) in zip(axes, results.items()):
    classes      = res["classes"]
    target_names = [LABEL_NAMES[c] for c in classes]
    cm     = confusion_matrix(res["y_test"], res["y_pred"], labels=classes)
    cm_pct = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-8) * 100
    sns.heatmap(
        cm_pct, ax=ax, annot=True, fmt=".1f",
        cmap="Blues",
        xticklabels=target_names,
        yticklabels=target_names,
        cbar_kws={"label": "%"},
        linewidths=0.5
    )
    ax.set_title(
        asana.title() + "\nAcc=" + str(round(res["accuracy"], 2)) +
        "  F1=" + str(round(res["f1"], 2)),
        fontsize=11, fontweight="bold"
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")

for ax in axes[len(results):]:
    ax.set_visible(False)

plt.suptitle("Confusion Matrices — 35 Joint Angles", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig(MODEL_DIR / "confusion_matrices.png", dpi=120, bbox_inches="tight")
plt.show()
print("Saved: models/confusion_matrices.png")

# ── FEATURE IMPORTANCE ─────────────────────────────────────────
fig, axes = plt.subplots(rows, cols, figsize=(7 * cols, 6 * rows))
axes = np.array(axes).flatten() if n > 1 else [axes]

for ax, (asana, res) in zip(axes, results.items()):
    clf = res["model"].named_steps["clf"]
    imp = clf.feature_importances_
    idx = np.argsort(imp)[::-1][:12]
    ax.barh(
        [FEATURE_NAMES[i].replace("_", " ").title() for i in idx[::-1]],
        imp[idx[::-1]],
        color="#3498db", edgecolor="white"
    )
    ax.set_title(asana.title() + " — Top 12 Joints", fontweight="bold")
    ax.set_xlabel("Importance Score")
    ax.grid(axis="x", alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)

for ax in axes[len(results):]:
    ax.set_visible(False)

plt.suptitle("Which Joints Matter Most?", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig(MODEL_DIR / "feature_importance.png", dpi=120, bbox_inches="tight")
plt.show()
print("Saved: models/feature_importance.png")

# ── FINAL SUMMARY ──────────────────────────────────────────────
print("\n" + "=" * 55)
print("  TRAINING COMPLETE — 35 JOINT ANGLE MODEL")
print("=" * 55)
print(f"  {'Asana':<18} {'Accuracy':>10}  {'F1 Score':>10}")
print("  " + "-" * 42)
for asana, res in results.items():
    print(f"  {asana:<18} {res['accuracy']:>10.4f}  {res['f1']:>10.4f}")
print("=" * 55)
avg_acc = np.mean([r["accuracy"] for r in results.values()])
avg_f1  = np.mean([r["f1"]       for r in results.values()])
print(f"  Average Accuracy : {avg_acc:.4f}  ({avg_acc*100:.1f}%)")
print(f"  Average F1       : {avg_f1:.4f}")
print("=" * 55)
print("\nDone! Run realtime_yoga.py next.")
