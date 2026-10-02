"""Train & evaluate models for spiral and wave drawings.  Run:  python train.py"""
import json
from pathlib import Path

import cv2
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import GroupKFold, StratifiedKFold, cross_val_predict, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from common import CLASSES, SIZES, ensure_data, features, preprocess

ROOT = Path(__file__).resolve().parent
DATA, OUT = ROOT / "data", ROOT / "outputs"
OUT.mkdir(exist_ok=True)
ensure_data(ROOT)
SEED = 42


def load(kind, split):
    X, y, ids = [], [], []
    for lab, cls in enumerate(CLASSES):
        for p in sorted((DATA / kind / split / cls).glob("*.png")):
            X.append(features(preprocess(p, SIZES[kind])))
            y.append(lab)
            ids.append(p.name[:3])  # subject id, e.g. V01
    return np.array(X), np.array(y), np.array(ids)


def models():
    return {
        "SVM (RBF)": make_pipeline(StandardScaler(), SVC(C=10, probability=True, random_state=SEED)),
        "Logistic Regression": make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=5000)),
        "Random Forest": RandomForestClassifier(n_estimators=400, random_state=SEED),
        "Gradient Boosting": GradientBoostingClassifier(random_state=SEED),
    }


def fit_best(kind):
    """Quickly train the chosen best model for one kind and save it (used when no saved model fits)."""
    try:
        name = json.load(open(OUT / "best_models.json"))[kind]
    except Exception:
        name = "SVM (RBF)"
    Xtr, ytr, _ = load(kind, "training")
    Xte, yte, _ = load(kind, "testing")
    m = models()[name].fit(np.vstack([Xtr, Xte]), np.concatenate([ytr, yte]))
    joblib.dump(m, OUT / f"best_{kind}.joblib")
    return m


def save_samples():
    fig, ax = plt.subplots(2, 4, figsize=(14, 6))
    for r, kind in enumerate(SIZES):
        for c, cls in enumerate(CLASSES):
            for k, p in enumerate(sorted((DATA / kind / "training" / cls).glob("*.png"))[:2]):
                a = ax[r, c * 2 + k]
                a.imshow(cv2.imread(str(p), 0), cmap="gray")
                a.set_title(f"{kind} - {cls}")
                a.axis("off")
    plt.tight_layout()
    plt.savefig(OUT / "samples.png", dpi=100)
    plt.close()


def main():
    save_samples()
    rows, best_names = [], {}
    for kind in SIZES:
        Xtr, ytr, gtr = load(kind, "training")
        Xte, yte, gte = load(kind, "testing")
        Xall, yall, gall = np.vstack([Xtr, Xte]), np.concatenate([ytr, yte]), np.concatenate([gtr, gte])
        print(f"\n=== {kind.upper()}  train={len(ytr)}  test={len(yte)} ===")
        best_cv, best = -1, None
        for name, m in models().items():
            cv = cross_val_score(m, Xtr, ytr, cv=StratifiedKFold(5, shuffle=True, random_state=SEED)).mean()
            m.fit(Xtr, ytr)
            pred = m.predict(Xte)
            subj = accuracy_score(yall, cross_val_predict(models()[name], Xall, yall, groups=gall, cv=GroupKFold(5)))
            r = dict(dataset=kind, model=name, cv_acc=cv, test_acc=accuracy_score(yte, pred),
                     f1=f1_score(yte, pred), auc=roc_auc_score(yte, m.predict_proba(Xte)[:, 1]),
                     subject_wise_acc=subj)
            rows.append(r)
            print(f"{name:20s} test acc {r['test_acc']:.3f} | f1 {r['f1']:.3f} | auc {r['auc']:.3f} | subject-wise {subj:.3f}")
            if cv > best_cv:  # choose by CV, not by test set
                best_cv, best = cv, (name, m, pred)
        best_names[kind] = best[0]
        joblib.dump(best[1], OUT / f"best_{kind}.joblib")
        ConfusionMatrixDisplay.from_predictions(yte, best[2], display_labels=CLASSES, cmap="Blues")
        plt.title(f"{kind} - {best[0]}")
        plt.savefig(OUT / f"confusion_{kind}.png", dpi=100)
        plt.close()

    df = pd.DataFrame(rows).round(3)
    df.to_csv(OUT / "results.csv", index=False)
    df.pivot(index="model", columns="dataset", values="test_acc").plot.bar(figsize=(8, 5), rot=15)
    plt.ylim(0, 1); plt.ylabel("Test accuracy"); plt.title("Model comparison (test set)")
    plt.tight_layout(); plt.savefig(OUT / "comparison.png", dpi=100); plt.close()
    json.dump(best_names, open(OUT / "best_models.json", "w"), indent=2)
    print("\nBest models (by CV):", best_names)
    print("DONE. ফলাফল 'outputs' ফোল্ডারে সেভ হয়েছে।")


if __name__ == "__main__":
    main()
