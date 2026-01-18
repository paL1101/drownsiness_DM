# src/train_rf.py
import os
import argparse
import joblib
import pandas as pd

from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier

FEATURES = [
    "ear_mean",
    "ear_min",
    "ear_std",
    "perclos",
    "blink_count",
    "avg_closure_ms",
    "max_closure_ms"
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, default="data/drowsiness_features.csv")
    parser.add_argument("--out_model", type=str, default="models/rf_model.pkl")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.out_model), exist_ok=True)

    df = pd.read_csv(args.csv)
    df = df.dropna(subset=FEATURES + ["label"]).copy()

    X = df[FEATURES].values
    y = df["label"].astype(int).values

    # Model dasar
    rf = RandomForestClassifier(
        random_state=42,
        n_jobs=-1,
        class_weight="balanced"
    )

    # Tuning ringan tapi efektif
    param_dist = {
        "n_estimators": [200, 300, 500, 700],
        "max_depth": [None, 8, 12, 16, 20, 30],
        "min_samples_leaf": [1, 2, 4],
        "min_samples_split": [2, 5, 10],
        "max_features": ["sqrt", "log2", None],
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    search = RandomizedSearchCV(
        estimator=rf,
        param_distributions=param_dist,
        n_iter=30,
        scoring="f1",           # fokus seimbang; bisa ganti "recall" untuk kelas ngantuk
        cv=cv,
        verbose=1,
        random_state=42,
        n_jobs=-1
    )
    search.fit(X, y)

    best = search.best_estimator_
    print("\nBest Params:", search.best_params_)
    print("Best CV Score (f1):", search.best_score_)

    # Evaluasi dengan CV (report ringkas menggunakan prediksi fold)
    # (Untuk laporan UAS, ini sudah kuat.)
    y_true_all = []
    y_pred_all = []

    for train_idx, test_idx in cv.split(X, y):
        Xtr, Xte = X[train_idx], X[test_idx]
        ytr, yte = y[train_idx], y[test_idx]
        best.fit(Xtr, ytr)
        pred = best.predict(Xte)

        y_true_all.extend(yte.tolist())
        y_pred_all.extend(pred.tolist())

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_true_all, y_pred_all))
    print("\nClassification Report:")
    print(classification_report(y_true_all, y_pred_all, digits=4))

    # Latih final pada seluruh data dan simpan
    best.fit(X, y)
    joblib.dump(best, args.out_model)
    print(f"\nModel tersimpan di: {args.out_model}")

    # Feature importance (nilai plus untuk laporan)
    importances = best.feature_importances_
    fi = sorted(zip(FEATURES, importances), key=lambda x: x[1], reverse=True)
    print("\nFeature Importance:")
    for name, val in fi:
        print(f"- {name:15s}: {val:.4f}")

if __name__ == "__main__":
    main()
