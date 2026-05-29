import joblib
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    precision_recall_curve,
    roc_curve
)


def evaluate_best_model(input_path="data/weather_processed.csv"):
    """
    Evaluate the best saved model on the held-out test set.

    Test set: 2025-01-01 onwards — data the model has never seen during
    training or validation. This split matches train_model.py exactly.

    Saves the following artifacts to models/:
        - test_results.csv          : scalar metrics
        - confusion_matrix.csv      : confusion matrix
        - classification_report.txt : per-class precision/recall/F1
        - roc_curve.csv             : FPR/TPR pairs for ROC plot in dashboard
        - pr_curve.csv              : Precision/Recall pairs for PR plot
    """

    # --- Veri ve model yükleme ---
    df = pd.read_csv(input_path)
    df["time"] = pd.to_datetime(df["time"])

    model = joblib.load("models/best_model.pkl")
    feature_columns = joblib.load("models/feature_columns.pkl")

    # --- Test seti: train_model.py ile aynı tarih sınırı ---
    test_df = df[df["time"] >= "2025-01-01"].copy()

    if len(test_df) == 0:
        raise ValueError("Test seti boş. 'end_date' ve split tarihlerini kontrol et.")

    # Eksik feature kontrolü
    missing_features = [c for c in feature_columns if c not in test_df.columns]
    if missing_features:
        raise ValueError(f"Test setinde eksik feature'lar: {missing_features}")

    X_test = test_df[feature_columns]
    y_test = test_df["rain_tomorrow"]

    print(f"Test seti: {len(test_df)} satır")
    print(f"Test yağmur oranı: {y_test.mean():.1%}")

    # --- Tahminler ---
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    # --- Scalar metrikler ---
    results = {
        "accuracy":  round(accuracy_score(y_test, y_pred), 4),
        "precision": round(precision_score(y_test, y_pred, zero_division=0), 4),
        "recall":    round(recall_score(y_test, y_pred, zero_division=0), 4),
        "f1_score":  round(f1_score(y_test, y_pred, zero_division=0), 4),
        "roc_auc":   round(roc_auc_score(y_test, y_prob), 4),
    }

    results_df = pd.DataFrame([results])
    results_df.to_csv("models/test_results.csv", index=False)

    # --- Confusion matrix ---
    cm = confusion_matrix(y_test, y_pred)
    cm_df = pd.DataFrame(
        cm,
        columns=["Predicted: No Rain", "Predicted: Rain"],
        index=["Actual: No Rain", "Actual: Rain"]
    )
    cm_df.to_csv("models/confusion_matrix.csv")

    # --- Classification report ---
    report = classification_report(
        y_test, y_pred,
        target_names=["No Rain", "Rain"],
        zero_division=0
    )
    with open("models/classification_report.txt", "w") as f:
        f.write(report)

    # --- ROC curve (dashboard'da çizmek için) ---
    fpr, tpr, roc_thresholds = roc_curve(y_test, y_prob)
    roc_df = pd.DataFrame({
        "fpr":       fpr,
        "tpr":       tpr,
        "threshold": roc_thresholds
    })
    roc_df.to_csv("models/roc_curve.csv", index=False)

    # --- Precision-Recall curve (imbalanced sınıflar için ek görsel) ---
    precision_vals, recall_vals, pr_thresholds = precision_recall_curve(y_test, y_prob)
    pr_df = pd.DataFrame({
        "precision": precision_vals,
        "recall":    recall_vals,
        "threshold": list(pr_thresholds) + [1.0]  # precision_recall_curve n+1 döndürür
    })
    pr_df.to_csv("models/pr_curve.csv", index=False)

    # --- Konsol çıktısı ---
    print("\n=== Test sonuçları ===")
    print(results_df.to_string(index=False))

    print("\n=== Confusion matrix ===")
    print(cm_df)

    tn, fp, fn, tp = cm.ravel()
    print(f"\n  True Negative  (doğru: yağmur yok): {tn}")
    print(f"  False Positive (yanlış alarm):       {fp}")
    print(f"  False Negative (kaçırılan yağmur):   {fn}")
    print(f"  True Positive  (doğru: yağmur var):  {tp}")

    print("\n=== Classification report ===")
    print(report)

    print("Tüm sonuçlar models/ klasörüne kaydedildi.")

    return results_df


if __name__ == "__main__":
    evaluate_best_model()