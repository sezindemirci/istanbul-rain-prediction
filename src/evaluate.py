import pandas as pd
import joblib

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)


def evaluate_best_model(input_path="data/weather_processed.csv"):
    df = pd.read_csv(input_path)
    df["time"] = pd.to_datetime(df["time"])

    model = joblib.load("models/best_model.pkl")
    feature_columns = joblib.load("models/feature_columns.pkl")

    test_df = df[df["time"] >= "2026-01-01"]

    X_test = test_df[feature_columns]
    y_test = test_df["rain_tomorrow"]

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    results = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1_score": f1_score(y_test, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, y_prob)
    }

    results_df = pd.DataFrame([results])
    results_df.to_csv("models/test_results.csv", index=False)

    cm = confusion_matrix(y_test, y_pred)
    cm_df = pd.DataFrame(
        cm,
        columns=["Predicted No Rain", "Predicted Rain"],
        index=["Actual No Rain", "Actual Rain"]
    )
    cm_df.to_csv("models/confusion_matrix.csv")

    report = classification_report(y_test, y_pred, zero_division=0)
    with open("models/classification_report.txt", "w") as f:
        f.write(report)

    print("Test evaluation completed.")
    print("Test results:")
    print(results_df)
    print("\nConfusion matrix:")
    print(cm_df)
    print("\nClassification report:")
    print(report)

    return results_df


if __name__ == "__main__":
    evaluate_best_model()
 
