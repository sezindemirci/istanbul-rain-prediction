import os
import joblib
import pandas as pd

from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score


def train_and_evaluate_models(input_path="data/weather_processed.csv"):
    df = pd.read_csv(input_path)
    df["time"] = pd.to_datetime(df["time"])

    feature_columns = [
        "temperature_2m_max",
        "temperature_2m_min",
        "temperature_2m_mean",
        "precipitation_sum",
        "wind_speed_10m_max",
        "month",
        "day_of_year",
        "precipitation_yesterday",
        "temp_mean_yesterday",
        "wind_speed_yesterday",
        "precipitation_3day_avg",
        "precipitation_7day_avg",
        "temperature_7day_avg"
    ]

    target_column = "rain_tomorrow"

    # Time-based split to avoid data leakage
    train_df = df[df["time"] < "2025-01-01"]
    validation_df = df[(df["time"] >= "2025-01-01") & (df["time"] < "2026-01-01")]
    test_df = df[df["time"] >= "2026-01-01"]

    X_train = train_df[feature_columns]
    y_train = train_df[target_column]

    X_val = validation_df[feature_columns]
    y_val = validation_df[target_column]

    X_test = test_df[feature_columns]
    y_test = test_df[target_column]

    models = {
        "Baseline Majority Class": DummyClassifier(strategy="most_frequent"),
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Random Forest": RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            class_weight="balanced"
        )
    }

    results = []

    os.makedirs("models", exist_ok=True)

    best_model = None
    best_model_name = None
    best_f1 = -1

    for model_name, model in models.items():
        model.fit(X_train, y_train)

        y_pred = model.predict(X_val)

        if hasattr(model, "predict_proba"):
            y_prob = model.predict_proba(X_val)[:, 1]
            roc_auc = roc_auc_score(y_val, y_prob)
        else:
            roc_auc = None

        accuracy = accuracy_score(y_val, y_pred)
        precision = precision_score(y_val, y_pred, zero_division=0)
        recall = recall_score(y_val, y_pred, zero_division=0)
        f1 = f1_score(y_val, y_pred, zero_division=0)

        results.append({
            "model": model_name,
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "roc_auc": roc_auc
        })

        if f1 > best_f1:
            best_f1 = f1
            best_model = model
            best_model_name = model_name

    results_df = pd.DataFrame(results)
    results_df.to_csv("models/model_validation_results.csv", index=False)

    joblib.dump(best_model, "models/best_model.pkl")
    joblib.dump(feature_columns, "models/feature_columns.pkl")

    print("Model training completed.")
    print("Validation results:")
    print(results_df)
    print(f"Best model: {best_model_name}")

    return results_df


if __name__ == "__main__":
    train_and_evaluate_models()
 
