import os
import joblib
import pandas as pd

from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score
)

from src.feature_engineering import get_feature_columns


def train_and_evaluate_models(input_path="data/weather_processed.csv"):
    """
    Train and compare multiple classification models to predict rain tomorrow.

    Split strategy: time-based (no random shuffle) to prevent data leakage.
        - Train:      2015-01-01 → 2022-12-31  (~8 years)
        - Validation: 2023-01-01 → 2024-12-31  (~2 years)
        - Test:       2025-01-01 → present      (~1.5 years, used in evaluate.py)

    Model selection criterion: ROC-AUC on validation set.
    ROC-AUC is preferred over F1 for imbalanced binary classification
    because it evaluates ranking quality across all thresholds.
    """

    df = pd.read_csv(input_path)
    df["time"] = pd.to_datetime(df["time"])

    # Feature listesini feature_engineering.py'dan al
    feature_columns = get_feature_columns()
    # API'den gelmeyen sütunlar varsa sessizce atla
    feature_columns = [c for c in feature_columns if c in df.columns]

    target_column = "rain_tomorrow"

    # --- Time-based split ---
    train_df      = df[df["time"] < "2023-01-01"]
    validation_df = df[(df["time"] >= "2023-01-01") & (df["time"] < "2025-01-01")]
    test_df       = df[df["time"] >= "2025-01-01"]

    X_train = train_df[feature_columns]
    y_train = train_df[target_column]

    X_val = validation_df[feature_columns]
    y_val = validation_df[target_column]

    X_test = test_df[feature_columns]
    y_test = test_df[target_column]

    print("=== Split özeti ===")
    print(f"Train:      {len(X_train)} satır — yağmur oranı: {y_train.mean():.1%}")
    print(f"Validation: {len(X_val)} satır — yağmur oranı: {y_val.mean():.1%}")
    print(f"Test:       {len(X_test)} satır — yağmur oranı: {y_test.mean():.1%}")

    # --- Model tanımları ---
    # Logistic Regression için StandardScaler pipeline içinde uygulanıyor.
    # Tree-based modeller (RF, GB) ölçeklemeye ihtiyaç duymaz.
    # class_weight="balanced": yağmur sınıfı azınlıkta olduğunda ceza ağırlığını dengeler.
    models = {
        "Baseline (majority class)": DummyClassifier(
            strategy="most_frequent"
        ),

        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(
                max_iter=1000,
                class_weight="balanced",
                C=1.0,
                random_state=42
            ))
        ]),

        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            min_samples_leaf=20,
            class_weight="balanced",
            random_state=42
        ),

        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.05,
            random_state=42
        ),
    }

    results = []
    os.makedirs("models", exist_ok=True)

    best_model = None
    best_model_name = None
    best_roc_auc = -1

    print("\n=== Model eğitimi ===")

    for name, model in models.items():
        model.fit(X_train, y_train)

        y_pred = model.predict(X_val)

        if hasattr(model, "predict_proba"):
            y_prob = model.predict_proba(X_val)[:, 1]
            roc_auc = roc_auc_score(y_val, y_prob)
        else:
            y_prob = None
            roc_auc = None

        result = {
            "model":     name,
            "accuracy":  round(accuracy_score(y_val, y_pred), 4),
            "precision": round(precision_score(y_val, y_pred, zero_division=0), 4),
            "recall":    round(recall_score(y_val, y_pred, zero_division=0), 4),
            "f1_score":  round(f1_score(y_val, y_pred, zero_division=0), 4),
            "roc_auc":   round(roc_auc, 4) if roc_auc is not None else None,
        }
        results.append(result)
        print(f"  {name}: ROC-AUC={result['roc_auc']}, F1={result['f1_score']}")

        # En iyi modeli ROC-AUC'a göre seç
        if roc_auc is not None and roc_auc > best_roc_auc:
            best_roc_auc = roc_auc
            best_model = model
            best_model_name = name

    # --- Sonuçları kaydet ---
    results_df = pd.DataFrame(results)
    results_df.to_csv("models/model_validation_results.csv", index=False)

    # --- En iyi modeli kaydet ---
    joblib.dump(best_model, "models/best_model.pkl")
    joblib.dump(feature_columns, "models/feature_columns.pkl")

    # --- Feature importance (sadece tree-based modeller) ---
    _save_feature_importance(best_model, best_model_name, feature_columns)

    print("\n=== Validation sonuçları ===")
    print(results_df.to_string(index=False))
    print(f"\nEn iyi model: {best_model_name} (ROC-AUC: {best_roc_auc:.4f})")
    print("Modeller kaydedildi: models/")

    return results_df


def _save_feature_importance(model, model_name, feature_columns):
    """
    Tree-based modeller için feature importance çıkarır ve CSV'ye kaydeder.
    Pipeline içindeki modeller için estimator'a erişir.
    """
    try:
        # Pipeline ise clf adımına bak
        if hasattr(model, "named_steps"):
            estimator = model.named_steps.get("clf", None)
        else:
            estimator = model

        if estimator is None or not hasattr(estimator, "feature_importances_"):
            return

        importance_df = pd.DataFrame({
            "feature":    feature_columns,
            "importance": estimator.feature_importances_
        }).sort_values("importance", ascending=False)

        importance_df.to_csv("models/feature_importance.csv", index=False)
        print(f"\nFeature importance kaydedildi (models/feature_importance.csv)")
        print(importance_df.head(10).to_string(index=False))

    except Exception as e:
        print(f"Feature importance kaydedilemedi: {e}")


if __name__ == "__main__":
    train_and_evaluate_models()