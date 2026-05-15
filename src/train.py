from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier


ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"
IMAGE_DIR = ROOT_DIR / "images"
MODEL_DIR = ROOT_DIR / "models"
DATA_PATH = DATA_DIR / "sdss_galaxy_data.csv"

RANDOM_STATE = 42
FEATURE_COLUMNS = ["u", "g", "r", "i", "z", "redshift", "u_g", "g_r", "r_i", "i_z"]
TARGET_COLUMN = "class"


def ensure_directories() -> None:
    DATA_DIR.mkdir(exist_ok=True)
    IMAGE_DIR.mkdir(exist_ok=True)
    MODEL_DIR.mkdir(exist_ok=True)


def generate_sdss_like_dataset(path: Path, rows: int = 3000) -> pd.DataFrame:
    """Create a realistic SDSS-like dataset so the project runs without external downloads."""
    rng = np.random.default_rng(RANDOM_STATE)

    labels = rng.choice(["GALAXY", "STAR", "QSO"], size=rows, p=[0.55, 0.30, 0.15])
    data = []

    for label in labels:
        if label == "GALAXY":
            redshift = rng.normal(0.18, 0.08)
            r_mag = rng.normal(17.4, 1.1)
            colors = rng.normal([1.6, 0.85, 0.42, 0.30], [0.25, 0.18, 0.13, 0.11])
        elif label == "STAR":
            redshift = rng.normal(0.02, 0.015)
            r_mag = rng.normal(16.2, 1.0)
            colors = rng.normal([1.1, 0.45, 0.18, 0.10], [0.20, 0.14, 0.10, 0.08])
        else:
            redshift = rng.normal(1.25, 0.45)
            r_mag = rng.normal(18.5, 1.2)
            colors = rng.normal([0.35, 0.18, 0.10, 0.05], [0.22, 0.16, 0.12, 0.09])

        redshift = max(redshift, 0.0)
        u_g, g_r, r_i, i_z = colors
        r = r_mag
        g = r + g_r
        u = g + u_g
        i = r - r_i
        z = i - i_z

        data.append(
            {
                "u": u,
                "g": g,
                "r": r,
                "i": i,
                "z": z,
                "redshift": redshift,
                "class": label,
            }
        )

    df = pd.DataFrame(data)

    # Add a small amount of missingness to demonstrate cleaning.
    feature_cols = ["u", "g", "r", "i", "z", "redshift"]
    for col in feature_cols:
        missing_idx = rng.choice(df.index, size=int(0.01 * rows), replace=False)
        df.loc[missing_idx, col] = np.nan

    df.to_csv(path, index=False)
    return df


def load_dataset() -> pd.DataFrame:
    if DATA_PATH.exists():
        print(f"Loading dataset from {DATA_PATH}")
        return pd.read_csv(DATA_PATH)

    print("No dataset found. Generating synthetic SDSS-like dataset...")
    return generate_sdss_like_dataset(DATA_PATH)


def clean_and_engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = df.drop_duplicates()

    required_columns = {"u", "g", "r", "i", "z", "redshift", TARGET_COLUMN}
    missing_columns = required_columns.difference(df.columns)
    if missing_columns:
        raise ValueError(f"Dataset is missing required columns: {sorted(missing_columns)}")

    df[TARGET_COLUMN] = df[TARGET_COLUMN].astype(str).str.upper().str.strip()

    df["u_g"] = df["u"] - df["g"]
    df["g_r"] = df["g"] - df["r"]
    df["r_i"] = df["r"] - df["i"]
    df["i_z"] = df["i"] - df["z"]
    return df


def save_eda_plots(df: pd.DataFrame) -> None:
    sns.set_theme(style="whitegrid", palette="Set2")

    plt.figure(figsize=(8, 5))
    sns.countplot(data=df, x=TARGET_COLUMN, order=df[TARGET_COLUMN].value_counts().index)
    plt.title("Class Distribution")
    plt.xlabel("Astronomical Object Class")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(IMAGE_DIR / "class_distribution.png", dpi=150)
    plt.close()

    plt.figure(figsize=(9, 6))
    sns.scatterplot(data=df, x="g_r", y="redshift", hue=TARGET_COLUMN, alpha=0.65)
    plt.title("Color Index g-r vs Redshift")
    plt.xlabel("g-r Color Index")
    plt.ylabel("Redshift")
    plt.tight_layout()
    plt.savefig(IMAGE_DIR / "color_redshift_scatter.png", dpi=150)
    plt.close()

    plt.figure(figsize=(10, 7))
    corr = df[FEATURE_COLUMNS].corr(numeric_only=True)
    sns.heatmap(corr, cmap="coolwarm", annot=True, fmt=".2f", linewidths=0.5)
    plt.title("Feature Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(IMAGE_DIR / "correlation_heatmap.png", dpi=150)
    plt.close()


def build_models() -> dict[str, Pipeline]:
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[("numeric", numeric_transformer, FEATURE_COLUMNS)]
    )

    return {
        "Logistic Regression": Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                (
                    "model",
                    LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
                ),
            ]
        ),
        "Decision Tree": Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("model", DecisionTreeClassifier(max_depth=8, random_state=RANDOM_STATE)),
            ]
        ),
        "Random Forest": Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                (
                    "model",
                    RandomForestClassifier(
                        n_estimators=200,
                        max_depth=12,
                        random_state=RANDOM_STATE,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
    }


def save_model_comparison(results: dict[str, float]) -> None:
    comparison = pd.DataFrame(
        {"model": list(results.keys()), "accuracy": list(results.values())}
    ).sort_values("accuracy", ascending=False)

    comparison.to_csv(MODEL_DIR / "model_comparison.csv", index=False)

    plt.figure(figsize=(8, 5))
    sns.barplot(data=comparison, x="accuracy", y="model", palette="viridis", hue="model", legend=False)
    plt.xlim(0, 1)
    plt.title("Model Accuracy Comparison")
    plt.xlabel("Test Accuracy")
    plt.ylabel("Model")
    plt.tight_layout()
    plt.savefig(IMAGE_DIR / "model_accuracy_comparison.png", dpi=150)
    plt.close()


def save_confusion_matrix(model: Pipeline, x_test: pd.DataFrame, y_test: pd.Series) -> None:
    plt.figure(figsize=(7, 6))
    ConfusionMatrixDisplay.from_estimator(
        model,
        x_test,
        y_test,
        cmap="Blues",
        values_format="d",
    )
    plt.title("Confusion Matrix - Best Model")
    plt.tight_layout()
    plt.savefig(IMAGE_DIR / "confusion_matrix.png", dpi=150)
    plt.close()


def save_feature_importance(model: Pipeline) -> None:
    classifier = model.named_steps["model"]
    if not hasattr(classifier, "feature_importances_"):
        return

    importances = pd.DataFrame(
        {
            "feature": FEATURE_COLUMNS,
            "importance": classifier.feature_importances_,
        }
    ).sort_values("importance", ascending=False)

    importances.to_csv(MODEL_DIR / "feature_importance.csv", index=False)

    plt.figure(figsize=(8, 5))
    sns.barplot(data=importances, x="importance", y="feature", palette="mako", hue="feature", legend=False)
    plt.title("Feature Importance - Best Model")
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.tight_layout()
    plt.savefig(IMAGE_DIR / "feature_importance.png", dpi=150)
    plt.close()


def main() -> None:
    ensure_directories()
    df = load_dataset()
    df = clean_and_engineer_features(df)

    print("\nDataset preview:")
    print(df.head())
    print("\nMissing values after loading:")
    print(df[FEATURE_COLUMNS + [TARGET_COLUMN]].isna().sum())

    save_eda_plots(df)

    x = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN]
    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    models = build_models()
    results = {}
    trained_models = {}

    for name, model in models.items():
        print(f"\nTraining {name}...")
        model.fit(x_train, y_train)
        predictions = model.predict(x_test)
        accuracy = accuracy_score(y_test, predictions)
        results[name] = accuracy
        trained_models[name] = model
        print(f"{name} accuracy: {accuracy:.4f}")
        print(classification_report(y_test, predictions))

    best_model_name = max(results, key=results.get)
    best_model = trained_models[best_model_name]
    print(f"\nBest model: {best_model_name} ({results[best_model_name]:.4f})")

    save_model_comparison(results)
    save_confusion_matrix(best_model, x_test, y_test)
    save_feature_importance(best_model)

    model_path = MODEL_DIR / "best_galaxy_classifier.joblib"
    joblib.dump(best_model, model_path)
    print(f"\nSaved best model to {model_path}")
    print(f"Saved plots to {IMAGE_DIR}")
    print(f"Saved metrics to {MODEL_DIR}")


if __name__ == "__main__":
    main()
