
import pandas as pd
import numpy as np
import cv2
import io

from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report
)


# ==============================
# TABULAR DATA ANALYSIS
# ==============================

def analyze_tabular(df):

    results = {}

    rows, columns = df.shape

    results["rows"] = rows
    results["columns"] = columns
    results["missing_values"] = int(df.isnull().sum().sum())
    results["duplicate_rows"] = int(df.duplicated().sum())

    results["missing_by_column"] = (
        df.isnull().sum().to_dict()
    )

    results["column_types"] = (
        df.dtypes.astype(str).to_dict()
    )

    results["outliers"] = {}

    numeric = df.select_dtypes(include=np.number)

    for column in numeric.columns:

        q1 = numeric[column].quantile(0.25)
        q3 = numeric[column].quantile(0.75)
        iqr = q3 - q1

        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr

        count = int(
            ((numeric[column] < lower) |
             (numeric[column] > upper)).sum()
        )

        results["outliers"][column] = count

    return results


# ==============================
# IMAGE DATA ANALYSIS
# ==============================

def analyze_image(uploaded_file):

    image_bytes = uploaded_file.getvalue()

    image = Image.open(
        io.BytesIO(image_bytes)
    ).convert("RGB")

    image_array = np.array(image)

    gray = cv2.cvtColor(
        image_array,
        cv2.COLOR_RGB2GRAY
    )

    height, width = gray.shape

    blur_score = float(
        cv2.Laplacian(gray, cv2.CV_64F).var()
    )

    brightness = float(np.mean(gray))
    contrast = float(np.std(gray))

    return {
        "image": uploaded_file.name,
        "width": width,
        "height": height,
        "blur_score": round(blur_score, 2),
        "brightness": round(brightness, 2),
        "contrast": round(contrast, 2),
        "possible_blur": blur_score < 100,
        "low_resolution": width < 100 or height < 100
    }


# ==============================
# CLASSIFICATION EVALUATION
# ==============================

def evaluate_classifier(df, target_column):

    if target_column not in df.columns:
        raise ValueError("Target column not found.")

    data = df.dropna(subset=[target_column]).copy()

    if data[target_column].nunique() < 2:
        raise ValueError("Target needs at least two classes.")

    X = data.drop(columns=[target_column])
    y = data[target_column]

    # Remove constant columns
    X = X.loc[:, X.nunique(dropna=False) > 1]

    if X.empty:
        raise ValueError("No usable feature columns.")

    numeric_features = X.select_dtypes(
        include=np.number
    ).columns.tolist()

    categorical_features = X.select_dtypes(
        exclude=np.number
    ).columns.tolist()

    transformers = []

    if numeric_features:
        numeric_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="median"))
        ])

        transformers.append((
            "numeric",
            numeric_pipeline,
            numeric_features
        ))

    if categorical_features:
        categorical_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore"))
        ])

        transformers.append((
            "categorical",
            categorical_pipeline,
            categorical_features
        ))

    preprocessor = ColumnTransformer(transformers)

    model = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            class_weight="balanced"
        ))
    ])

    stratify = y if y.value_counts().min() >= 2 else None

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=stratify
    )

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    labels = sorted(y.unique(), key=str)

    cm = confusion_matrix(
        y_test,
        predictions,
        labels=labels
    )

    return {
        "accuracy": float(accuracy_score(y_test, predictions)),
        "precision": float(precision_score(
            y_test, predictions,
            average="weighted", zero_division=0
        )),
        "recall": float(recall_score(
            y_test, predictions,
            average="weighted", zero_division=0
        )),
        "f1_score": float(f1_score(
            y_test, predictions,
            average="weighted", zero_division=0
        )),
        "confusion_matrix": cm.tolist(),
        "labels": [str(x) for x in labels],
        "classification_report": classification_report(
            y_test, predictions, zero_division=0
        )
    }


# ==============================
# DEFECT DETECTION EVALUATION
# ==============================

def evaluate_defects(actual, predicted):

    actual = pd.Series(actual)
    predicted = pd.Series(predicted)

    if len(actual) != len(predicted):
        raise ValueError("Actual and predicted lengths must match.")

    if actual.isna().any() or predicted.isna().any():
        raise ValueError("Labels cannot contain missing values.")

    if not set(actual.unique()).issubset({0, 1}):
        raise ValueError("Actual labels must be 0 or 1.")

    if not set(predicted.unique()).issubset({0, 1}):
        raise ValueError("Predicted labels must be 0 or 1.")

    cm = confusion_matrix(
        actual,
        predicted,
        labels=[0, 1]
    )

    return {
        "accuracy": float(accuracy_score(actual, predicted)),
        "precision": float(precision_score(
            actual, predicted, zero_division=0
        )),
        "recall": float(recall_score(
            actual, predicted, zero_division=0
        )),
        "f1_score": float(f1_score(
            actual, predicted, zero_division=0
        )),
        "confusion_matrix": cm.tolist(),
        "labels": ["Normal", "Defect"]
    }
