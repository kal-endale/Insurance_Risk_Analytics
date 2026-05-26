import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix, mean_absolute_error, mean_squared_error, r2_score
import numpy as np

def train_evaluate_classification_model(X_train, y_train, X_test, y_test, model_type='LogisticRegression', random_state=42):
    """
    Trains and evaluates a classification model.

    Args:
        X_train, y_train, X_test, y_test: Training and testing data.
        model_type (str): Type of classification model ('LogisticRegression', 'RandomForest', 'GradientBoosting').
        random_state (int): Random seed for reproducibility.

    Returns:
        dict: Evaluation metrics and the trained model.
    """
    print(f"--- Training {model_type} Classification Model ---")
    model = None
    if model_type == 'LogisticRegression':
        model = LogisticRegression(random_state=random_state, solver='liblinear', class_weight='balanced')
    elif model_type == 'RandomForest':
        model = RandomForestClassifier(random_state=random_state, class_weight='balanced')
    elif model_type == 'GradientBoosting':
        model = GradientBoostingClassifier(random_state=random_state)
    else:
        raise ValueError(f"Unsupported classification model_type: {model_type}")

    # Ensure feature matrices are numeric: coerce non-numeric to NaN and fill missing values
    try:
        X_train = X_train.apply(pd.to_numeric, errors='coerce').fillna(0)
        X_test = X_test.apply(pd.to_numeric, errors='coerce').fillna(0)
    except Exception:
        # If X_train/X_test are numpy arrays, convert directly
        import numpy as _np
        X_train = _np.asarray(X_train, dtype=float)
        X_test = _np.asarray(X_test, dtype=float)

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    report = classification_report(y_test, y_pred, output_dict=True)
    roc_auc = roc_auc_score(y_test, y_proba)
    conf_mat = confusion_matrix(y_test, y_pred)

    print(f"\n--- Evaluation for {model_type} ---")
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    print(f"ROC AUC Score: {roc_auc:.4f}")

    return {
        'model': model,
        'classification_report': report,
        'roc_auc_score': roc_auc,
        'confusion_matrix': conf_mat
    }

def train_evaluate_regression_model(X_train, y_train, X_test, y_test, model_type='RandomForestRegressor', random_state=42):
    """
    Trains and evaluates a regression model.

    Args:
        X_train, y_train, X_test, y_test: Training and testing data.
        model_type (str): Type of regression model ('RandomForestRegressor', 'GradientBoostingRegressor').
        random_state (int): Random seed for reproducibility.

    Returns:
        dict: Evaluation metrics and the trained model.
    """
    print(f"--- Training {model_type} Regression Model ---")
    model = None
    if model_type == 'RandomForestRegressor':
        model = RandomForestRegressor(random_state=random_state, n_jobs=-1)
    elif model_type == 'GradientBoostingRegressor':
        model = GradientBoostingRegressor(random_state=random_state)
    else:
        raise ValueError(f"Unsupported regression model_type: {model_type}")

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    print(f"\n--- Evaluation for {model_type} ---")
    print(f"Mean Absolute Error (MAE): {mae:.2f}")
    print(f"Mean Squared Error (MSE): {mse:.2f}")
    print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
    print(f"R-squared (R2) Score: {r2:.4f}")

    return {
        'model': model,
        'mae': mae,
        'mse': mse,
        'rmse': rmse,
        'r2_score': r2
    }
