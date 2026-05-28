import pandas as pd
import numpy as np
import shap
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error,r2_score
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt


def remove_high_missing_columns(df, threshold=0.5):
    """
    Remove columns with missing values above threshold.
    """
    missing_ratio = df.isnull().mean()

    cols_to_drop = missing_ratio[missing_ratio > threshold].index

    df = df.drop(columns=cols_to_drop)

    return df


def separate_features(df, target_column=None):
    """
    Separate numerical and categorical columns.
    """
    numerical_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()

    categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

    if target_column:
        if target_column in numerical_cols:
            numerical_cols.remove(target_column)

        if target_column in categorical_cols:
            categorical_cols.remove(target_column)

    return numerical_cols, categorical_cols


def create_preprocessor(numerical_cols, categorical_cols):
    """
    Create preprocessing pipeline.
    """

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median"))
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore"))
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numerical_cols),
            ("cat", categorical_transformer, categorical_cols)
        ]
    )

    return preprocessor

def calculate_vehicle_age(df, vehicle_year_column="VehicleYear"):
    """
    Create vehicle age feature.
    """

    current_year = pd.Timestamp.now().year

    df["VehicleAge"] = current_year - df[vehicle_year_column]

    return df


def calculate_policy_duration(
    df,
    start_date_column="PolicyStartDate",
    end_date_column="PolicyEndDate"
):
    """
    Create policy duration feature.
    """

    df[start_date_column] = pd.to_datetime(df[start_date_column])
    df[end_date_column] = pd.to_datetime(df[end_date_column])

    df["PolicyDuration"] = (
        df[end_date_column] - df[start_date_column]
    ).dt.days

    return df


def create_claim_indicator(df, claims_column="TotalClaims"):
    """
    Binary classification target:
    1 -> claim exists
    0 -> no claim
    """

    df["HasClaim"] = np.where(df[claims_column] > 0, 1, 0)

    return df


def feature_engineering(df):
    """
    Run all feature engineering steps.
    """

    if "VehicleYear" in df.columns:
        df = calculate_vehicle_age(df)

    if "PolicyStartDate" in df.columns and "PolicyEndDate" in df.columns:
        df = calculate_policy_duration(df)

    if "TotalClaims" in df.columns:
        df = create_claim_indicator(df)

    return df

def split_data(X,y,test_size=0.2,random_state=42):
    """
    Split dataset into train and test sets.
    """

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state
    )

    return X_train, X_test, y_train, y_test


# ======================================================
# Linear Regression
# ======================================================

def train_linear_regression(X_train, y_train):
    """
    Train Linear Regression model.
    """

    model = LinearRegression()

    model.fit(X_train, y_train)

    return model


# ======================================================
# Random Forest
# ======================================================

def train_random_forest(
    X_train,
    y_train,
    tune_hyperparameters=True
):
    """
    Train Random Forest model with optional tuning.
    """

    rf_model = RandomForestRegressor(
        random_state=42
    )

    if tune_hyperparameters:

        param_grid = {
            "n_estimators": [100, 200],
            "max_depth": [10, 20, None],
            "min_samples_split": [2, 5],
            "min_samples_leaf": [1, 2]
        }

        grid_search = GridSearchCV(
            estimator=rf_model,
            param_grid=param_grid,
            cv=3,
            scoring="neg_root_mean_squared_error",
            n_jobs=-1,
            verbose=1
        )

        grid_search.fit(X_train, y_train)

        best_model = grid_search.best_estimator_

        print("Best Random Forest Params:")
        print(grid_search.best_params_)

        return best_model

    else:

        rf_model.fit(X_train, y_train)

        return rf_model


# ======================================================
# XGBoost
# ======================================================

def train_xgboost(
    X_train,
    y_train,
    tune_hyperparameters=True
):
    """
    Train XGBoost Regressor with optional tuning.
    """

    xgb_model = XGBRegressor(
        objective="reg:squarederror",
        random_state=42
    )

    if tune_hyperparameters:

        param_grid = {
            "n_estimators": [100, 200],
            "learning_rate": [0.01, 0.1],
            "max_depth": [3, 6],
            "subsample": [0.8, 1.0],
            "colsample_bytree": [0.8, 1.0]
        }

        grid_search = GridSearchCV(
            estimator=xgb_model,
            param_grid=param_grid,
            cv=3,
            scoring="neg_root_mean_squared_error",
            n_jobs=-1,
            verbose=1
        )

        grid_search.fit(X_train, y_train)

        best_model = grid_search.best_estimator_

        print("Best XGBoost Params:")
        print(grid_search.best_params_)

        return best_model

    else:

        xgb_model.fit(X_train, y_train)

        return xgb_model

def evaluate_regression_model(
    model,
    X_test,
    y_test,
    model_name="Model"
):
    """
    Evaluate regression model performance.
    """

    predictions = model.predict(X_test)

    rmse = np.sqrt(
        mean_squared_error(y_test, predictions)
    )

    r2 = r2_score(y_test, predictions)

    print(f"\n{model_name} Performance")
    print("-" * 40)
    print(f"RMSE: {rmse:.4f}")
    print(f"R² Score: {r2:.4f}")

    return {
        "model": model_name,
        "rmse": rmse,
        "r2": r2
    }


def get_feature_names(preprocessor, numerical_cols, categorical_cols):
    """
    Extract transformed feature names from preprocessor.
    """

    cat_encoder = (
        preprocessor.named_transformers_["cat"]
        .named_steps["encoder"]
    )

    encoded_cat_features = cat_encoder.get_feature_names_out(
        categorical_cols
    )

    all_features = (
        numerical_cols +
        list(encoded_cat_features)
    )

    return all_features


# ======================================================
# Compute SHAP Values
# ======================================================

def compute_shap_values(model, X_input, feature_names):
    """
    Compute SHAP values from a scikit-learn Pipeline by safely extracting 
    the preprocessing steps and the underlying tree model.
    """
    # 1. Separate the preprocessing and the raw machine learning model
    if hasattr(model, 'named_steps') and 'model' in model.named_steps:
        preprocessor = model.named_steps.get('preprocessor', None)
        raw_model = model.named_steps['model']
    else:
        preprocessor = None
        raw_model = model

    # 2. Transform the data to pure numbers if a preprocessor exists
    if preprocessor is not None:
        # This converts categorical string columns into numeric One-Hot matrices
        X_numeric = preprocessor.transform(X_input)
    else:
        X_numeric = X_input

    # 3. If the result is a sparse matrix, convert it to a dense array for SHAP
    if hasattr(X_numeric, "toarray"):
        X_numeric = X_numeric.toarray()
    elif isinstance(X_numeric, pd.DataFrame):
        X_numeric = X_numeric.to_numpy()

    # Ensure everything is strictly numeric floats/ints now
    X_numeric = np.asarray(X_numeric, dtype=np.float64)

    # 4. Use TreeExplainer on the pure numeric matrix
    explainer = shap.TreeExplainer(raw_model)
    shap_values = explainer(X_numeric)

    # 5. Build the output DataFrame safely
    data_values = shap_values.values
    if len(data_values.shape) == 1:
        data_values = [data_values]

    shap_df = pd.DataFrame(
        data_values,
        columns=feature_names
    )

    return shap_values, shap_df
# ======================================================
# SHAP Summary Plot
# ======================================================

def plot_shap_summary(
    shap_values,
    X_processed,
    feature_names,
    save_path="outputs/shap_summary.png"
):
    """
    Generate SHAP summary plot.
    """

    plt.figure(figsize=(12, 8))

    shap.summary_plot(
        shap_values,
        features=X_processed,
        feature_names=feature_names,
        show=False
    )

    plt.tight_layout()

    #plt.savefig(save_path)

    print(f"SHAP summary plot saved to {save_path}")


# ======================================================
# SHAP Feature Importance Plot
# ======================================================

def plot_shap_importance(
    shap_values,
    X_processed,
    feature_names,
    save_path="outputs/shap_bar.png"
):
    """
    Generate SHAP feature importance bar plot.
    """

    plt.figure(figsize=(10, 6))

    shap.summary_plot(
        shap_values,
        features=X_processed,
        feature_names=feature_names,
        plot_type="bar",
        show=False
    )

    plt.tight_layout()

    #plt.savefig(save_path)

    print(f"SHAP importance plot saved to {save_path}")


# ======================================================
# Top Features
# ======================================================

def get_top_features(
    shap_df,
    top_n=10
):
    """
    Return top influential features.
    """

    mean_abs_shap = shap_df.abs().mean()

    top_features = (
        mean_abs_shap
        .sort_values(ascending=False)
        .head(top_n)
    )

    return top_features

def generate_business_interpretation(top_features):
    """
    Generate business-friendly explanations.
    """

    interpretations = []

    for feature, importance in top_features.items():

        explanation = f"""
Feature: {feature}

Business Impact:
This feature has a strong influence on claim severity predictions.
Higher values or specific categories in {feature} significantly
affect the expected insurance claim amount.

Model Importance Score:
{importance:.4f}
"""

        interpretations.append(explanation)

    return interpretations