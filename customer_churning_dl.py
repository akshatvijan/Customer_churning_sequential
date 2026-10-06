import os
import warnings
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

warnings.filterwarnings("ignore")


# ==========================================
# 1. LOAD DATASET
# ==========================================
def load_data(filepath="customer_churn_1M.csv"):
    """Load raw dataset from CSV file."""
    if not os.path.exists(filepath):
        # Fallback to parent directory if running inside subfolder
        parent_path = os.path.join("..", filepath)
        if os.path.exists(parent_path):
            filepath = parent_path

    print(f"Loading dataset from: {filepath}")
    df = pd.read_csv(filepath)
    print("Dataset Shape:", df.shape)
    return df


# ==========================================
# 2. DATA CLEANING
# ==========================================
def clean_data(df):
    """Clean signup_date and remove duplicate rows."""
    df = df.copy()

    # Convert signup_date into datetime
    df["signup_date"] = pd.to_datetime(df["signup_date"], errors="coerce")
    print(df["signup_date"].dtype)
    print("Invalid dates:", df["signup_date"].isnull().sum())

    # Remove duplicate rows
    before = len(df)
    df = df.drop_duplicates()
    after = len(df)
    print("Rows before:", before)
    print("Rows after :", after)
    print("Duplicates removed:", before - after)

    return df


# ==========================================
# 3. FEATURE ENGINEERING
# ==========================================
def engineer_features(df):
    """Extract temporal and behavioral features."""
    df = df.copy()

    # Extract useful information from signup date
    df["signup_year"] = df["signup_date"].dt.year
    df["signup_month"] = df["signup_date"].dt.month
    df["signup_quarter"] = df["signup_date"].dt.quarter
    df["signup_dayofweek"] = df["signup_date"].dt.dayofweek

    # Customer Behaviour Features
    safe_tenure = df["tenure"].replace(0, np.nan)

    # Average charges per tenure
    df["avg_charge_per_tenure"] = df["totalcharges"] / safe_tenure

    # Customer support burden
    df["support_burden"] = df["num_complaints"].fillna(0) + df["num_service_calls"].fillna(0)

    # Payment risk
    df["payment_risk"] = df["late_payments"].fillna(0) + df["num_complaints"].fillna(0)

    # Service usage ratio
    df["service_usage_ratio"] = df["num_services"] / safe_tenure

    # Data usage per service
    df["data_usage_per_service"] = df["avg_monthly_gb"] / df["num_services"].replace(0, np.nan)

    # Interaction recency
    df["interaction_recency_score"] = df["days_since_last_interaction"] / 30

    # Monthly income
    df["monthly_income"] = df["annual_income"] / 12

    # Income / monthly charge
    df["income_charge_ratio"] = df["monthly_income"] / df["monthlycharges"].replace(0, np.nan)

    # Handle Infinite Values
    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    print("Feature engineering completed.")
    return df


# ==========================================
# 4. PREPARE X AND Y
# ==========================================
def prepare_X_y(df):
    """Drop identifier and target columns to separate X and y."""
    drop_columns = ["customer_id", "signup_date", "churn"]
    X = df.drop(columns=drop_columns)
    y = df["churn"]
    print("X Shape:", X.shape)
    print("y Shape:", y.shape)
    return X, y


# ==========================================
# 5. TRAIN / VALIDATION / TEST SPLIT
# ==========================================
def split_data(X, y):
    """70% Train, 15% Validation, 15% Test stratified split."""
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, random_state=42, stratify=y
    )

    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp
    )

    print("Training:", X_train.shape)
    print("Validation:", X_val.shape)
    print("Testing:", X_test.shape)

    return X_train, X_val, X_test, y_train, y_val, y_test


# ==========================================
# 6. IDENTIFY FEATURE TYPES
# ==========================================
def get_feature_types(X_train):
    """Identify categorical and numerical columns."""
    categorical_features = X_train.select_dtypes(include=["object"]).columns.tolist()
    numerical_features = X_train.select_dtypes(include=["int64", "float64"]).columns.tolist()

    print("Categorical Features:", categorical_features)
    print("Numerical Features:", numerical_features)
    return categorical_features, numerical_features


# ==========================================
# 7. STATSMODELS FEATURE SELECTION
# ==========================================
def run_statsmodels_selection(X_train, y_train, categorical_features, sample_size=50000):
    """Run Statsmodels Binomial GLM on training sample for feature significance."""
    rng = np.random.RandomState(42)
    sample_indices = rng.choice(
        len(X_train), size=min(sample_size, len(X_train)), replace=False
    )

    X_stats = X_train.iloc[sample_indices].copy()
    y_stats = y_train.iloc[sample_indices].copy()

    # Temporary Encoding for Statsmodels
    X_stats_encoded = pd.get_dummies(
        X_stats, columns=categorical_features, drop_first=True, dtype=float
    )
    X_stats_encoded = X_stats_encoded.fillna(X_stats_encoded.median())
    X_stats_encoded = X_stats_encoded.fillna(0)

    # Statsmodels GLM with intercept
    X_stats_sm = sm.add_constant(X_stats_encoded, has_constant="add")
    glm_model = sm.GLM(y_stats, X_stats_sm, family=sm.families.Binomial())
    glm_result = glm_model.fit(maxiter=100)

    # Feature Significance DataFrame
    stats_results = pd.DataFrame({
        "feature": glm_result.params.index,
        "coefficient": glm_result.params.values,
        "p_value": glm_result.pvalues.values
    })
    stats_results = stats_results[stats_results["feature"] != "const"]
    stats_results = stats_results.sort_values("p_value")

    significant_features = stats_results[stats_results["p_value"] < 0.05]["feature"].tolist()
    print("Significant features:", len(significant_features))
    return stats_results, significant_features


# ==========================================
# 8. FINAL PREPROCESSING PIPELINE
# ==========================================
def build_and_transform_pipeline(X_train, X_val, X_test, numerical_features, categorical_features):
    """Build ColumnTransformer and transform train, validation, and test datasets."""
    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])

    preprocessor = ColumnTransformer([
        ("numerical", numeric_pipeline, numerical_features),
        ("categorical", categorical_pipeline, categorical_features)
    ])

    # Fit only on training data
    X_train_processed = preprocessor.fit_transform(X_train)
    X_val_processed = preprocessor.transform(X_val)
    X_test_processed = preprocessor.transform(X_test)

    feature_names = preprocessor.get_feature_names_out()

    X_train_processed_df = pd.DataFrame(X_train_processed, columns=feature_names, index=X_train.index)
    X_val_processed_df = pd.DataFrame(X_val_processed, columns=feature_names, index=X_val.index)
    X_test_processed_df = pd.DataFrame(X_test_processed, columns=feature_names, index=X_test.index)

    return preprocessor, feature_names, X_train_processed_df, X_val_processed_df, X_test_processed_df


# ==========================================
# 9. CORRELATION BASED FEATURE SELECTION
# ==========================================
def select_uncorrelated_features(X_train_processed_df, X_val_processed_df, X_test_processed_df, feature_names, threshold=0.95):
    """Filter out highly collinear features based on correlation matrix."""
    corr_matrix = X_train_processed_df.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

    high_corr_features = [
        column for column in upper.columns if any(upper[column] > threshold)
    ]

    selected_features = [
        feature for feature in feature_names if feature not in high_corr_features
    ]

    print("Features before selection:", len(feature_names))
    print("Features after selection:", len(selected_features))

    X_train_final = X_train_processed_df[selected_features]
    X_val_final = X_val_processed_df[selected_features]
    X_test_final = X_test_processed_df[selected_features]

    return selected_features, X_train_final, X_val_final, X_test_final


# ==========================================
# 10. CONVERT TO NUMPY & SEQUENTIAL FORMAT
# ==========================================
def format_numpy_arrays(X_train_final, X_val_final, X_test_final, y_train, y_val, y_test):
    """Convert to Tabular float32, Sequential 3D, and Target float32."""
    X_train_tabular = X_train_final.values.astype(np.float32)
    X_val_tabular = X_val_final.values.astype(np.float32)
    X_test_tabular = X_test_final.values.astype(np.float32)

    X_train_sequential = X_train_tabular.reshape(X_train_tabular.shape[0], 1, X_train_tabular.shape[1])
    X_val_sequential = X_val_tabular.reshape(X_val_tabular.shape[0], 1, X_val_tabular.shape[1])
    X_test_sequential = X_test_tabular.reshape(X_test_tabular.shape[0], 1, X_test_tabular.shape[1])

    y_train_final = y_train.values.astype(np.float32)
    y_val_final = y_val.values.astype(np.float32)
    y_test_final = y_test.values.astype(np.float32)

    return (
        X_train_tabular, X_val_tabular, X_test_tabular,
        X_train_sequential, X_val_sequential, X_test_sequential,
        y_train_final, y_val_final, y_test_final
    )


# ==========================================
# 11. SAVE PROCESSED ARTIFACTS
# ==========================================
def save_artifacts(output_dir,
                   X_train_tabular, X_val_tabular, X_test_tabular,
                   X_train_sequential, X_val_sequential, X_test_sequential,
                   y_train_final, y_val_final, y_test_final,
                   preprocessor, selected_features, stats_results,
                   numerical_features, categorical_features):
    """Save all processed numpy arrays and joblib objects."""
    os.makedirs(output_dir, exist_ok=True)

    # TABULAR DATA
    np.save(f"{output_dir}/X_train_tabular.npy", X_train_tabular)
    np.save(f"{output_dir}/X_val_tabular.npy", X_val_tabular)
    np.save(f"{output_dir}/X_test_tabular.npy", X_test_tabular)

    # SEQUENTIAL-READY DATA
    np.save(f"{output_dir}/X_train_sequential.npy", X_train_sequential)
    np.save(f"{output_dir}/X_val_sequential.npy", X_val_sequential)
    np.save(f"{output_dir}/X_test_sequential.npy", X_test_sequential)

    # TARGET
    np.save(f"{output_dir}/y_train.npy", y_train_final)
    np.save(f"{output_dir}/y_val.npy", y_val_final)
    np.save(f"{output_dir}/y_test.npy", y_test_final)

    # PREPROCESSING OBJECTS
    joblib.dump(preprocessor, f"{output_dir}/preprocessor.pkl")
    joblib.dump(selected_features, f"{output_dir}/selected_features.pkl")
    joblib.dump(stats_results, f"{output_dir}/statsmodels_results.pkl")
    joblib.dump(numerical_features, f"{output_dir}/numerical_features.pkl")
    joblib.dump(categorical_features, f"{output_dir}/categorical_features.pkl")

    print("Data files and preprocessing objects saved successfully.")


# ==========================================
# MAIN EXECUTION PIPELINE
# ==========================================
def main(filepath="customer_churn_1M.csv", output_dir="kapil_preprocessed_data"):
    # 1. Load data
    df = load_data(filepath)

    # 2. Clean data
    df = clean_data(df)

    # 3. Feature engineering
    df = engineer_features(df)

    # 4. Prepare X and y
    X, y = prepare_X_y(df)

    # 5. Split train/val/test
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, random_state=42, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp
    )

    # 6. Feature types
    categorical_features, numerical_features = get_feature_types(X_train)

    # 7. Statsmodels selection
    stats_results, significant_features = run_statsmodels_selection(X_train, y_train, categorical_features)

    # 8. ColumnTransformer Preprocessing
    preprocessor, feature_names, X_train_processed_df, X_val_processed_df, X_test_processed_df = build_and_transform_pipeline(
        X_train, X_val, X_test, numerical_features, categorical_features
    )

    # 9. Correlation based selection
    selected_features, X_train_final, X_val_final, X_test_final = select_uncorrelated_features(
        X_train_processed_df, X_val_processed_df, X_test_processed_df, feature_names
    )

    # 10. Format NumPy arrays
    (
        X_train_tabular, X_val_tabular, X_test_tabular,
        X_train_sequential, X_val_sequential, X_test_sequential,
        y_train_final, y_val_final, y_test_final
    ) = format_numpy_arrays(
        X_train_final, X_val_final, X_test_final, y_train, y_val, y_test
    )

    # 11. Save artifacts
    save_artifacts(
        output_dir,
        X_train_tabular, X_val_tabular, X_test_tabular,
        X_train_sequential, X_val_sequential, X_test_sequential,
        y_train_final, y_val_final, y_test_final,
        preprocessor, selected_features, stats_results,
        numerical_features, categorical_features
    )

    # 12. Final Preprocessing Output Banner
    print("=" * 60)
    print("KAPIL - FINAL PREPROCESSING OUTPUT")
    print("=" * 60)

    print("\nOriginal Dataset:")
    print(df.shape)

    print("\nTrain / Validation / Test:")
    print("Train:", X_train.shape)
    print("Val  :", X_val.shape)
    print("Test :", X_test.shape)

    print("\nFinal Tabular Data:")
    print("Train:", X_train_tabular.shape)
    print("Val  :", X_val_tabular.shape)
    print("Test :", X_test_tabular.shape)

    print("\nSequential-Ready Data:")
    print("Train:", X_train_sequential.shape)
    print("Val  :", X_val_sequential.shape)
    print("Test :", X_test_sequential.shape)

    print("\nFinal Selected Features:",
          len(selected_features))

    print("\nSaved Location:")
    print(output_dir)


if __name__ == "__main__":
    main()
