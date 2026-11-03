import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer

TARGET_COLUMN = "price_usd"

NUMERIC_FEATURES = [
    "car_age",
    "mileage_km",
    "mileage_per_year",
    "engine_volume_liters",
    "is_newer_car",
    "is_high_mileage"
]

CATEGORICAL_FEATURES = [
    "make",
    "fuel_type",
    "transmission",
    "drive_unit",
    "segment",
    "condition"
]

def get_feature_columns():
    return NUMERIC_FEATURES + CATEGORICAL_FEATURES

def split_features_and_target(df: pd.DataFrame):
    cols = [c for c in get_feature_columns() if c in df.columns]
    X = df[cols].copy()
    y = df[TARGET_COLUMN].copy()
    return X, y

def build_preprocessor():
    num_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    
    cat_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", num_transformer, NUMERIC_FEATURES),
            ("cat", cat_transformer, CATEGORICAL_FEATURES)
        ],
        remainder="drop"
    )
    return preprocessor
