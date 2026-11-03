import os
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from src.data_cleaning import clean_data
from src.data_preprocessing import build_preprocessor, split_features_and_target
from src.feature_engineering import build_features


def train_and_save_model():
  if not os.path.exists("data/cars_features.csv"):
    raw_df = pd.read_csv("data/cars.csv")
    cleaned_df = clean_data(raw_df)
    df = build_features(cleaned_df)
    df.to_csv("data/cars_features.csv", index=False)
  else:
    df = pd.read_csv("data/cars_features.csv")

  X, y = split_features_and_target(df)
  X_train, X_test, y_train, y_test = train_test_split(
      X, y, test_size=0.2, random_state=42
  )

  model = Pipeline(
      steps=[
          ("preprocessor", build_preprocessor()),
          (
              "regressor",
              RandomForestRegressor(
                  n_estimators=50, max_depth=12, random_state=42
              ),
          ),
      ]
  )

  print("Treniranje optimizovanog Random Forest modela...")
  model.fit(X_train, y_train)

  os.makedirs("models", exist_ok=True)
  model_path = "models/car_price_model.joblib"
  joblib.dump(model, model_path, compress=3)
  print(f"Model uspesno sacuvan u: {model_path}")


if __name__ == "__main__":
  train_and_save_model()