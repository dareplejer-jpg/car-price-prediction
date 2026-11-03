import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.data_preprocessing import split_features_and_target

def evaluate_model():
    df = pd.read_csv("data/cars_features.csv")
    X, y = split_features_and_target(df)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = joblib.load("models/car_price_model.joblib")
    y_pred = model.predict(X_test)
    
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = mse ** 0.5
    r2 = r2_score(y_test, y_pred)
    
    print("=== METRIKE REGRESIJE ===")
    print(f"MAE:  ${mae:.2f}")
    print(f"MSE:  {mse:.2f}")
    print(f"RMSE: ${rmse:.2f}")
    print(f"R2:   {r2:.4f}")
    
    print("\n=== PRIMERI PREDIKCIJA ===")
    examples = pd.DataFrame({
        "Stvarna cena": y_test.values[:5],
        "Predvidjena cena": y_pred[:5],
        "Greska": y_test.values[:5] - y_pred[:5],
        "Apsolutna greska": abs(y_test.values[:5] - y_pred[:5])
    })
    print(examples.to_string(index=False))

if __name__ == "__main__":
    evaluate_model()
