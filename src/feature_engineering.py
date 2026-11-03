import pandas as pd

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    
    current_year = 2026
    df["car_age"] = current_year - df["year"]
    df["car_age"] = df["car_age"].apply(lambda x: max(x, 0))
    
    df["mileage_per_year"] = df["mileage_km"] / (df["car_age"] + 1)
    
    if "volume_cm3" in df.columns:
        df["engine_volume_liters"] = df["volume_cm3"] / 1000.0
    else:
        df["engine_volume_liters"] = 2.0
        
    df["is_newer_car"] = (df["year"] >= 2015).astype(int)
    df["is_high_mileage"] = (df["mileage_km"] > 200000).astype(int)
    
    return df

if __name__ == "__main__":
    df = pd.read_csv("data/cars_cleaned.csv")
    df_features = build_features(df)
    df_features.to_csv("data/cars_features.csv", index=False)
    print(f"Feature engineering zavrsen. Sacuvano u data/cars_features.csv")
