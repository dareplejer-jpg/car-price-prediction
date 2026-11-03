import pandas as pd
import re

def clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    new_cols = []
    for c in df.columns:
        clean = c.strip().lower()
        clean = clean.replace("(", "_").replace(")", "").replace("-", "_").replace(" ", "_")
        clean = re.sub(r"_+", "_", clean).strip("_")
        new_cols.append(clean)
    df.columns = new_cols
    
    rename_dict = {
        "priceusd": "price_usd",
        "mileage_kilometers": "mileage_km",
        "volume_cm3": "volume_cm3"
    }
    df = df.rename(columns=rename_dict)
    return df

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = clean_column_names(df)
    
    # Uklanjanje nedostajucih vrednosti za kljucne kolone
    df = df.dropna(subset=["price_usd", "year", "mileage_km"]).copy()
    
    # Ciscenje nerealnih vrednosti
    df = df[(df["price_usd"] >= 200) & (df["price_usd"] <= 200000)]
    df = df[(df["year"] >= 1980) & (df["year"] <= 2026)]
    df = df[(df["mileage_km"] >= 0) & (df["mileage_km"] <= 1000000)]
    
    # Standardizacija tekstualnih vrednosti
    text_cols = df.select_dtypes(include=["object"]).columns
    for col in text_cols:
        df[col] = df[col].astype(str).str.strip().str.lower()
        df[col] = df[col].replace(["nan", "none", "null", ""], pd.NA)
        
    return df.reset_index(drop=True)

if __name__ == "__main__":
    raw_df = pd.read_csv("data/cars.csv")
    cleaned_df = clean_data(raw_df)
    cleaned_df.to_csv("data/cars_cleaned.csv", index=False)
    print(f"Ciscenje zavrseno. Sacuvano {len(cleaned_df)} redova u data/cars_cleaned.csv")
