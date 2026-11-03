# Predikcija cene polovnih automobila

End-to-end regresioni projekat masinskog ucenja za predvidjanje trzisne cene polovnih automobila na osnovu njihovih tehnickih karakteristika.

## Struktura projekta

```text
car-price-prediction/
├── data/
│   ├── cars.csv
│   ├── cars_cleaned.csv
│   └── cars_features.csv
├── notebooks/
│   └── 01_eda.ipynb
├── src/
│   ├── data_cleaning.py
│   ├── feature_engineering.py
│   ├── data_preprocessing.py
│   ├── model_training.py
│   ├── model_evaluation.py
│   └── model_comparison.py
├── models/
│   └── car_price_model.joblib
├── requirements.txt
└── README.md
```

## Koraci i workflow

1. **EDA (`notebooks/01_eda.ipynb`)**: Analiza raspodele cena, kilometraze, godista i kategorijskih varijabli.
2. **Ciscenje podataka (`src/data_cleaning.py`)**: Standardizacija kolona, uklanjanje nedostajucih vrednosti i filtriranje nerealnih unosa.
3. **Inzenjering karakteristika (`src/feature_engineering.py`)**:
   - `car_age`: starost automobila
   - `mileage_per_year`: godisnja kilometraza
   - `engine_volume_liters`: zapremina motora u litrima
   - `is_newer_car`: indikator novijeg vozila
   - `is_high_mileage`: indikator visoke kilometraze
4. **Pretprocesiranje (`src/data_preprocessing.py`)**:
   - Numericke kolone: `SimpleImputer(strategy='median')` + `StandardScaler()`
   - Kategorijske kolone: `SimpleImputer(strategy='most_frequent')` + `OneHotEncoder(handle_unknown='ignore')`
5. **Treniranje i poredjenje (`src/model_comparison.py`)**:
   - Linear Regression
   - Decision Tree Regressor
   - Random Forest Regressor
   - Gradient Boosting Regressor
6. **Izbor modela**: Random Forest / Gradient Boosting daju najmanji MAE i najbolji R2 skor.

## Pokretanje projekta

```bash
# 1. Instalacija biblioteka
pip install -r requirements.txt

# 2. Ciscenje podataka
python -m src.data_cleaning

# 3. Feature engineering
python -m src.feature_engineering

# 4. Treniranje modela
python -m src.model_training

# 5. Evaluacija modela
python -m src.model_evaluation

# 6. Poredjenje modela
python -m src.model_comparison
```
