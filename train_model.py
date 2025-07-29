import pandas as pd
import numpy as np
import lightgbm as lgb
import optuna
import joblib
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from pandas.errors import EmptyDataError

def create_mock_historical_data(filename="historical_data.csv"):
    """Crea un archivo CSV de datos históricos de ejemplo para demostración."""
    print("Creando archivo de datos históricos de ejemplo...")
    data = {
        'race_id': [f'2023-R{i}' for i in range(1, 21) for _ in range(8)],
        'race_date': pd.to_datetime([f'2023-0{((i-1)//5)+1}-{((i-1)%5)*5+1}' for i in range(1, 21) for _ in range(8)]),
        'runner_id': [f'H{j}' for i in range(1, 21) for j in range(1, 9)],
        'horse_name': [f'Horse_{j}' for i in range(1, 21) for j in range(1, 9)],
        'course': np.random.choice(['Ayr', 'York', 'Newmarket'], 20 * 8),
        'official_rating': np.random.randint(70, 100, 20 * 8),
        'age': np.random.randint(2, 6, 20 * 8),
        'weight_lbs': np.random.randint(120, 140, 20 * 8),
        'jockey_name': np.random.choice(['JockeyA', 'JockeyB', 'JockeyC'], 20 * 8),
        'trainer_name': np.random.choice(['TrainerX', 'TrainerY', 'TrainerZ'], 20 * 8),
        'swot_balance_score': np.random.randint(-5, 5, 20 * 8),
        'in_running_comment': np.random.choice(['led early', 'ran on well', 'faded late', 'slow start'], 20 * 8),
        'finish_position': [np.random.permutation([1, 2, 3, 4, 5, 6, 7, 8])[j] for i in range(20) for j in range(8)]
    }
    df = pd.DataFrame(data)
    df.to_csv(filename, index=False)
    print(f"Archivo '{filename}' creado con éxito.")
    return df

def feature_engineering_pipeline():
    """Define el pipeline de preprocesamiento de características."""
    
    numeric_features = ['official_rating', 'age', 'weight_lbs', 'swot_balance_score']
    categorical_features = ['course', 'jockey_name', 'trainer_name']
    text_feature = 'in_running_comment'

    numeric_transformer = Pipeline(steps=[('scaler', StandardScaler())])
    categorical_transformer = Pipeline(steps=[('onehot', OneHotEncoder(handle_unknown='ignore'))])
    text_transformer = Pipeline(steps=[('tfidf', TfidfVectorizer(max_features=50))])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features),
            ('text', text_transformer, text_feature)
        ],
        remainder='passthrough'
    )
    
    return preprocessor

def objective(trial, X, y, preprocessor):
    """Función objetivo para la optimización con Optuna."""
    
    # CORRECCIÓN: Cambiado a 'binary' ya que predecimos si gana (1) o no (0).
    model_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', lgb.LGBMClassifier(objective='binary', random_state=42, verbosity=-1))
    ])

    param_grid = {
        "classifier__n_estimators": trial.suggest_int("n_estimators", 200, 2000, step=100),
        "classifier__learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "classifier__num_leaves": trial.suggest_int("num_leaves", 20, 300),
        "classifier__max_depth": trial.suggest_int("max_depth", 3, 12),
        "classifier__lambda_l1": trial.suggest_float("lambda_l1", 1e-8, 10.0, log=True),
        "classifier__lambda_l2": trial.suggest_float("lambda_l2", 1e-8, 10.0, log=True),
    }
    
    model_pipeline.set_params(**param_grid)

    tscv = TimeSeriesSplit(n_splits=5)
    scores = []
    for train_index, val_index in tscv.split(X):
        X_train, X_val = X.iloc[train_index], X.iloc[val_index]
        y_train, y_val = y.iloc[train_index], y.iloc[val_index]
        
        # Añadido control para evitar error en splits sin ganadores
        if len(y_train.unique()) < 2:
            continue

        model_pipeline.fit(X_train, y_train)
        from sklearn.metrics import log_loss
        y_pred_proba = model_pipeline.predict_proba(X_val)
        score = log_loss(y_val, y_pred_proba)
        scores.append(score)

    return np.mean(scores) if scores else float('inf')

def main():
    """Función principal para entrenar y guardar el modelo."""
    try:
        df = pd.read_csv("historical_data.csv", parse_dates=['race_date'])
    except (FileNotFoundError, EmptyDataError):
        print("El archivo 'historical_data.csv' no se encontró o está vacío. Creando datos de ejemplo...")
        df = create_mock_historical_data()

    df = df.sort_values(by='race_date')
    
    df['is_winner'] = (df['finish_position'] == 1).astype(int)
    
    features = [
        'official_rating', 'age', 'weight_lbs', 'swot_balance_score', 
        'course', 'jockey_name', 'trainer_name', 'in_running_comment'
    ]
    X = df[features]
    y = df['is_winner']

    preprocessor = feature_engineering_pipeline()

    print("Iniciando optimización de hiperparámetros con Optuna...")
    study = optuna.create_study(direction="minimize")
    study.optimize(lambda trial: objective(trial, X, y, preprocessor), n_trials=50)

    print(f"Mejor puntuación (log_loss): {study.best_value}")
    print(f"Mejores hiperparámetros: {study.best_params}")

    print("Entrenando modelo final con los mejores parámetros...")
    # CORRECCIÓN: Cambiado a 'binary' para el modelo final también.
    final_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', lgb.LGBMClassifier(objective='binary', random_state=42, verbosity=-1, **study.best_params))
    ])
    
    final_pipeline.fit(X, y)

    joblib.dump(final_pipeline, 'lgbm_model.joblib')
    print("¡Éxito! Modelo guardado como 'lgbm_model.joblib'")

if __name__ == "__main__":
    main()
