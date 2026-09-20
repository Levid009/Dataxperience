"""
Módulo de Modelado Predictivo y Evaluación Econométrica (SECOP II).
Implementa el pipeline de Machine Learning supervisado para predecir
el valor total adjudicado de contratos de obra pública en Cundinamarca.
"""

import os
from typing import Any, Dict, List, Optional, Tuple
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

NUMERIC_FEATURES: List[str] = [
    "precio_base_log",
    "duracion_dias_prevista",
    "proveedores_unicos_con",
]

CATEGORICAL_FEATURES: List[str] = [
    "modalidad_de_contratacion",
]

TARGET_COL: str = "valor_total_adjudicacion"


def prepare_dataset(
    df: pd.DataFrame,
    target_col: str = TARGET_COL,
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Prepara matriz de características (X) y vector objetivo (y) aplicando np.log1p.
    """
    required_cols = [
        "precio_base",
        "duracion_dias_prevista",
        "proveedores_unicos_con",
        "modalidad_de_contratacion",
        target_col,
    ]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Faltan columnas requeridas en el DataFrame: {missing}")

    X = pd.DataFrame(
        {
            "precio_base_log": np.log1p(df["precio_base"].clip(lower=0)),
            "duracion_dias_prevista": df["duracion_dias_prevista"].fillna(0.0),
            "proveedores_unicos_con": df["proveedores_unicos_con"].fillna(1.0),
            "modalidad_de_contratacion": df["modalidad_de_contratacion"]
            .fillna("No Definido")
            .astype(str),
        }
    )

    y = np.log1p(df[target_col].clip(lower=0))
    return X, y


def build_preprocessor(
    numeric_features: Optional[List[str]] = None,
    categorical_features: Optional[List[str]] = None,
) -> ColumnTransformer:
    """
    Construye el ColumnTransformer para estandarizar numéricas y codificar categóricas.
    """
    if numeric_features is None:
        numeric_features = NUMERIC_FEATURES
    if categorical_features is None:
        categorical_features = CATEGORICAL_FEATURES

    return ColumnTransformer(
        transformers=[
            (
                "num",
                StandardScaler(),
                numeric_features,
            ),
            (
                "cat",
                OneHotEncoder(drop="first", handle_unknown="ignore"),
                categorical_features,
            ),
        ],
        remainder="drop",
        verbose_feature_names_out=True,
    )


def build_model_pipeline(
    model_type: str = "ridge",
    alpha: float = 1.0,
    random_state: int = 42,
) -> Pipeline:
    """
    Construye el Pipeline con ColumnTransformer y estimador lineal regularizado.
    """
    preprocessor = build_preprocessor()

    if model_type.lower() == "ridge":
        regressor = Ridge(alpha=alpha, random_state=random_state)
    elif model_type.lower() == "linear":
        regressor = LinearRegression()
    else:
        raise ValueError(
            f"Tipo de modelo no soportado: '{model_type}'. Use 'ridge' o 'linear'."
        )

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("regressor", regressor),
        ]
    )


def calculate_mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calcula el Error Porcentual Absoluto Medio (MAPE) en porcentaje (0 - 100%).
    """
    mask = y_true > 0
    if not np.any(mask):
        return 0.0
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100.0)


def train_and_evaluate(
    pipeline: Pipeline,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> Dict[str, Any]:
    """
    Ajusta el pipeline y evalúa en escala logarítmica y real en COP con np.expm1.
    """
    pipeline.fit(X_train, y_train)

    y_pred_log = pipeline.predict(X_test)
    y_test_log = y_test.to_numpy() if isinstance(y_test, pd.Series) else y_test

    y_test_real = np.expm1(y_test_log)
    y_pred_real = np.expm1(y_pred_log)

    r2_log = float(r2_score(y_test_log, y_pred_log))
    mae_log = float(mean_absolute_error(y_test_log, y_pred_log))
    rmse_log = float(np.sqrt(mean_squared_error(y_test_log, y_pred_log)))

    r2_real = float(r2_score(y_test_real, y_pred_real))
    mae_real = float(mean_absolute_error(y_test_real, y_pred_real))
    mae_millones_cop = mae_real / 1e6
    mape_real = calculate_mape(y_test_real, y_pred_real)

    metrics = {
        "r2_log": r2_log,
        "mae_log": mae_log,
        "rmse_log": rmse_log,
        "r2_real": r2_real,
        "mae_real": mae_real,
        "mae_millones_cop": mae_millones_cop,
        "mape_real": mape_real,
        "n_train": len(X_train),
        "n_test": len(X_test),
    }

    return {
        "pipeline": pipeline,
        "metrics": metrics,
        "y_test_log": y_test_log,
        "y_pred_log": y_pred_log,
        "y_test_real": y_test_real,
        "y_pred_real": y_pred_real,
        "residuals_log": y_test_log - y_pred_log,
        "residuals_real": y_test_real - y_pred_real,
    }


def clean_feature_label(name: str) -> str:
    """Formatea nombres técnicos de features a etiquetas legibles."""
    name = name.replace("num__", "").replace("cat__", "")
    translations = {
        "precio_base_log": "Precio Base [log1p]",
        "duracion_dias_prevista": "Duración Prevista (Días)",
        "proveedores_unicos_con": "Concurrencia Oferentes (N°)",
        "modalidad_de_contratacion_": "Modalidad: ",
    }
    for k, v in translations.items():
        name = name.replace(k, v)
    return name


def get_feature_coefficients(pipeline: Pipeline) -> pd.DataFrame:
    """Extrae los coeficientes estandarizados del modelo alineados con las features."""
    preprocessor: ColumnTransformer = pipeline.named_steps["preprocessor"]
    regressor = pipeline.named_steps["regressor"]

    raw_names = preprocessor.get_feature_names_out()
    coefs = regressor.coef_

    records = []
    for raw_name, coef in zip(raw_names, coefs):
        display_name = clean_feature_label(raw_name)
        direction = "Positivo (Incrementa Monto)" if coef > 0 else "Negativo (Disminuye Monto)"
        records.append(
            {
                "variable": raw_name,
                "variable_limpia": display_name,
                "coeficiente": float(coef),
                "direccion": direction,
                "magnitud": float(abs(coef)),
            }
        )

    df_coef = pd.DataFrame(records).sort_values(by="magnitud", ascending=False)
    df_coef.reset_index(drop=True, inplace=True)
    return df_coef


def save_model(pipeline: Pipeline, filepath: str) -> str:
    """Serializa el pipeline entrenado con joblib."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    joblib.dump(pipeline, filepath)
    return os.path.abspath(filepath)


def load_model(filepath: str) -> Pipeline:
    """Carga un pipeline serializado con joblib."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"No se encontró el archivo del modelo en: '{filepath}'")
    return joblib.load(filepath)
