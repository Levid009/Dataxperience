"""
Módulo de Feature Engineering Inicial para Auditoría de Contratación Pública (SECOP II).
Calcula indicadores de desvío presupuestal y estandarización temporal.
"""

from typing import Optional
import unicodedata
import pandas as pd
import numpy as np


def parse_duration_to_days(duracion: any, unidad: any) -> Optional[float]:
    """
    Convierte un valor de duración y su unidad a días previstos.
    Estandariza la unidad limpiando espacios, convirtiendo a minúsculas y normalizando acentos.

    Factores de conversión estándar:
    - 'dia' / 'dias': 1 día
    - 'mes' / 'meses': 30 días
    - 'ano' / 'ano(s)' / 'año' / 'años': 365 días

    Args:
        duracion: Valor numérico de duración prevista.
        unidad: Cadena de texto indicando la unidad de tiempo.

    Returns:
        Optional[float]: Duración estandarizada en días o np.nan si no es computable.
    """
    try:
        dur_num = float(duracion)
    except (ValueError, TypeError):
        return np.nan

    if pd.isna(unidad):
        return dur_num

    # Estandarización de texto (minúsculas, sin espacios, sin tildes)
    u_str = str(unidad).strip().lower()
    u_str = unicodedata.normalize("NFKD", u_str).encode("ASCII", "ignore").decode("utf-8")

    factor = 1.0
    if "mes" in u_str:
        factor = 30.0
    elif "ano" in u_str:
        factor = 365.0
    elif "dia" in u_str:
        factor = 1.0
    else:
        # Si no se reconoce la unidad por defecto se preserva el número
        factor = 1.0

    return dur_num * factor


def standardize_duration_series(
    duracion_series: pd.Series, unidad_series: pd.Series
) -> pd.Series:
    """
    Vectoriza la estandarización de la duración prevista del contrato a días.

    Args:
        duracion_series: Serie con las magnitudes de duración.
        unidad_series: Serie con las unidades textuales (Mes(es), día(s), etc.).

    Returns:
        pd.Series: Serie con la duración prevista en días (float64).
    """
    return [
        parse_duration_to_days(d, u)
        for d, u in zip(duracion_series, unidad_series)
    ]


def add_initial_features(
    df: pd.DataFrame,
    price_col: str = "precio_base",
    award_col: str = "valor_total_adjudicacion",
    duration_col: str = "duracion",
    unit_col: str = "unidad_de_duracion",
) -> pd.DataFrame:
    """
    Calcula las variables ingenieriles requeridas para el análisis y modelado:
    1. 'sobrecosto_pesos' = valor_total_adjudicacion - precio_base
    2. 'ratio_sobrecosto' = sobrecosto_pesos / precio_base
    3. 'tiene_sobrecosto' = 1 si sobrecosto_pesos > 0 else 0
    4. 'duracion_dias_prevista' = duración estandarizada a días

    Args:
        df: DataFrame preprocesado y saneado.
        price_col: Nombre de la columna de precio base / presupuesto inicial.
        award_col: Nombre de la columna de valor total adjudicado.
        duration_col: Nombre de la columna de magnitud de duración.
        unit_col: Nombre de la columna de unidad de duración.

    Returns:
        pd.DataFrame: DataFrame enriquecido con las nuevas columnas calculadas.
    """
    df_feat = df.copy()

    # 1. Delta monetario de sobrecosto (pesos colombianos COP)
    df_feat["sobrecosto_pesos"] = df_feat[award_col] - df_feat[price_col]

    # 2. Ratio de sobrecosto respecto al presupuesto inicial
    # Seguro frente a divisiones por cero (garantizado previamente por limpieza)
    df_feat["ratio_sobrecosto"] = np.where(
        df_feat[price_col] > 0,
        df_feat["sobrecosto_pesos"] / df_feat[price_col],
        0.0
    )

    # 3. Variable objetivo binaria de clasificación
    df_feat["tiene_sobrecosto"] = (df_feat["sobrecosto_pesos"] > 0).astype(int)

    # 4. Duración prevista estandarizada a días
    if duration_col in df_feat.columns and unit_col in df_feat.columns:
        df_feat["duracion_dias_prevista"] = standardize_duration_series(
            df_feat[duration_col], df_feat[unit_col]
        )
        df_feat["duracion_dias_prevista"] = pd.to_numeric(
            df_feat["duracion_dias_prevista"], errors="coerce"
        )
    elif duration_col in df_feat.columns:
        df_feat["duracion_dias_prevista"] = pd.to_numeric(
            df_feat[duration_col], errors="coerce"
        )

    return df_feat
