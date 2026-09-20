"""
Módulo de Auditoría Estadística y Detección de Anomalías (SECOP II).
Proporciona funciones puras para medidas de tendencia central, asimetría,
dispersión, detección de valores atípicos (Tukey IQR) y contraste descriptivo.
"""

from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd


def calculate_central_tendency_and_skewness(
    df: pd.DataFrame, columns: List[str]
) -> pd.DataFrame:
    """
    Calcula media, mediana, moda y coeficiente de asimetría de Fisher-Pearson (skewness)
    para una lista de variables numéricas, cuantificando la divergencia Media vs. Mediana.

    Args:
        df: DataFrame que contiene las variables a analizar.
        columns: Lista de nombres de columnas numéricas.

    Returns:
        pd.DataFrame: Tabla estructurada con las métricas por cada variable.
    """
    records = []
    for col in columns:
        if col not in df.columns:
            continue
        series = pd.to_numeric(df[col], errors="coerce").dropna()
        if series.empty:
            continue

        media = float(series.mean())
        mediana = float(series.median())
        moda_series = series.mode()
        moda = float(moda_series.iloc[0]) if not moda_series.empty else np.nan
        asimetria = float(series.skew())

        # Divergencia porcentual: ((media - mediana) / mediana) * 100
        # Si mediana == 0, se calcula la diferencia absoluta
        if mediana != 0:
            divergencia_pct = ((media - mediana) / abs(mediana)) * 100.0
        else:
            divergencia_pct = np.nan

        # Clasificación del sesgo
        if asimetria > 0.5:
            interpretacion_sesgo = "Asimetría Positiva (Sesgo a la derecha)"
        elif asimetria < -0.5:
            interpretacion_sesgo = "Asimetría Negativa (Sesgo a la izquierda)"
        else:
            interpretacion_sesgo = "Aproximadamente Simétrica"

        records.append({
            "variable": col,
            "media": media,
            "mediana": mediana,
            "moda": moda,
            "skewness": asimetria,
            "divergencia_media_vs_mediana_pct": divergencia_pct,
            "interpretacion_sesgo": interpretacion_sesgo,
        })

    result_df = pd.DataFrame(records)
    if not result_df.empty:
        result_df.set_index("variable", inplace=True)
    return result_df


def calculate_dispersion_measures(
    df: pd.DataFrame, columns: List[str]
) -> pd.DataFrame:
    """
    Calcula medidas de dispersión paramétricas y no paramétricas:
    desviación estándar, varianza, valor mínimo, máximo, rango y rango intercuartílico (IQR).

    Args:
        df: DataFrame de entrada.
        columns: Lista de nombres de columnas numéricas.

    Returns:
        pd.DataFrame: Tabla con las métricas de dispersión por variable.
    """
    records = []
    for col in columns:
        if col not in df.columns:
            continue
        series = pd.to_numeric(df[col], errors="coerce").dropna()
        if series.empty:
            continue

        std = float(series.std())
        var = float(series.var())
        val_min = float(series.min())
        val_max = float(series.max())
        rango = val_max - val_min
        q1 = float(series.quantile(0.25))
        q3 = float(series.quantile(0.75))
        iqr = q3 - q1
        mean = float(series.mean())
        cv_pct = (std / abs(mean) * 100.0) if mean != 0 else np.nan

        records.append({
            "variable": col,
            "desviacion_estandar": std,
            "varianza": var,
            "min": val_min,
            "max": val_max,
            "rango": rango,
            "q1": q1,
            "q3": q3,
            "iqr": iqr,
            "coeficiente_variacion_pct": cv_pct,
        })

    result_df = pd.DataFrame(records)
    if not result_df.empty:
        result_df.set_index("variable", inplace=True)
    return result_df


def segment_dispersion_by_modality(
    df: pd.DataFrame,
    value_col: str = "valor_total_adjudicacion",
    modality_col: str = "modalidad_de_contratacion",
) -> pd.DataFrame:
    """
    Segmenta las medidas de tendencia central y dispersión (std, IQR, mediana)
    según la modalidad de contratación administrativa.

    Args:
        df: DataFrame con datos de contratación.
        value_col: Columna numérica a analizar.
        modality_col: Columna categórica de agrupación.

    Returns:
        pd.DataFrame: Tabla comparativa agrupada por modalidad.
    """
    grouped = df.groupby(modality_col)[value_col]

    summary = grouped.agg(
        conteo="count",
        media="mean",
        mediana="median",
        std="std",
        q1=lambda s: s.quantile(0.25),
        q3=lambda s: s.quantile(0.75),
    )
    summary["iqr"] = summary["q3"] - summary["q1"]
    summary["porcentaje_contratos"] = (summary["conteo"] / len(df)) * 100.0

    return summary.sort_values(by="conteo", ascending=False)


def detect_tukey_outliers(
    series: pd.Series, k: float = 1.5
) -> Dict[str, Union[float, int, pd.Series]]:
    """
    Aplica el método clásico de Tukey (Boxplot rule) para identificar valores atípicos:
    - Límite inferior: Q1 - k * IQR
    - Límite superior: Q3 + k * IQR

    Args:
        series: Serie numérica de pandas.
        k: Factor multiplicador del IQR (típicamente 1.5 para outliers moderados).

    Returns:
        Dict con los umbrales calculados, conteos y máscara booleana de atípicos.
    """
    s_clean = pd.to_numeric(series, errors="coerce")
    q1 = float(s_clean.quantile(0.25))
    q3 = float(s_clean.quantile(0.75))
    iqr = q3 - q1

    lower_bound = q1 - (k * iqr)
    upper_bound = q3 + (k * iqr)

    is_upper_outlier = s_clean > upper_bound
    is_lower_outlier = s_clean < lower_bound
    is_outlier = is_upper_outlier | is_lower_outlier

    total_valid = int(s_clean.notnull().sum())
    count_outliers = int(is_outlier.sum())
    pct_outliers = (count_outliers / total_valid * 100.0) if total_valid > 0 else 0.0

    return {
        "q1": q1,
        "q3": q3,
        "iqr": iqr,
        "limite_inferior": lower_bound,
        "limite_superior": upper_bound,
        "conteo_outliers": count_outliers,
        "conteo_outliers_superiores": int(is_upper_outlier.sum()),
        "conteo_outliers_inferiores": int(is_lower_outlier.sum()),
        "porcentaje_outliers": pct_outliers,
        "mask_outlier": is_outlier,
        "mask_upper_outlier": is_upper_outlier,
    }


def extract_top_outliers(
    df: pd.DataFrame,
    col: str,
    top_n: int = 10,
    k: float = 1.5,
    info_cols: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Extrae el Top N de contratos con valores atípicos superiores según el criterio de Tukey,
    ordenados de mayor a menor magnitud.

    Args:
        df: DataFrame completo.
        col: Columna numérica sobre la cual evaluar los outliers.
        top_n: Número de registros atípicos a retornar.
        k: Factor multiplicativo del IQR (1.5).
        info_cols: Columnas descriptivas a incluir en la tabla de salida.

    Returns:
        pd.DataFrame: Top N contratos atípicos con contexto administrativo.
    """
    if info_cols is None:
        info_cols = [
            "id_del_portafolio",
            "referencia_del_proceso",
            "entidad",
            "ciudad_entidad",
            "modalidad_de_contratacion",
            "precio_base",
            "valor_total_adjudicacion",
            "duracion_dias_prevista",
        ]

    tukey_info = detect_tukey_outliers(df[col], k=k)
    upper_mask = tukey_info["mask_upper_outlier"]

    outliers_df = df[upper_mask].copy()

    # Seleccionar columnas existentes
    selected_cols = [c for c in info_cols if c in outliers_df.columns]
    if col not in selected_cols:
        selected_cols.append(col)

    sorted_outliers = outliers_df[selected_cols].sort_values(by=col, ascending=False)
    return sorted_outliers.head(top_n)


def hypothesis_contrast_competition(
    df: pd.DataFrame,
    competition_col: str = "proveedores_unicos_con",
    value_col: str = "valor_total_adjudicacion",
    overcost_flag_col: str = "tiene_sobrecosto",
) -> pd.DataFrame:
    """
    Realiza un contraste descriptivo agrupando por nivel de competencia (número de oferentes):
    Evalúa si un mayor número de oferentes se asocia con menores montos de adjudicación
    y menor probabilidad de sobrecosto.

    Segmentos de competencia:
    - '1 oferente (Sin competencia directiva)'
    - '2 a 3 oferentes (Baja competencia)'
    - '4 a 9 oferentes (Competencia media)'
    - '10 o más oferentes (Alta concurrencia)'

    Args:
        df: DataFrame saneado.
        competition_col: Columna con el número de oferentes únicos.
        value_col: Columna de valor adjudicado.
        overcost_flag_col: Columna binaria indicando sobrecosto.

    Returns:
        pd.DataFrame: Resumen estructurado por segmento de concurrencia.
    """
    df_temp = df.copy()

    def _categorize_competition(val):
        if pd.isna(val) or val <= 1:
            return "1 oferente (Unipersonal)"
        elif 2 <= val <= 3:
            return "2 - 3 oferentes (Baja concurrencia)"
        elif 4 <= val <= 9:
            return "4 - 9 oferentes (Media concurrencia)"
        else:
            return "10+ oferentes (Alta concurrencia)"

    df_temp["rango_competencia"] = df_temp[competition_col].apply(_categorize_competition)

    grouped = df_temp.groupby("rango_competencia").agg(
        total_contratos=("id_del_portafolio", "count"),
        mediana_valor_adjudicado=(value_col, "median"),
        media_valor_adjudicado=(value_col, "mean"),
        tasa_sobrecosto_pct=(overcost_flag_col, lambda s: s.mean() * 100.0),
        mediana_duracion_dias=("duracion_dias_prevista", "median"),
    )

    # Orden lógico de concurrencia
    orden = [
        "1 oferente (Unipersonal)",
        "2 - 3 oferentes (Baja concurrencia)",
        "4 - 9 oferentes (Media concurrencia)",
        "10+ oferentes (Alta concurrencia)",
    ]
    grouped = grouped.reindex([cat for cat in orden if cat in grouped.index])
    grouped["porcentaje_del_total"] = (grouped["total_contratos"] / len(df_temp)) * 100.0

    return grouped
