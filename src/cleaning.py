"""
Módulo de Auditoría de Calidad y Limpieza de Datos para Contratación Pública (SECOP II).
Sigue estándares PEP8 y ofrece funciones modulares y reutilizables.
"""

from typing import Dict, List, Optional, Tuple
import unicodedata
import pandas as pd
import numpy as np


def clean_currency_series(series: pd.Series) -> pd.Series:
    """
    Limpia cadenas monetarias eliminando signos de moneda, comas, puntos confusos
    y espacios en blanco, convirtiendo los valores a float numérico.

    Args:
        series: Serie de pandas con valores monetarios (str, float, int o mixtos).

    Returns:
        pd.Series: Serie convertida a float64 (np.nan si no es convertible).
    """
    if series.empty:
        return series.astype(float)

    def _clean_val(val):
        if pd.isna(val):
            return np.nan
        if isinstance(val, (int, float)):
            return float(val)
        val_str = str(val).strip().replace("$", "").replace(" ", "").replace("\xa0", "")
        # Si tiene formato con comas de miles y punto decimal: 1,500,000.50 -> 1500000.50
        # O formato latino: 1.500.000,50 -> 1500000.50
        if "," in val_str and "." in val_str:
            if val_str.rfind(",") > val_str.rfind("."):
                # Coma es el separador decimal
                val_str = val_str.replace(".", "").replace(",", ".")
            else:
                # Punto es el separador decimal
                val_str = val_str.replace(",", "")
        elif "," in val_str:
            # Podría ser separador de miles o decimal
            parts = val_str.split(",")
            if len(parts[-1]) == 2:  # ej: 1500,50
                val_str = val_str.replace(",", ".")
            else:
                val_str = val_str.replace(",", "")
        return val_str

    cleaned_str = series.apply(_clean_val)
    return pd.to_numeric(cleaned_str, errors="coerce")


def parse_date_series(series: pd.Series) -> pd.Series:
    """
    Convierte series de fecha a formato datetime estandarizado (ISO 8601).

    Args:
        series: Serie de pandas con fechas en texto u objetos fecha.

    Returns:
        pd.Series: Serie convertida a datetime64[ns].
    """
    return pd.to_datetime(series, errors="coerce", format="ISO8601")


def normalize_text(text: Optional[str]) -> str:
    """
    Estandariza texto eliminando espacios sobrantes, normalizando a minúsculas
    y retirando tildes/acentos diacríticos.

    Args:
        text: Cadena de texto a normalizar.

    Returns:
        str: Texto normalizado o cadena vacía si es nulo.
    """
    if pd.isna(text):
        return ""
    text = str(text).strip().lower()
    text = unicodedata.normalize("NFKD", text).encode("ASCII", "ignore").decode("utf-8")
    return " ".join(text.split())


def clean_secop_dataset(
    df: pd.DataFrame,
    id_column: str = "id_del_portafolio",
    price_col: str = "precio_base",
    award_col: str = "valor_total_adjudicacion",
    date_cols: Optional[List[str]] = None,
) -> Tuple[pd.DataFrame, Dict[str, any]]:
    """
    Aplica el flujo integral de auditoría de calidad y saneamiento sobre el DataFrame SECOP II:
    1. Registro de dimensiones iniciales.
    2. Filtrado de registros con precio_base <= 0 y valores erróneos.
    3. Casteo riguroso de columnas monetarias.
    4. Eliminación de duplicados por identificador de proceso/contrato.
    5. Normalización y tipado de fechas.
    6. Generación del reporte de auditoría de calidad.

    Args:
        df: DataFrame original con datos crudos de SECOP II.
        id_column: Nombre de la columna identificadora única del proceso/contrato.
        price_col: Nombre de la columna de precio base / presupuesto oficial.
        award_col: Nombre de la columna de valor adjudicado.
        date_cols: Lista de columnas de fecha a convertir a datetime.

    Returns:
        Tuple[pd.DataFrame, Dict[str, any]]: DataFrame depurado y diccionario con métricas de calidad.
    """
    if date_cols is None:
        date_cols = ["fecha_de_publicacion_del"]

    df_clean = df.copy()
    initial_rows = len(df_clean)

    # 1. Casteo numérico de columnas financieras
    if price_col in df_clean.columns:
        df_clean[price_col] = clean_currency_series(df_clean[price_col])
    if award_col in df_clean.columns:
        df_clean[award_col] = clean_currency_series(df_clean[award_col])

    # 2. Filtrado de registros inválidos (precio_base <= 0 o NaN)
    invalid_price_mask = df_clean[price_col].isna() | (df_clean[price_col] <= 0)
    invalid_award_mask = df_clean[award_col].isna() | (df_clean[award_col] <= 0)
    invalid_financial_rows = (invalid_price_mask | invalid_award_mask).sum()

    df_clean = df_clean[~invalid_price_mask & ~invalid_award_mask].copy()

    # 3. Deduplicación por identificador
    duplicate_ids = 0
    if id_column in df_clean.columns:
        duplicate_ids = df_clean.duplicated(subset=[id_column], keep="first").sum()
        df_clean = df_clean.drop_duplicates(subset=[id_column], keep="first").copy()

    # 4. Parseo de fechas
    for col in date_cols:
        if col in df_clean.columns:
            df_clean[col] = parse_date_series(df_clean[col])

    # 5. Estandarización de texto en columnas de categorización
    text_cols = [
        "entidad",
        "departamento_entidad",
        "ciudad_entidad",
        "modalidad_de_contratacion",
        "tipo_de_contrato",
        "estado_del_procedimiento",
        "unidad_de_duracion",
    ]
    for col in text_cols:
        if col in df_clean.columns:
            df_clean[col] = df_clean[col].astype(str).str.strip()

    final_rows = len(df_clean)

    audit_report = {
        "filas_iniciales": initial_rows,
        "filas_finales": final_rows,
        "filas_descartadas_precio_invalido": int(invalid_financial_rows),
        "duplicados_eliminados": int(duplicate_ids),
        "porcentaje_retencion": round((final_rows / initial_rows) * 100, 2) if initial_rows > 0 else 0,
        "nulos_por_columna": df_clean.isnull().sum().to_dict(),
    }

    return df_clean, audit_report
