"""
Pruebas unitarias automatizadas para la Etapa 1: Ingesta, Limpieza y Feature Engineering.
Ejecutar con: pytest tests/test_etapa_1.py -v
"""

import pandas as pd
import numpy as np
import pytest

from src.cleaning import (
    clean_currency_series,
    parse_date_series,
    normalize_text,
    clean_secop_dataset,
)
from src.features import (
    parse_duration_to_days,
    standardize_duration_series,
    add_initial_features,
)


def test_clean_currency_series():
    series = pd.Series(["$ 1,500,000", "2.500.000,50", " 350000 ", np.nan, 120000])
    cleaned = clean_currency_series(series)

    assert cleaned.iloc[0] == 1500000.0
    assert cleaned.iloc[1] == 2500000.50
    assert cleaned.iloc[2] == 350000.0
    assert pd.isna(cleaned.iloc[3])
    assert cleaned.iloc[4] == 120000.0


def test_parse_date_series():
    series = pd.Series(["2023-05-12T00:00:00.000", "2024-01-01", "invalido", None])
    dates = parse_date_series(series)

    assert dates.iloc[0].year == 2023
    assert dates.iloc[1].year == 2024
    assert pd.isna(dates.iloc[2])
    assert pd.isna(dates.iloc[3])


def test_parse_duration_to_days():
    # Meses
    assert parse_duration_to_days(3, "Mes(es)") == 90.0
    assert parse_duration_to_days("2", "meses") == 60.0
    # Días
    assert parse_duration_to_days(45, "día(s)") == 45.0
    assert parse_duration_to_days("15", "dias") == 15.0
    # Años
    assert parse_duration_to_days(1, "Año(s)") == 365.0
    assert parse_duration_to_days("0.5", "anos") == 182.5
    # Inválidos
    assert pd.isna(parse_duration_to_days("invalido", "dias"))


def test_clean_secop_dataset_filtering():
    data = {
        "id_del_portafolio": ["A1", "A2", "A2", "A3", "A4"],
        "precio_base": [1000000, 2000000, 2000000, 0, -500],  # 0 y -500 deben descartarse
        "valor_total_adjudicacion": [1200000, 1900000, 1900000, 500000, 100000],
        "fecha_de_publicacion_del": ["2023-01-01", "2023-02-01", "2023-02-01", "2023-03-01", "2023-04-01"],
    }
    df = pd.DataFrame(data)
    df_clean, report = clean_secop_dataset(df)

    # A3 y A4 descartadas por precio <= 0.
    # A2 duplicada, se debe conservar 1.
    assert len(df_clean) == 2
    assert set(df_clean["id_del_portafolio"]) == {"A1", "A2"}
    assert report["filas_descartadas_precio_invalido"] == 2
    assert report["duplicados_eliminados"] == 1


def test_add_initial_features():
    data = {
        "precio_base": [100.0, 200.0, 150.0],
        "valor_total_adjudicacion": [120.0, 180.0, 150.0],
        "duracion": [2, 30, 1],
        "unidad_de_duracion": ["Mes(es)", "día(s)", "Año(s)"],
    }
    df = pd.DataFrame(data)
    df_feat = add_initial_features(df)

    # sobrecosto_pesos
    assert df_feat["sobrecosto_pesos"].iloc[0] == 20.0
    assert df_feat["sobrecosto_pesos"].iloc[1] == -20.0
    assert df_feat["sobrecosto_pesos"].iloc[2] == 0.0

    # ratio_sobrecosto
    assert df_feat["ratio_sobrecosto"].iloc[0] == pytest.approx(0.20)
    assert df_feat["ratio_sobrecosto"].iloc[1] == pytest.approx(-0.10)
    assert df_feat["ratio_sobrecosto"].iloc[2] == 0.0

    # tiene_sobrecosto
    assert df_feat["tiene_sobrecosto"].iloc[0] == 1
    assert df_feat["tiene_sobrecosto"].iloc[1] == 0
    assert df_feat["tiene_sobrecosto"].iloc[2] == 0

    # duracion_dias_prevista
    assert df_feat["duracion_dias_prevista"].iloc[0] == 60.0
    assert df_feat["duracion_dias_prevista"].iloc[1] == 30.0
    assert df_feat["duracion_dias_prevista"].iloc[2] == 365.0
