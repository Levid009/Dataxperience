"""
Pruebas unitarias automatizadas para la Etapa 2: Análisis Estadístico y Outliers.
Ejecutar con: pytest tests/test_etapa_2.py -v
"""

import numpy as np
import pandas as pd
import pytest

from src.statistics import (
    calculate_central_tendency_and_skewness,
    calculate_dispersion_measures,
    segment_dispersion_by_modality,
    detect_tukey_outliers,
    extract_top_outliers,
    hypothesis_contrast_competition,
)


@pytest.fixture
def sample_data():
    """Genera un DataFrame controlado con propiedades estadísticas conocidas."""
    np.random.seed(42)
    # 90 valores normales alrededor de 100, más 10 outliers superiores
    base_values = np.random.normal(loc=100, scale=10, size=90)
    outliers = np.array([500, 600, 700, 800, 900, 1000, 1200, 1500, 1800, 2000])
    values = np.concatenate([base_values, outliers])

    modalidades = ["Licitación Pública"] * 40 + ["Selección Abreviada"] * 40 + ["Mínima Cuantía"] * 20
    oferentes = [1] * 25 + [2] * 25 + [5] * 25 + [15] * 25
    tiene_sobrecosto = [0] * 90 + [1] * 10

    df = pd.DataFrame({
        "id_del_portafolio": [f"ID_{i}" for i in range(100)],
        "referencia_del_proceso": [f"REF_{i}" for i in range(100)],
        "entidad": ["Alcaldía Test"] * 100,
        "ciudad_entidad": ["Bogotá"] * 100,
        "valor_total_adjudicacion": values,
        "precio_base": values * 0.95,
        "sobrecosto_pesos": [50.0] * 100,
        "duracion_dias_prevista": np.random.randint(30, 180, size=100),
        "modalidad_de_contratacion": modalidades,
        "proveedores_unicos_con": oferentes,
        "tiene_sobrecosto": tiene_sobrecosto,
    })
    return df


def test_calculate_central_tendency_and_skewness(sample_data):
    df_res = calculate_central_tendency_and_skewness(sample_data, ["valor_total_adjudicacion"])

    assert "valor_total_adjudicacion" in df_res.index
    media = df_res.loc["valor_total_adjudicacion", "media"]
    mediana = df_res.loc["valor_total_adjudicacion", "mediana"]
    skewness = df_res.loc["valor_total_adjudicacion", "skewness"]

    # Con 10 outliers extremos, la media debe ser significativamente mayor que la mediana
    assert media > mediana
    assert skewness > 1.0  # Fuerte asimetría positiva
    assert "Positiva" in df_res.loc["valor_total_adjudicacion", "interpretacion_sesgo"]


def test_calculate_dispersion_measures(sample_data):
    df_res = calculate_dispersion_measures(sample_data, ["valor_total_adjudicacion"])

    assert "valor_total_adjudicacion" in df_res.index
    std = df_res.loc["valor_total_adjudicacion", "desviacion_estandar"]
    iqr = df_res.loc["valor_total_adjudicacion", "iqr"]
    rango = df_res.loc["valor_total_adjudicacion", "rango"]

    assert std > 0
    assert iqr > 0
    assert rango > iqr


def test_segment_dispersion_by_modality(sample_data):
    df_res = segment_dispersion_by_modality(sample_data, "valor_total_adjudicacion", "modalidad_de_contratacion")

    assert len(df_res) == 3
    assert set(df_res.index) == {"Licitación Pública", "Selección Abreviada", "Mínima Cuantía"}
    assert df_res.loc["Licitación Pública", "conteo"] == 40
    assert df_res.loc["Mínima Cuantía", "conteo"] == 20


def test_detect_tukey_outliers():
    # Serie con 10 valores entre 10 y 20, y 2 outliers claros (100 y 200)
    series = pd.Series([10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 100, 200])
    res = detect_tukey_outliers(series, k=1.5)

    assert res["conteo_outliers_superiores"] >= 2
    assert res["limite_superior"] < 100
    assert res["limite_inferior"] < 10
    assert res["porcentaje_outliers"] > 0


def test_extract_top_outliers(sample_data):
    top10 = extract_top_outliers(sample_data, "valor_total_adjudicacion", top_n=5)

    assert len(top10) == 5
    # Debe estar ordenado de forma estrictamente descendente
    valores = top10["valor_total_adjudicacion"].tolist()
    assert valores == sorted(valores, reverse=True)


def test_hypothesis_contrast_competition(sample_data):
    comp_df = hypothesis_contrast_competition(sample_data)

    assert not comp_df.empty
    assert "mediana_valor_adjudicado" in comp_df.columns
    assert "tasa_sobrecosto_pct" in comp_df.columns
    assert comp_df["total_contratos"].sum() == 100
