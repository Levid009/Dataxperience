"""
Pruebas unitarias automatizadas para la Etapa 3: Modelado Predictivo y Storytelling.
Ejecutar con: pytest tests/test_etapa_3.py -v
"""

import os
import numpy as np
import pandas as pd
import pytest
from sklearn.pipeline import Pipeline

from src.models import (
    prepare_dataset,
    build_preprocessor,
    build_model_pipeline,
    train_and_evaluate,
    get_feature_coefficients,
    save_model,
    load_model,
    calculate_mape,
)
from src.storytelling_viz import (
    plot_prediccion_vs_real,
    plot_importancia_coeficientes,
    plot_resumen_ejecutivo_storytelling,
)
from src.pipeline_etapa_3 import run_pipeline_etapa_3


@pytest.fixture
def sample_contracts_df():
    np.random.seed(42)
    n = 120

    base_prices = np.random.uniform(5e7, 2e9, size=n)
    oferentes = np.random.randint(1, 15, size=n)
    discounts = 1.0 - (oferentes * 0.005) + np.random.normal(0, 0.02, size=n)
    adjudicado = np.clip(base_prices * discounts, 1e6, None)

    modalidades = np.random.choice(
        [
            "Licitación Pública Obra Publica",
            "Selección Abreviada de Menor Cuantía",
            "Mínima Cuantía",
            "Contratación Directa (con ofertas)",
        ],
        size=n,
    )

    return pd.DataFrame(
        {
            "id_del_portafolio": [f"PORT_{i}" for i in range(n)],
            "referencia_del_proceso": [f"REF_{i}" for i in range(n)],
            "precio_base": base_prices,
            "valor_total_adjudicacion": adjudicado,
            "duracion_dias_prevista": np.random.randint(15, 365, size=n).astype(float),
            "proveedores_unicos_con": oferentes.astype(float),
            "modalidad_de_contratacion": modalidades,
        }
    )


def test_prepare_dataset(sample_contracts_df):
    X, y = prepare_dataset(sample_contracts_df)

    assert len(X) == len(sample_contracts_df)
    assert len(y) == len(sample_contracts_df)
    assert "precio_base_log" in X.columns
    assert "duracion_dias_prevista" in X.columns
    assert "proveedores_unicos_con" in X.columns
    assert "modalidad_de_contratacion" in X.columns

    np.testing.assert_allclose(
        X["precio_base_log"].values,
        np.log1p(sample_contracts_df["precio_base"].values),
        rtol=1e-5,
    )
    np.testing.assert_allclose(
        y.values,
        np.log1p(sample_contracts_df["valor_total_adjudicacion"].values),
        rtol=1e-5,
    )


def test_prepare_dataset_missing_column():
    bad_df = pd.DataFrame({"precio_base": [100.0]})
    with pytest.raises(ValueError, match="Faltan columnas requeridas"):
        prepare_dataset(bad_df)


def test_build_preprocessor(sample_contracts_df):
    X, _ = prepare_dataset(sample_contracts_df)
    preprocessor = build_preprocessor()

    transformed = preprocessor.fit_transform(X)
    assert transformed.shape[0] == len(X)
    assert transformed.shape[1] > 3
    assert not np.isnan(transformed).any()

    feature_names = preprocessor.get_feature_names_out()
    assert len(feature_names) == transformed.shape[1]
    assert any("precio_base_log" in fn for fn in feature_names)


def test_build_model_pipeline_invalid_type():
    with pytest.raises(ValueError, match="Tipo de modelo no soportado"):
        build_model_pipeline(model_type="invalid_model")


def test_calculate_mape():
    y_true = np.array([100.0, 200.0, 300.0])
    y_pred = np.array([110.0, 180.0, 300.0])
    mape = calculate_mape(y_true, y_pred)
    assert pytest.approx(mape, 0.01) == 6.6667


def test_train_and_evaluate(sample_contracts_df):
    X, y = prepare_dataset(sample_contracts_df)
    n_train = 90
    X_train, X_test = X.iloc[:n_train], X.iloc[n_train:]
    y_train, y_test = y.iloc[:n_train], y.iloc[n_train:]

    pipeline = build_model_pipeline(model_type="ridge", alpha=1.0)
    results = train_and_evaluate(pipeline, X_train, y_train, X_test, y_test)

    metrics = results["metrics"]
    assert "r2_real" in metrics
    assert "r2_log" in metrics
    assert "mae_real" in metrics
    assert "mae_millones_cop" in metrics
    assert "mape_real" in metrics

    assert metrics["r2_real"] > 0.85
    assert metrics["r2_log"] > 0.85
    assert metrics["mae_real"] > 0
    assert metrics["mape_real"] > 0
    assert len(results["y_pred_real"]) == len(X_test)


def test_get_feature_coefficients(sample_contracts_df):
    X, y = prepare_dataset(sample_contracts_df)
    pipeline = build_model_pipeline(model_type="ridge")
    pipeline.fit(X, y)

    coef_df = get_feature_coefficients(pipeline)
    assert isinstance(coef_df, pd.DataFrame)
    assert "variable" in coef_df.columns
    assert "variable_limpia" in coef_df.columns
    assert "coeficiente" in coef_df.columns
    assert "direccion" in coef_df.columns
    assert "magnitud" in coef_df.columns

    magnitudes = coef_df["magnitud"].tolist()
    assert magnitudes == sorted(magnitudes, reverse=True)


def test_save_and_load_model(sample_contracts_df, tmp_path):
    X, y = prepare_dataset(sample_contracts_df)
    pipeline = build_model_pipeline(model_type="ridge")
    pipeline.fit(X, y)

    saved_file = tmp_path / "model.joblib"
    path_ret = save_model(pipeline, str(saved_file))
    assert os.path.exists(path_ret)

    loaded_pipeline = load_model(str(saved_file))
    assert isinstance(loaded_pipeline, Pipeline)

    pred_orig = pipeline.predict(X.iloc[:5])
    pred_load = loaded_pipeline.predict(X.iloc[:5])
    np.testing.assert_allclose(pred_orig, pred_load)


def test_storytelling_visualizations(sample_contracts_df, tmp_path):
    X, y = prepare_dataset(sample_contracts_df)
    pipeline = build_model_pipeline()
    results = train_and_evaluate(pipeline, X.iloc[:90], y.iloc[:90], X.iloc[90:], y.iloc[90:])
    coef_df = get_feature_coefficients(results["pipeline"])

    fig4_path = str(tmp_path / "fig4_test.png")
    fig5_path = str(tmp_path / "fig5_test.png")
    fig6_path = str(tmp_path / "fig6_test.png")

    p4 = plot_prediccion_vs_real(results["y_test_log"], results["y_pred_log"], results["metrics"], fig4_path)
    p5 = plot_importancia_coeficientes(coef_df, fig5_path)
    p6 = plot_resumen_ejecutivo_storytelling(
        sample_contracts_df,
        results["y_test_real"],
        results["y_pred_real"],
        coef_df,
        results["metrics"],
        fig6_path,
    )

    for p in [p4, p5, p6]:
        assert os.path.exists(p)
        assert os.path.getsize(p) > 15000


def test_run_pipeline_etapa_3_integration(tmp_path):
    input_csv = "data/processed/secop_cundinamarca_obras_clean.csv"
    if not os.path.exists(input_csv):
        pytest.skip(f"No existe el archivo de datos {input_csv}")

    model_dir = tmp_path / "models"
    figures_dir = tmp_path / "figures"
    model_file = str(model_dir / "test_model.joblib")

    res = run_pipeline_etapa_3(
        input_path=input_csv,
        model_path=model_file,
        figures_dir=str(figures_dir),
    )

    assert "pipeline" in res
    assert "metrics" in res
    assert os.path.exists(model_file)
    for fig in res["figure_paths"]:
        assert os.path.exists(fig)
