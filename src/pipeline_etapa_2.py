"""
Pipeline de Ejecución - Etapa 2: Análisis Estadístico y Exploratorio (EDA).
Auditoría Predictiva de Contratación Pública en Cundinamarca (SECOP II).
"""

import os
import sys
import argparse
import pandas as pd

# Remover el directorio 'src' de sys.path para no hacer shadow a la librería estándar 'statistics'
script_dir = os.path.dirname(os.path.abspath(__file__))
while script_dir in sys.path:
    sys.path.remove(script_dir)

# Asegurar que la raíz del proyecto esté en sys.path
project_root = os.path.abspath(os.path.join(script_dir, ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.statistics import (
    calculate_central_tendency_and_skewness,
    calculate_dispersion_measures,
    segment_dispersion_by_modality,
    detect_tukey_outliers,
    extract_top_outliers,
    hypothesis_contrast_competition,
)
from src.visualization import (
    plot_distribution_with_kde,
    plot_boxplots_by_modality,
    plot_dispersion_competition_vs_amount,
)


def _fmt_money(val: float) -> str:
    """Formatea valores monetarios a pesos colombianos con separador de miles."""
    if pd.isna(val):
        return "N/A"
    return f"${val:,.2f}"


def run_pipeline_etapa_2(
    input_path: str = "data/processed/secop_cundinamarca_obras_clean.csv",
    figures_dir: str = "reports/figures",
) -> dict:
    """
    Ejecuta el flujo integral de análisis estadístico de la Etapa 2:
    - Tendencia central y análisis de asimetrías.
    - Dispersión y segmentación por modalidad.
    - Detección de anomalías y outliers (Tukey IQR).
    - Contraste de hipótesis sobre concurrencia y competencia.
    - Generación y almacenamiento de 3 figuras gráficas en alta definición.

    Args:
        input_path: Ruta del CSV procesado y saneado.
        figures_dir: Directorio donde almacenar las figuras generadas.

    Returns:
        dict: Diccionario con los DataFrames y métricas calculadas.
    """
    print("=" * 90)
    print("ETAPA 2: ANÁLISIS ESTADÍSTICO DESCRIPTIVO Y DETECCIÓN DE ANOMALÍAS (SECOP II)")
    print("=" * 90)

    if not os.path.exists(input_path):
        raise FileNotFoundError(
            f"No se encontró el archivo de entrada: '{input_path}'. "
            "Por favor ejecuta primero 'python src/pipeline_etapa_1.py'."
        )

    os.makedirs(figures_dir, exist_ok=True)

    # 1. Carga del Dataset Limpio
    print(f"\n[1/5] Cargando dataset limpio: {input_path}...")
    df = pd.read_csv(input_path, encoding="utf-8")
    print(f"      -> Total contratos cargados: {len(df):,}")

    numeric_vars = [
        "precio_base",
        "valor_total_adjudicacion",
        "sobrecosto_pesos",
        "duracion_dias_prevista",
    ]

    pd.set_option("display.max_columns", 10)
    pd.set_option("display.width", 1000)

    # 2. Medidas de Tendencia Central y Asimetría
    print("\n[2/5] Calculando medidas de tendencia central y asimetría (Media vs. Mediana)...")
    central_df = calculate_central_tendency_and_skewness(df, numeric_vars)

    print("\n" + "-" * 90)
    print("TABLA 1: MEDIDAS DE TENDENCIA CENTRAL Y COEFICIENTE DE ASIMETRÍA (SKEWNESS)")
    print("-" * 90)
    display_central = central_df.astype(object)
    for row_idx in display_central.index:
        for c in ["media", "mediana", "moda"]:
            val = central_df.loc[row_idx, c]
            if "duracion" in row_idx:
                display_central.loc[row_idx, c] = f"{val:,.1f} días" if pd.notna(val) else "N/A"
            else:
                display_central.loc[row_idx, c] = _fmt_money(val)
    display_central["skewness"] = central_df["skewness"].map("{:,.4f}".format)
    display_central["divergencia_media_vs_mediana_pct"] = central_df[
        "divergencia_media_vs_mediana_pct"
    ].map("{:+,.2f}%".format)
    print(display_central[["media", "mediana", "skewness", "divergencia_media_vs_mediana_pct", "interpretacion_sesgo"]])

    print("\n>>> DICTAMEN DE AUDITORÍA (POR QUÉ LA MEDIANA ES EL CONTRATO TÍPICO):")
    media_val = central_df.loc["valor_total_adjudicacion", "media"]
    mediana_val = central_df.loc["valor_total_adjudicacion", "mediana"]
    skew_val = central_df.loc["valor_total_adjudicacion", "skewness"]
    print(
        f"    El 'valor_total_adjudicacion' exhibe un coeficiente de asimetría de {skew_val:.2f} (sesgo positivo extremo).\n"
        f"    La media aritmética ({_fmt_money(media_val)}) supera en un "
        f"{central_df.loc['valor_total_adjudicacion', 'divergencia_media_vs_mediana_pct']:+.1f}% a la mediana ({_fmt_money(mediana_val)}).\n"
        f"    En contratación pública, la MEDIA está severamente distorsionada por un reducido número de megaproyectos\n"
        f"    (outliers superiores de hasta $219 mil millones). Por ende, LA MEDIANA ({_fmt_money(mediana_val)}) representa\n"
        f"    de forma fidedigna y robusta el contrato de obra típico contratado en los municipios de Cundinamarca."
    )

    # 3. Medidas de Dispersión y Variabilidad
    print("\n[3/5] Calculando dispersión y segmentación por modalidad de contratación...")
    dispersion_df = calculate_dispersion_measures(df, numeric_vars)

    print("\n" + "-" * 90)
    print("TABLA 2: MEDIDAS DE DISPERSIÓN Y VARIABILIDAD PARAMÉTRICA Y NO PARAMÉTRICA")
    print("-" * 90)
    display_disp = dispersion_df.astype(object)
    for row_idx in display_disp.index:
        for col in ["desviacion_estandar", "min", "max", "rango", "q1", "q3", "iqr"]:
            val = dispersion_df.loc[row_idx, col]
            if "duracion" in row_idx:
                display_disp.loc[row_idx, col] = f"{val:,.1f} días" if pd.notna(val) else "N/A"
            else:
                display_disp.loc[row_idx, col] = _fmt_money(val)
    display_disp["coeficiente_variacion_pct"] = dispersion_df["coeficiente_variacion_pct"].map("{:,.2f}%".format)
    print(display_disp[["desviacion_estandar", "q1", "q3", "iqr", "rango", "coeficiente_variacion_pct"]])

    # Segmentación por Modalidad
    modality_disp = segment_dispersion_by_modality(df, "valor_total_adjudicacion", "modalidad_de_contratacion")
    print("\n" + "-" * 90)
    print("TABLA 3: DISPERSIÓN DEL VALOR ADJUDICADO POR MODALIDAD DE CONTRATACIÓN")
    print("-" * 90)
    display_mod = modality_disp.copy()
    for col in ["media", "mediana", "std", "q1", "q3", "iqr"]:
        display_mod[col] = display_mod[col].apply(_fmt_money)
    display_mod["porcentaje_contratos"] = display_mod["porcentaje_contratos"].map("{:.2f}%".format)
    print(display_mod[["conteo", "porcentaje_contratos", "mediana", "iqr", "media", "std"]])

    # 4. Outliers de Tukey y Casos Críticos
    print("\n[4/5] Aplicando Criterio de Tukey (IQR) para detección de anomalías y casos críticos...")
    tukey_valor = detect_tukey_outliers(df["valor_total_adjudicacion"])
    tukey_duracion = detect_tukey_outliers(df["duracion_dias_prevista"])

    print(f"\n* Umbral Tukey Monto: Q3 + 1.5*IQR = {_fmt_money(tukey_valor['limite_superior'])}")
    print(f"  -> Contratos Outliers Superiores: {tukey_valor['conteo_outliers_superiores']} ({tukey_valor['porcentaje_outliers']:.2f}% del total)")

    top10_monto = extract_top_outliers(df, "valor_total_adjudicacion", top_n=10)
    print("\n" + "-" * 90)
    print("TABLA 4: TOP 10 CONTRATOS ATÍPICOS SUPERIORES POR MONTO (OUTLIERS TUKEY)")
    print("-" * 90)
    display_top_monto = top10_monto[[
        "id_del_portafolio", "entidad", "ciudad_entidad",
        "modalidad_de_contratacion", "valor_total_adjudicacion", "duracion_dias_prevista"
    ]].copy()
    display_top_monto["valor_total_adjudicacion"] = display_top_monto["valor_total_adjudicacion"].apply(_fmt_money)
    print(display_top_monto.to_string(index=False))

    top10_duracion = extract_top_outliers(df, "duracion_dias_prevista", top_n=10)
    print("\n" + "-" * 90)
    print("TABLA 5: TOP 10 CONTRATOS ATÍPICOS SUPERIORES POR DURACIÓN PREVISTA (DÍAS)")
    print("-" * 90)
    display_top_dur = top10_duracion[[
        "id_del_portafolio", "entidad", "ciudad_entidad",
        "duracion_dias_prevista", "valor_total_adjudicacion"
    ]].copy()
    display_top_dur["valor_total_adjudicacion"] = display_top_dur["valor_total_adjudicacion"].apply(_fmt_money)
    print(display_top_dur.to_string(index=False))

    # Contraste de Hipótesis: Competencia vs Monto y Sobrecosto
    competition_df = hypothesis_contrast_competition(df)
    print("\n" + "-" * 90)
    print("TABLA 6: CONTRASTE DESCRIPTIVO - CONCURRENCIA DE OFERENTES VS. ADJUDICACIÓN Y RIESGO")
    print("-" * 90)
    display_comp = competition_df.copy()
    display_comp["mediana_valor_adjudicado"] = display_comp["mediana_valor_adjudicado"].apply(_fmt_money)
    display_comp["media_valor_adjudicado"] = display_comp["media_valor_adjudicado"].apply(_fmt_money)
    display_comp["tasa_sobrecosto_pct"] = display_comp["tasa_sobrecosto_pct"].map("{:.2f}%".format)
    display_comp["porcentaje_del_total"] = display_comp["porcentaje_del_total"].map("{:.2f}%".format)
    print(display_comp[["total_contratos", "porcentaje_del_total", "mediana_valor_adjudicado", "tasa_sobrecosto_pct", "mediana_duracion_dias"]])

    # 5. Generación de Figuras de Visualización
    print(f"\n[5/5] Generando figuras de alta resolución (300 DPI) en '{figures_dir}'...")

    f1 = plot_distribution_with_kde(
        df, "valor_total_adjudicacion", os.path.join(figures_dir, "fig1_distribucion_montos.png")
    )
    print(f"      -> Guardado: {f1}")

    f2 = plot_boxplots_by_modality(
        df, "valor_total_adjudicacion", "modalidad_de_contratacion", os.path.join(figures_dir, "fig2_boxplot_modalidades.png")
    )
    print(f"      -> Guardado: {f2}")

    f3 = plot_dispersion_competition_vs_amount(
        df, "proveedores_unicos_con", "valor_total_adjudicacion", "tiene_sobrecosto", os.path.join(figures_dir, "fig3_dispersion_oferentes.png")
    )
    print(f"      -> Guardado: {f3}")

    print("\n" + "=" * 90)
    print("ETAPA 2 FINALIZADA EXITOSAMENTE - TODAS LAS MÉTRICAS Y FIGURAS GENERADAS")
    print("=" * 90)

    return {
        "central_df": central_df,
        "dispersion_df": dispersion_df,
        "modality_disp": modality_disp,
        "tukey_valor": tukey_valor,
        "top10_monto": top10_monto,
        "top10_duracion": top10_duracion,
        "competition_df": competition_df,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pipeline Etapa 2 - Análisis Estadístico SECOP II")
    parser.add_argument(
        "--input",
        default="data/processed/secop_cundinamarca_obras_clean.csv",
        help="Ruta del archivo CSV limpio de entrada",
    )
    parser.add_argument(
        "--figures-dir",
        default="reports/figures",
        help="Directorio de destino para las figuras PNG",
    )
    args = parser.parse_args()

    run_pipeline_etapa_2(input_path=args.input, figures_dir=args.figures_dir)
