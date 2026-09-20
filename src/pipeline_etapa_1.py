"""
Pipeline de Ejecución - Etapa 1: Ingesta, Limpieza y Feature Engineering Inicial.
Auditoría Predictiva de Contratación Pública en Cundinamarca (SECOP II).
"""

import os
import sys
import argparse
import pandas as pd

# Asegurar que el directorio raíz esté en sys.path al ejecutar directamente el script
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.cleaning import clean_secop_dataset
from src.features import add_initial_features


def run_pipeline_etapa_1(
    input_path: str = "data/raw/secop_obras_cundinamarca.csv",
    output_path: str = "data/processed/secop_cundinamarca_obras_clean.csv",
    chunksize: int = 10000,
) -> pd.DataFrame:
    """
    Ejecuta el flujo completo de la Etapa 1:
    - Carga del dataset crudo.
    - Limpieza, filtrado de inconsistencias y deduplicación.
    - Feature engineering de sobrecosto y duración estandarizada.
    - Almacenamiento en 'data/processed/secop_cundinamarca_obras_clean.csv'.
    - Impresión de métricas de calidad y auditoría.

    Args:
        input_path: Ruta del CSV crudo de entrada.
        output_path: Ruta destino del CSV limpio procesado.
        chunksize: Tamaño de lote si se requiere procesamiento por bloques.

    Returns:
        pd.DataFrame: DataFrame procesado y validado.
    """
    print("=" * 80)
    print("ETAPA 1: AUDITORÍA, LIMPIEZA Y FEATURE ENGINEERING - SECOP II (CUNDINAMARCA)")
    print("=" * 80)

    if not os.path.exists(input_path):
        raise FileNotFoundError(
            f"No se encontró el archivo de entrada: '{input_path}'. "
            "Por favor ejecuta primero 'conexion-secop.py' para descargar los datos."
        )

    # 1. Carga del dataset crudo
    print(f"\n[1/4] Leyendo dataset de entrada: {input_path}...")
    df_raw = pd.read_csv(input_path, encoding="utf-8")
    filas_iniciales = len(df_raw)
    print(f"      -> Total filas leídas: {filas_iniciales:,}")
    print(f"      -> Columnas disponibles: {list(df_raw.columns)}")

    # 2. Limpieza y Auditoría de Calidad
    print("\n[2/4] Ejecutando saneamiento y auditoría de calidad...")
    df_clean, audit_report = clean_secop_dataset(
        df=df_raw,
        id_column="id_del_portafolio",
        price_col="precio_base",
        award_col="valor_total_adjudicacion",
        date_cols=["fecha_de_publicacion_del"],
    )
    print(f"      -> Filas descartadas (precio_base <= 0 o inválido): {audit_report['filas_descartadas_precio_invalido']}")
    print(f"      -> Duplicados removidos (por id_del_portafolio):    {audit_report['duplicados_eliminados']}")
    print(f"      -> Filas depuradas:                                {audit_report['filas_finales']:,} ({audit_report['porcentaje_retencion']}% retenido)")

    # 3. Feature Engineering Inicial
    print("\n[3/4] Generando variables ingenieriles...")
    df_features = add_initial_features(
        df=df_clean,
        price_col="precio_base",
        award_col="valor_total_adjudicacion",
        duration_col="duracion",
        unit_col="unidad_de_duracion",
    )
    print("      -> 'sobrecosto_pesos'        (valor_total_adjudicacion - precio_base)")
    print("      -> 'ratio_sobrecosto'        (sobrecosto_pesos / precio_base)")
    print("      -> 'tiene_sobrecosto'        (1 si sobrecosto_pesos > 0 else 0)")
    print("      -> 'duracion_dias_prevista'  (días normalizados según meses/días/años)")

    # 4. Exportación del Dataset Final
    print(f"\n[4/4] Guardando dataset limpio en: {output_path}...")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df_features.to_csv(output_path, index=False, encoding="utf-8")
    print(f"      -> Guardado exitoso ({len(df_features):,} registros).")

    # 5. Reporte Ejecutivo de Calidad
    print("\n" + "=" * 80)
    print("REPORTE EJECUTIVO DE CALIDAD Y AUDITORÍA DE DATOS")
    print("=" * 80)

    porcentaje_sobrecosto = df_features["tiene_sobrecosto"].mean() * 100
    conteo_sobrecosto = df_features["tiene_sobrecosto"].sum()
    filas_finales = len(df_features)

    print(f"\n1. Balance de Registros:")
    print(f"   - Filas Iniciales: {filas_iniciales:,}")
    print(f"   - Filas Finales:   {filas_finales:,}")
    print(f"   - Tasa de retención de datos limpios: {(filas_finales / filas_iniciales) * 100:.2f}%")

    print(f"\n2. Métricas de Sobrecosto Presupuestal:")
    print(f"   - Contratos con Sobrecosto (Adición): {conteo_sobrecosto:,} de {filas_finales:,}")
    print(f"   - Porcentaje Exacto con Sobrecosto:  {porcentaje_sobrecosto:.4f}%")

    print(f"\n3. Resumen Estadístico de Columnas Numéricas (.describe()):")
    numeric_cols = [
        "precio_base",
        "valor_total_adjudicacion",
        "sobrecosto_pesos",
        "ratio_sobrecosto",
        "duracion_dias_prevista",
        "proveedores_unicos_con",
    ]
    summary_df = df_features[numeric_cols].describe().T
    pd.set_option("display.float_format", lambda x: f"{x:,.2f}")
    pd.set_option("display.max_columns", 10)
    pd.set_option("display.width", 1000)
    print(summary_df[["count", "mean", "std", "min", "50%", "max"]])

    print(f"\n4. Resumen de Tipos de Datos y Valores Nulos:")
    types_df = pd.DataFrame({
        "Dtype": df_features.dtypes,
        "No Nulos": df_features.notnull().sum(),
        "Nulos": df_features.isnull().sum(),
        "% Nulos": (df_features.isnull().sum() / filas_finales * 100).round(2)
    })
    print(types_df)

    print("\n" + "=" * 80)
    print("PROCESO DE ETAPA 1 FINALIZADO EXITOSAMENTE")
    print("=" * 80)

    return df_features


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pipeline Etapa 1 SECOP II")
    parser.add_argument(
        "--input",
        default="data/raw/secop_obras_cundinamarca.csv",
        help="Ruta del archivo CSV de entrada",
    )
    parser.add_argument(
        "--output",
        default="data/processed/secop_cundinamarca_obras_clean.csv",
        help="Ruta del archivo CSV de salida depurado",
    )
    args = parser.parse_args()

    run_pipeline_etapa_1(input_path=args.input, output_path=args.output)
