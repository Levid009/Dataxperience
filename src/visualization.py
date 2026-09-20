"""
Módulo de Visualización Estadística y Analítica de Datos (SECOP II).
Genera gráficos de alta resolución (300 DPI) para auditoría pública y reportes ejecutivos.
"""

import os
from typing import Optional
import matplotlib
matplotlib.use("Agg")  # Backend no interactivo para entornos headless / scripts
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
import seaborn as sns


def setup_visual_style() -> None:
    """Configura el tema tipográfico y estético profesional para las visualizaciones."""
    sns.set_theme(style="whitegrid", font="sans-serif")
    plt.rcParams.update({
        "font.size": 11,
        "axes.labelsize": 12,
        "axes.titlesize": 13,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "figure.titlesize": 14,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
    })


def _format_cop(val: float, pos: Optional[int] = None) -> str:
    """Formatea valores monetarios grandes en miles de millones o millones COP."""
    if val >= 1e12:
        return f"${val/1e12:.1f}T"
    elif val >= 1e9:
        return f"${val/1e9:.1f}B"
    elif val >= 1e6:
        return f"${val/1e6:.0f}M"
    elif val >= 1e3:
        return f"${val/1e3:.0f}K"
    else:
        return f"${val:.0f}"


def plot_distribution_with_kde(
    df: pd.DataFrame,
    col: str = "valor_total_adjudicacion",
    output_path: str = "reports/figures/fig1_distribucion_montos.png",
) -> str:
    """
    Genera un Histograma con estimación de densidad por kernel (KDE) en escala logarítmica
    para exhibir la marcada asimetría (skewness) de los montos contratados,
    resaltando la discrepancia entre Media y Mediana.

    Args:
        df: DataFrame de entrada.
        col: Columna numérica a graficar.
        output_path: Ruta destino para guardar el archivo PNG.

    Returns:
        str: Ruta absoluta del archivo generado.
    """
    setup_visual_style()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    series = df[col].dropna()
    media = series.mean()
    mediana = series.median()

    fig, ax = plt.subplots(figsize=(10, 6))

    # Histograma con KDE en escala logarítmica
    sns.histplot(
        series,
        kde=True,
        log_scale=True,
        color="#1f77b4",
        edgecolor="white",
        linewidth=0.8,
        alpha=0.6,
        ax=ax,
    )

    # Líneas de referencia para Mediana y Media
    ax.axvline(
        mediana,
        color="#2ca02c",
        linestyle="--",
        linewidth=2.2,
        label=f"Mediana: {_format_cop(mediana)} COP (Contrato Típico)",
    )
    ax.axvline(
        media,
        color="#d62728",
        linestyle="-.",
        linewidth=2.2,
        label=f"Media: {_format_cop(media)} COP (Distorsionada por Megaproyectos)",
    )

    ax.set_title(
        "Distribución del Valor Adjudicado en Contratos de Obra (Cundinamarca)\n"
        "Escala Logarítmica - Evidencia de Fuerte Asimetría Positiva",
        pad=15,
        fontweight="bold",
    )
    ax.set_xlabel("Valor Total Adjudicado (COP - Escala Logarítmica)", labelpad=10)
    ax.set_ylabel("Frecuencia (Número de Contratos)", labelpad=10)
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(_format_cop))
    ax.legend(loc="upper right", frameon=True, facecolor="white", edgecolor="#cccccc")

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    return output_path


def plot_boxplots_by_modality(
    df: pd.DataFrame,
    value_col: str = "valor_total_adjudicacion",
    modality_col: str = "modalidad_de_contratacion",
    output_path: str = "reports/figures/fig2_boxplot_modalidades.png",
) -> str:
    """
    Genera un Boxplot horizontal de los montos contratados segmentado por modalidad
    de contratación en escala logarítmica, exponiendo los valores atípicos (outliers de Tukey).

    Args:
        df: DataFrame de entrada.
        value_col: Columna con valores adjudicados.
        modality_col: Columna con la modalidad administrativa.
        output_path: Ruta destino para guardar el gráfico.

    Returns:
        str: Ruta del archivo generado.
    """
    setup_visual_style()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Ordenar modalidades por mediana de valor adjudicado descendente
    order = (
        df.groupby(modality_col)[value_col]
        .median()
        .sort_values(ascending=False)
        .index
    )

    fig, ax = plt.subplots(figsize=(12, 7))

    sns.boxplot(
        data=df,
        y=modality_col,
        x=value_col,
        hue=modality_col,
        legend=False,
        order=order,
        palette="Blues_r",
        showmeans=True,
        meanprops={
            "marker": "D",
            "markerfacecolor": "#d62728",
            "markeredgecolor": "black",
            "markersize": 6,
        },
        flierprops={
            "marker": "o",
            "markerfacecolor": "#e377c2",
            "markeredgecolor": "#7f7f7f",
            "markersize": 4,
            "alpha": 0.6,
        },
        ax=ax,
    )

    ax.set_xscale("log")
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(_format_cop))

    ax.set_title(
        "Dispersión y Outliers de Monto Adjudicado según Modalidad de Contratación\n"
        "(Rombo rojo: Media | Línea central: Mediana | Puntos: Outliers Tukey)",
        pad=15,
        fontweight="bold",
    )
    ax.set_xlabel("Valor Adjudicado (COP - Escala Logarítmica)", labelpad=10)
    ax.set_ylabel("Modalidad de Contratación Pública", labelpad=10)

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    return output_path


def plot_dispersion_competition_vs_amount(
    df: pd.DataFrame,
    competition_col: str = "proveedores_unicos_con",
    value_col: str = "valor_total_adjudicacion",
    overcost_col: str = "tiene_sobrecosto",
    output_path: str = "reports/figures/fig3_dispersion_oferentes.png",
) -> str:
    """
    Genera un diagrama de dispersión (Scatter Plot) correlacionando el nivel de competencia
    (número de proponentes que ofertaron) con la cuantía adjudicada del contrato,
    diferenciando aquellos casos con desvío presupuestal / adición.

    Args:
        df: DataFrame de entrada.
        competition_col: Columna con oferentes únicos.
        value_col: Columna de monto contratado.
        overcost_col: Columna binaria indicadora de sobrecosto.
        output_path: Ruta destino del gráfico.

    Returns:
        str: Ruta del archivo generado.
    """
    setup_visual_style()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    fig, ax = plt.subplots(figsize=(11, 7))

    # Graficar contratos normales
    normales = df[df[overcost_col] == 0]
    ax.scatter(
        normales[competition_col],
        normales[value_col],
        color="#1f77b4",
        alpha=0.5,
        edgecolor="white",
        linewidth=0.5,
        s=45,
        label="Sin Sobrecosto (Normal)",
    )

    # Resaltar contratos con sobrecosto
    sobrecosto = df[df[overcost_col] == 1]
    ax.scatter(
        sobrecosto[competition_col],
        sobrecosto[value_col],
        color="#ff7f0e",
        alpha=0.9,
        edgecolor="black",
        linewidth=1.2,
        s=90,
        marker="^",
        label=f"Con Sobrecosto ({len(sobrecosto)} contratos)",
    )

    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(_format_cop))

    ax.set_title(
        "Relación entre Concurrencia de Oferentes y Cuantía del Contrato\n"
        "Auditoría de Competencia en Obras Públicas de Cundinamarca",
        pad=15,
        fontweight="bold",
    )
    ax.set_xlabel("Número de Oferentes Únicos con Respuesta", labelpad=10)
    ax.set_ylabel("Valor Total Adjudicado (COP - Escala Logarítmica)", labelpad=10)
    ax.legend(loc="upper right", frameon=True, facecolor="white", edgecolor="#cccccc")

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    return output_path
