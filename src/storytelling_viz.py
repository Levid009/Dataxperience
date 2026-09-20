"""
Módulo de Visualización y Storytelling Analítico (SECOP II).
Genera gráficos de alta resolución (300 DPI) para comunicación ejecutiva,
auditoría de contratación pública y toma de decisiones.
"""

import os
from typing import Any, Dict, Optional
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def setup_storytelling_style() -> None:
    """Configura el tema tipográfico y estético para las visualizaciones."""
    sns.set_theme(style="whitegrid", font="sans-serif")
    plt.rcParams.update({
        "font.size": 11,
        "axes.labelsize": 12,
        "axes.titlesize": 13,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "figure.titlesize": 15,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
    })


def _format_cop(val: float, pos: Optional[int] = None) -> str:
    """Formatea valores monetarios a unidades comprensibles en COP."""
    if abs(val) >= 1e12:
        return f"${val/1e12:.1f}T"
    elif abs(val) >= 1e9:
        return f"${val/1e9:.1f}B"
    elif abs(val) >= 1e6:
        return f"${val/1e6:.1f}M"
    elif abs(val) >= 1e3:
        return f"${val/1e3:.0f}K"
    return f"${val:.0f}"


def plot_prediccion_vs_real(
    y_test_log: np.ndarray,
    y_pred_log: np.ndarray,
    metrics: Dict[str, Any],
    output_path: str = "reports/figures/fig4_prediccion_vs_real.png",
) -> str:
    """Scatter plot de Valores Reales vs. Predicciones en escala logarítmica con diagonal y = x."""
    setup_storytelling_style()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    fig, ax = plt.subplots(figsize=(9, 7))

    ax.scatter(
        y_test_log,
        y_pred_log,
        color="#1f77b4",
        alpha=0.45,
        edgecolors="none",
        s=35,
        label="Contratos de Prueba (Test Set)",
    )

    min_val = min(float(np.min(y_test_log)), float(np.min(y_pred_log)))
    max_val = max(float(np.max(y_test_log)), float(np.max(y_pred_log)))
    margin = (max_val - min_val) * 0.05
    line_x = np.linspace(min_val - margin, max_val + margin, 100)

    ax.plot(
        line_x,
        line_x,
        color="#d62728",
        linestyle="--",
        linewidth=2.0,
        label="Predicción Perfecta ($y = \\hat{y}$)",
    )

    band = 0.10
    ax.fill_between(
        line_x,
        line_x - band,
        line_x + band,
        color="#2ca02c",
        alpha=0.12,
        label="Banda de Tolerancia Estrecha",
    )

    ax.set_xlim(min_val - margin, max_val + margin)
    ax.set_ylim(min_val - margin, max_val + margin)

    ax.set_title(
        "Ajuste Predictivo: Valor Real vs. Estimación del Modelo",
        fontweight="bold",
        pad=15,
    )
    ax.set_xlabel("Valor Real Observado: $\\log(1 + \\text{Valor Adjudicado})$", labelpad=8)
    ax.set_ylabel("Valor Estimado por el Modelo: $\\log(1 + \\hat{y})$", labelpad=8)

    r2_val = metrics.get("r2_real", metrics.get("r2_log", 0.0))
    mae_m = metrics.get("mae_millones_cop", 0.0)
    mape = metrics.get("mape_real", 0.0)

    textbox_text = (
        f"Métricas en Test:\n"
        f"• $R^2$ (Escala Real): {r2_val:.4f}\n"
        f"• MAE: ${mae_m:,.2f}M COP\n"
        f"• MAPE: {mape:.2f}%\n"
        f"• Contratos Evaluados: {len(y_test_log):,}"
    )
    ax.text(
        0.05,
        0.95,
        textbox_text,
        transform=ax.transAxes,
        verticalalignment="top",
        bbox=dict(boxstyle="round,pad=0.6", facecolor="white", edgecolor="#bdc3c7", alpha=0.92),
        fontsize=10.5,
        fontfamily="monospace",
    )

    ax.legend(loc="lower right", frameon=True, facecolor="white", framealpha=0.9)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close(fig)

    return os.path.abspath(output_path)


def plot_importancia_coeficientes(
    coef_df: pd.DataFrame,
    output_path: str = "reports/figures/fig5_importancia_coeficientes.png",
) -> str:
    """Gráfico de barras horizontales con los coeficientes estandarizados del modelo."""
    setup_storytelling_style()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    df_plot = coef_df.sort_values(by="coeficiente", ascending=True).copy()

    fig, ax = plt.subplots(figsize=(10, 6.5))

    colors = ["#d9534f" if c < 0 else "#1f77b4" for c in df_plot["coeficiente"]]

    bars = ax.barh(
        df_plot["variable_limpia"],
        df_plot["coeficiente"],
        color=colors,
        edgecolor="none",
        height=0.65,
        alpha=0.88,
    )

    ax.axvline(0, color="black", linewidth=1.0, linestyle="-", alpha=0.7)

    for bar in bars:
        width = bar.get_width()
        offset = 0.02 if width >= 0 else -0.02
        ha = "left" if width >= 0 else "right"
        ax.text(
            width + offset,
            bar.get_y() + bar.get_height() / 2,
            f"{width:+.4f}",
            va="center",
            ha=ha,
            fontsize=9.5,
            fontweight="bold",
            color="#2c3e50",
        )

    x_min = df_plot["coeficiente"].min()
    x_max = df_plot["coeficiente"].max()
    pad = (x_max - x_min) * 0.12
    ax.set_xlim(x_min - pad, x_max + pad)

    ax.text(
        0.02,
        0.04,
        "(-) Coeficientes Negativos: Presionan el monto a la baja (Ej. Competencia, Mínima Cuantía)",
        transform=ax.transAxes,
        fontsize=9,
        color="#d9534f",
        fontweight="semibold",
    )
    ax.text(
        0.48,
        0.95,
        "(+) Coeficientes Positivos: Incrementan el valor adjudicado (Ej. Precio Base)",
        transform=ax.transAxes,
        fontsize=9,
        color="#1f77b4",
        fontweight="semibold",
    )

    ax.set_title(
        "Impacto Relativo de Factores en el Valor Adjudicado (Coeficientes Estandarizados)",
        fontweight="bold",
        pad=15,
    )
    ax.set_xlabel("Coeficiente Estandarizado (Impacto en escala logarítmica)", labelpad=10)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close(fig)

    return os.path.abspath(output_path)


def plot_resumen_ejecutivo_storytelling(
    df: pd.DataFrame,
    y_test_real: np.ndarray,
    y_pred_real: np.ndarray,
    coef_df: pd.DataFrame,
    metrics: Dict[str, Any],
    output_path: str = "reports/figures/fig6_resumen_ejecutivo_storytelling.png",
) -> str:
    """Dashboard Multipanel Ejecutivo (2x2) con síntesis analítica y de auditoría."""
    setup_storytelling_style()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(
        "AUDITORÍA PREDICTIVA Y GOBERNANZA DE CONTRATACIÓN PÚBLICA (SECOP II - CUNDINAMARCA)",
        fontsize=15,
        fontweight="bold",
        y=0.99,
    )

    # Panel (a): Distribución
    ax_a = axes[0, 0]
    sns.kdeplot(
        df["precio_base"],
        log_scale=True,
        ax=ax_a,
        color="#7f7f7f",
        linestyle="--",
        label="Precio Base Presupuestado",
        fill=True,
        alpha=0.15,
    )
    sns.kdeplot(
        df["valor_total_adjudicacion"],
        log_scale=True,
        ax=ax_a,
        color="#1f77b4",
        label="Valor Total Adjudicado",
        fill=True,
        alpha=0.25,
    )
    ax_a.set_title("(a) Distribución Presupuesto vs. Adjudicación", fontweight="bold")
    ax_a.set_xlabel("Monto del Contrato (Escala Logarítmica COP)")
    ax_a.set_ylabel("Densidad Estimada")
    ax_a.legend(loc="upper left")

    # Panel (b): Competencia y Ahorro
    ax_b = axes[0, 1]
    df_clean = df.copy()
    df_clean["ratio_adjudicacion"] = (
        df_clean["valor_total_adjudicacion"] / df_clean["precio_base"]
    )
    ratio_subset = df_clean[
        (df_clean["ratio_adjudicacion"] > 0.4) & (df_clean["ratio_adjudicacion"] <= 1.2)
    ]
    ax_b.scatter(
        ratio_subset["proveedores_unicos_con"],
        ratio_subset["ratio_adjudicacion"] * 100,
        alpha=0.4,
        color="#2ca02c",
        edgecolors="none",
        s=30,
    )
    ax_b.axhline(100, color="#d62728", linestyle=":", linewidth=1.5, label="100% (Sin Descuento)")
    ax_b.set_title("(b) Competencia: Oferentes vs. % Adjudicado sobre Base", fontweight="bold")
    ax_b.set_xlabel("Número de Oferentes Únicos Concurrentes")
    ax_b.set_ylabel("% Adjudicado vs. Presupuesto Base")
    ax_b.set_xlim(0, min(50, ratio_subset["proveedores_unicos_con"].max()))
    ax_b.legend(loc="lower left")

    # Panel (c): Residuos y Error
    ax_c = axes[1, 0]
    pct_errors = ((y_pred_real - y_test_real) / y_test_real) * 100.0
    pct_errors_clip = np.clip(pct_errors, -30, 30)

    sns.histplot(
        pct_errors_clip,
        bins=35,
        kde=True,
        ax=ax_c,
        color="#34495e",
        edgecolor="white",
        alpha=0.7,
    )
    ax_c.axvline(0, color="#27ae60", linestyle="--", linewidth=1.8, label="Error Cero (Predicción Exacta)")
    ax_c.set_title(
        f"(c) Precisión del Modelo (MAPE: {metrics.get('mape_real', 0.0):.2f}% | $R^2$: {metrics.get('r2_real', 0.0):.4f})",
        fontweight="bold",
    )
    ax_c.set_xlabel("Error Relativo Porcentual: $(\\hat{y} - y) / y$ (%)")
    ax_c.set_ylabel("Frecuencia de Contratos")
    ax_c.legend(loc="upper right")

    # Panel (d): Matriz de Alertas
    ax_d = axes[1, 1]
    residuos_millones = (y_test_real - y_pred_real) / 1e6

    colores_riesgo = []
    for res, real in zip(residuos_millones, y_test_real):
        if res > 100.0:
            colores_riesgo.append("#e74c3c")
        elif res < -100.0:
            colores_riesgo.append("#f39c12")
        else:
            colores_riesgo.append("#2980b9")

    ax_d.scatter(
        y_pred_real / 1e6,
        y_test_real / 1e6,
        c=colores_riesgo,
        alpha=0.55,
        s=35,
        edgecolors="none",
    )

    max_scope = max(float(np.percentile(y_pred_real / 1e6, 98)), 5000)
    ax_d.plot([0, max_scope], [0, max_scope], color="gray", linestyle="--", linewidth=1.2)
    ax_d.set_xlim(0, max_scope)
    ax_d.set_ylim(0, max_scope)

    ax_d.set_title("(d) Matriz de Alertas Tempranas de Auditoría", fontweight="bold")
    ax_d.set_xlabel("Valor Estimado por el Modelo (Millones COP)")
    ax_d.set_ylabel("Valor Real Adjudicado (Millones COP)")

    ax_d.scatter([], [], c="#2980b9", label="Normal (Alineado)", s=30)
    ax_d.scatter([], [], c="#e74c3c", label="Alerta Alta (Sobreprecio)", s=30)
    ax_d.scatter([], [], c="#f39c12", label="Alerta Media (Descuento Anormal)", s=30)
    ax_d.legend(loc="upper left", frameon=True)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close(fig)

    return os.path.abspath(output_path)
