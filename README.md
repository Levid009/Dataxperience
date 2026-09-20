# Auditoría Predictiva y Gobernanza de Contratación Pública (SECOP II)

[![Python](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/)
[![Pandas](https://img.shields.io/badge/Pandas-2.2-orange.svg)](https://pandas.pydata.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.9-green.svg)](https://scikit-learn.org/)
[![Tests](https://img.shields.io/badge/Tests-21%20Passed-brightgreen.svg)](https://pytest.org/)
[![Status](https://img.shields.io/badge/Status-Soluci%C3%B3n%20Integral%20Completada-success.svg)](#)

Plataforma integral de **Data Engineering, Auditoría Estadística y Machine Learning** diseñada para fiscalizar, analizar y predecir adjudicaciones en contratos de obra pública del departamento de Cundinamarca a partir de datos oficiales de **SECOP II** (*datos.gov.co*).

---

## 📺 Demostración en Video del Proyecto

| 🎬 **Etapa 1: Ingesta, Calidad y Limpieza de Datos** | 🎬 **Etapa 2: Análisis Estadístico y Detección de Anomalías** |
| :---: | :---: |
| [![Video Demostración Etapa 1](https://img.youtube.com/vi/0IS5f_Mr-ik/maxresdefault.jpg)](https://youtu.be/0IS5f_Mr-ik) | [![Video Demostración Etapa 2](https://img.youtube.com/vi/EQvxJx8ZS-g/maxresdefault.jpg)](https://www.youtube.com/watch?v=EQvxJx8ZS-g) |
| [▶️ Ver Video Explicativo - Etapa 1](https://youtu.be/0IS5f_Mr-ik) | [▶️ Ver Video Explicativo - Etapa 2](https://www.youtube.com/watch?v=EQvxJx8ZS-g) |

---

## 🏛️ Visión General y Arquitectura

El sistema implementa un flujo de trabajo analítico integral estructurado en tres pilares complementarios:

```text
                  ┌────────────────────────────────────────┐
                  │    Extracción SODA API (datos.gov.co)  │
                  └──────────────────┬─────────────────────┘
                                     ▼
        ┌─────────────────────────────────────────────────────────┐
        │  1. DATA ENGINEERING & SANEAMIENTO TRANSACCIONAL        │
        │     • Limpieza de divisas, normalización de fechas      │
        │     • Validación de consistencia financiera y nulos     │
        │     • Deduplicación y feature engineering inicial       │
        └────────────────────────────┬────────────────────────────┘
                                     ▼
        ┌─────────────────────────────────────────────────────────┐
        │  2. AUDITORÍA ESTADÍSTICA & DETECCIÓN DE OUTLIERS       │
        │     • Diagnóstico de asimetría extrema (Skewness > 10)  │
        │     • Dispersión por modalidades de contratación        │
        │     • Detección de atípicos con regla de Tukey (IQR)    │
        └────────────────────────────┬────────────────────────────┘
                                     ▼
        ┌─────────────────────────────────────────────────────────┐
        │  3. MACHINE LEARNING PREDICTIVO & STORYTELLING          │
        │     • Estabilización de varianza con np.log1p           │
        │     • Pipeline con ColumnTransformer (Scaler + OHE)     │
        │     • Modelo Ridge (R² = 0.9989, MAPE = 5.09%)          │
        │     • Dashboard ejecutivo y figuras a 300 DPI           │
        └─────────────────────────────────────────────────────────┘
```

---

## 📁 Estructura del Repositorio

```text
Dataxperience/
├── conexion-secop.py               # Extracción directa SODA v2 desde datos.gov.co
├── requirements.txt                # Dependencias del proyecto
├── .gitignore                      # Reglas de exclusión de Git
├── README.md                       # Documentación ejecutiva y técnica
├── data/
│   ├── raw/                        # Datos crudos extraídos de la API
│   │   └── secop_obras_cundinamarca.csv
│   └── processed/                  # Dataset saneado y procesado
│       └── secop_cundinamarca_obras_clean.csv
├── models/
│   └── modelo_regresion_contratos.joblib # Modelo predictivo serializado
├── reports/
│   └── figures/                    # Figuras analíticas y de storytelling (300 DPI)
│       ├── fig1_distribucion_montos.png
│       ├── fig2_boxplot_modalidades.png
│       ├── fig3_dispersion_oferentes.png
│       ├── fig4_prediccion_vs_real.png
│       ├── fig5_importancia_coeficientes.png
│       └── fig6_resumen_ejecutivo_storytelling.png
├── src/
│   ├── __init__.py
│   ├── cleaning.py                 # Saneamiento y validación transaccional
│   ├── features.py                 # Feature engineering (sobrecostos, plazos)
│   ├── statistics.py               # Medidas estadísticas descriptivas y Tukey IQR
│   ├── visualization.py            # Visualizaciones analíticas del EDA
│   ├── models.py                   # Pipeline supervisado scikit-learn y métricas
│   ├── storytelling_viz.py         # Dashboards y visualizaciones de storytelling
│   ├── pipeline_etapa_1.py         # Orquestador: Ingesta y Limpieza
│   ├── pipeline_etapa_2.py         # Orquestador: Análisis Estadístico y Outliers
│   └── pipeline_etapa_3.py         # Orquestador: Modelado Predictivo y Storytelling
└── tests/
    ├── test_etapa_1.py             # Pruebas unitarias de ingeniería de datos
    ├── test_etapa_2.py             # Pruebas unitarias de análisis estadístico
    └── test_etapa_3.py             # Pruebas unitarias de modelado y persistencia
```

---

## ⚙️ Instalación y Configuración

Se recomienda utilizar un entorno con **Python 3.10+**:

```bash
# Clonar el repositorio
git clone https://github.com/Levid009/Dataxperience.git
cd Dataxperience

# Instalar dependencias
pip install -r requirements.txt
```

---

## 🚀 Guía de Ejecución

### Ejecución Modular por Etapas

1. **Ingesta y Limpieza de Datos (Data Engineering):**
   ```bash
   python conexion-secop.py            # Descarga de datos abiertos vía API
   python src/pipeline_etapa_1.py       # Saneamiento y validaciones de negocio
   ```

2. **Auditoría Estadística y Análisis Exploratorio (EDA):**
   ```bash
   python src/pipeline_etapa_2.py       # Tendencia central, dispersión y outliers
   ```

3. **Modelado Predictivo y Storytelling (Machine Learning):**
   ```bash
   python src/pipeline_etapa_3.py       # Entrenamiento, métricas y dashboard final
   ```

### Validación Automatizada (Tests)
Ejecuta la suite integral de 21 pruebas unitarias:
```bash
python -m pytest tests/ -v
```

---

## 📊 Hallazgos y Resultados Consolidados

| Dimensión | Hallazgo Clave | Impacto en Auditoría |
| :--- | :--- | :--- |
| **Calidad del Dato** | Retención del **84.91%** de registros válidos (2,070 de 2,438). | Se eliminan duplicados y registros inconsistentes sin pérdida de información representativa. |
| **Distribución de Fondos** | Asimetría extrema (*skewness > 10*). Mediana: \$243.8M vs Media: \$1,385M COP. | La mediana representa el contrato típico; la media está distorsionada por megaproyectos. |
| **Detección de Atípicos** | Identificación sistemática de contratos anómalos mediante Tukey IQR ($1.5 \times \text{IQR}$). | Permite focalizar auditorías en contratos con desvíos presupuestales desproporcionados. |
| **Capacidad Predictiva** | Modelo regularizado con **$R^2 = 0.9989$** y **$\text{MAPE} = 5.09\%$**. | Capacidad de estimar con alta precisión el valor de mercado esperado para nuevos contratos. |
| **Efecto Competencia** | Coeficiente de concurrencia negativo (**-0.0192**). | Evidencia empírica de que una mayor cantidad de oferentes reduce el monto final adjudicado (ahorro público). |
| **Herramienta Operativa** | Modelo exportado en `models/modelo_regresion_contratos.joblib`. | Listo para integrarse como semáforo de alerta temprana en sistemas de contratación pública. |
