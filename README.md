# Auditoría Predictiva de Contratación Pública en Obras de Infraestructura (Cundinamarca)

[![Python](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/)
[![Pandas](https://img.shields.io/badge/Pandas-2.2-orange.svg)](https://pandas.pydata.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.9-green.svg)](https://scikit-learn.org/)
[![Tests](https://img.shields.io/badge/Tests-21%20Passed-brightgreen.svg)](https://pytest.org/)
[![Status](https://img.shields.io/badge/Status-Etapas%201%2C%202%20y%203%20Completadas-success.svg)](#)

Sistema integral de auditoría de datos, detección de anomalías y modelado predictivo de adjudicaciones y desvíos presupuestales en contratos de obra pública del departamento de Cundinamarca, utilizando datos oficiales de la plataforma **SECOP II** (*Portal de Datos Abiertos de Colombia - datos.gov.co*).

---

## 🏛️ Contexto y Ciclo de Vida Analítico

La contratación pública de obras de infraestructura representa una de las mayores asignaciones del erario público en Colombia. Este proyecto implementa un ciclo de vida analítico riguroso estructurado en tres etapas complementarias:

1. **Etapa 1: Ingesta, Auditoría de Calidad y Limpieza de Datos (Data Engineering)**  
   Extracción automatizada vía API Socrata (SODA v2), saneamiento de valores monetarios, normalización de fechas a ISO 8601, descarte de inconsistencias operativas (`precio_base <= 0`), deduplicación por ID de proceso y feature engineering inicial de sobrecosto y duración estandarizada.
2. **Etapa 2: Análisis Estadístico Descriptivo y Detección de Anomalías (Data Science)**  
   Análisis de asimetría (Media vs. Mediana), dispersión segmentada por modalidad contractual, detección de valores atípicos (criterio Tukey IQR) y contraste de hipótesis econométricas sobre la concurrencia de oferentes.
3. **Etapa 3: Modelado Predictivo, Visualización y Storytelling (Machine Learning)**  
   Pipeline de regresión lineal regularizada (`Ridge`) con estabilización de varianza (`log1p`), preprocesamiento con `ColumnTransformer` (`StandardScaler` + `OneHotEncoder`), evaluación dual en pesos colombianos ($R^2 = 0.9989$, $\text{MAPE} = 5.09\%$), persistencia del modelo en `models/` y generación de figuras de storytelling a 300 DPI en `reports/figures/`.

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
│   └── modelo_regresion_contratos.joblib # Pipeline entrenado y serializado
├── reports/
│   └── figures/                    # Figuras analíticas en alta resolución (300 DPI)
│       ├── fig1_distribucion_montos.png
│       ├── fig2_boxplot_modalidades.png
│       ├── fig3_dispersion_oferentes.png
│       ├── fig4_prediccion_vs_real.png
│       ├── fig5_importancia_coeficientes.png
│       └── fig6_resumen_ejecutivo_storytelling.png
├── src/
│   ├── __init__.py
│   ├── cleaning.py                 # Saneamiento, casteo y deduplicación
│   ├── features.py                 # Cálculo de sobrecostos y duración temporal
│   ├── statistics.py               # Métricas de tendencia central, dispersión y Tukey IQR
│   ├── visualization.py            # Gráficos estadísticos del EDA
│   ├── models.py                   # Pipeline de modelado predictivo y evaluación
│   ├── storytelling_viz.py         # Visualizaciones ejecutivas y storytelling a 300 DPI
│   ├── pipeline_etapa_1.py         # Orquestador: Ingesta y Limpieza
│   ├── pipeline_etapa_2.py         # Orquestador: EDA y Estadísticas
│   └── pipeline_etapa_3.py         # Orquestador: Modelado Predictivo y Storytelling
└── tests/
    ├── test_etapa_1.py             # Pruebas unitarias de limpieza y features
    ├── test_etapa_2.py             # Pruebas unitarias de análisis estadístico
    └── test_etapa_3.py             # Pruebas unitarias de modelado y persistencia
```

---

## ⚙️ Instalación y Requisitos

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

### 1. Ingesta y Depuración de Datos (Etapa 1)
```bash
python conexion-secop.py            # Descarga desde SECOP II (SODA API)
python src/pipeline_etapa_1.py       # Saneamiento y generación de features
```

### 2. Análisis Estadístico y Detección de Anomalías (Etapa 2)
```bash
python src/pipeline_etapa_2.py       # Estadísticas descriptivas, outliers y EDA
```

### 3. Modelado Predictivo y Storytelling (Etapa 3)
```bash
python src/pipeline_etapa_3.py       # Entrenamiento Ridge, evaluación y visualizaciones
```

### 4. Ejecutar Suite de Pruebas Automatizadas
```bash
python -m pytest tests/ -v          # 21 pruebas unitarias integradas
```

---

## 📊 Síntesis de Resultados y Métricas Clave

### Auditoría y Calidad del Dato (Etapa 1)
* **2,070 contratos depurados** retenidos de 2,438 registros crudos (tasa de retención del **84.91%**).
* Identificación y tipificación de contratos con desvíos y plazos estandarizados en días.

### Hallazgos Estadísticos y Outliers (Etapa 2)
* Fuerte asimetría positiva en los montos adjudicados (*skewness > 10*), donde la media distorsiona significativamente el contrato típico respecto a la mediana.
* Detección de anomalías y megaproyectos atípicos mediante el criterio Tukey IQR ($1.5 \times \text{IQR}$).

### Desempeño del Modelo Predictivo (Etapa 3)
* **$R^2$ Score (Escala Real COP):** **0.9989** (explica el 99.89% de la varianza en adjudicaciones).
* **MAPE (Error Porcentual Absoluto Medio):** **5.09%**.
* **Efecto de la Competencia:** Coeficiente negativo de concurrencia de oferentes ($-0.0192$), evidenciando empíricamente que mayor participación de proponentes genera ahorros presupuestales para el Estado.
* **Modelo Operativo:** Serializado en `models/modelo_regresion_contratos.joblib` como herramienta de semáforo preventivo.
