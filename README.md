# Auditoría Predictiva de Contratación Pública en Obras de Infraestructura (Cundinamarca)

[![Python](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/)
[![Pandas](https://img.shields.io/badge/Pandas-3.0-orange.svg)](https://pandas.pydata.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.9-green.svg)](https://scikit-learn.org/)
[![Tests](https://img.shields.io/badge/Tests-PyTest%20Passed-brightgreen.svg)](https://pytest.org/)
[![Status](https://img.shields.io/badge/Status-Etapa%201%20Completada-success.svg)](#)

Sistema integral de auditoría de datos, detección de anomalías y modelado predictivo de sobrecostos y desvíos presupuestales en contratos de obra pública del departamento de Cundinamarca, utilizando datos oficiales de la plataforma **SECOP II** (*Portal de Datos Abiertos de Colombia - datos.gov.co*).

---

## 🏛️ Contexto y Justificación

La contratación pública de obras de infraestructura representa una de las mayores asignaciones del erario público en Colombia. No obstante, las asimetrías de información, modificaciones presupuestales y adiciones contractuales recurrentes plantean retos de auditoría, control fiscal y gobernanza.

Este proyecto aborda la problemática a través de un ciclo de vida analítico riguroso estructurado en tres etapas:

1. **Etapa 1: Ingesta, Auditoría de Calidad y Limpieza de Datos (Data Engineering)**  
   Extracción automatizada vía API Socrata (SODA v2), saneamiento de valores monetarios, normalización de fechas a ISO 8601, descarte de inconsistencias operativas (`precio_base <= 0`), deduplicación por ID de proceso y feature engineering inicial de sobrecosto y duración estandarizada.
2. **Etapa 2: Análisis Estadístico Descriptivo y Detección de Anomalías (Data Science)** *(En progreso)*  
   Análisis de asimetría (Media vs. Mediana), dispersión segmentada por modalidad, detección de outliers (criterio Tukey IQR) y fiscalización del límite legal del 50% de adición presupuestal (Ley 80 de 1993, art. 40).
3. **Etapa 3: Modelado Predictivo, Evaluación y Narrativa Ejecutiva (Machine Learning)** *(Próxima)*  
   Pipeline de clasificación supervisada (`scikit-learn`) con `ColumnTransformer`, balanceo de clases y Regresión Logística interpretable para auditoría pública.

---

## 📁 Estructura del Repositorio

```text
Dataxperience/
├── conexion-secop.py               # Extracción directa SODA v2 desde datos.gov.co (endpoint p6dx-8zbt)
├── requirements.txt                # Dependencias del proyecto
├── .gitignore                      # Reglas de exclusión de Git
├── README.md                       # Documentación ejecutiva y técnica
├── data/
│   ├── raw/                        # Datos crudos extraídos de la API
│   │   └── secop_obras_cundinamarca.csv
│   └── processed/                  # Dataset saneado y con variables ingenieriles
│       └── secop_cundinamarca_obras_clean.csv
├── src/
│   ├── __init__.py
│   ├── cleaning.py                 # Módulo de saneamiento, casteo y deduplicación
│   ├── features.py                 # Módulo de cálculo de sobrecostos y duración en días
│   └── pipeline_etapa_1.py         # Orquestador del flujo con CLI y reporte de auditoría
└── tests/
    └── test_etapa_1.py             # Suite de pruebas unitarias automatizadas con pytest
```

---

## ⚙️ Instalación y Requisitos

Se recomienda utilizar un entorno virtual con **Python 3.10+**:

```bash
# Clonar el repositorio
git clone https://github.com/Levid009/Dataxperience.git
cd Dataxperience

# Instalar dependencias
pip install -r requirements.txt
```

---

## 🚀 Ejecución - Etapa 1

### 1. Extracción de Datos Oficiales (SECOP II)
Descarga los contratos de obra vigentes o adjudicados en Cundinamarca directamente desde el portal oficial:
```bash
python conexion-secop.py
```

### 2. Ejecutar el Pipeline de Limpieza y Feature Engineering
Procesa el dataset crudo, aplica las reglas de auditoría y genera el dataset final depurado:
```bash
python src/pipeline_etapa_1.py
```

### 3. Ejecutar Pruebas Automatizadas
Verifica la integridad de las transformaciones y reglas de negocio:
```bash
python -m pytest tests/test_etapa_1.py -v
```

---

## 📊 Balance de Calidad y Resultados (Etapa 1)

| Métrica | Valor Obtenido |
| :--- | :--- |
| **Registros Iniciales (Raw)** | 2,438 |
| **Registros con `precio_base <= 0` descartados** | 5 |
| **Duplicados removidos (`id_del_portafolio`)** | 363 |
| **Registros Finales Depurados** | **2,070** |
| **Tasa de Retención de Datos Limpios** | **84.91%** |
| **Contratos con Sobrecosto / Adición** | 7 (0.3382%) |
| **Dataset Resultante** | `data/processed/secop_cundinamarca_obras_clean.csv` |

### Variables Ingenieriles Creadas
* **`sobrecosto_pesos`**: Diferencia monetaria ($Valor\_Total\_Adjudicacion - Precio\_Base$).
* **`ratio_sobrecosto`**: Magnitud porcentual del desvío ($Sobrecosto\_Pesos / Precio\_Base$).
* **`tiene_sobrecosto`**: Flag binario (1 si $Sobrecosto\_Pesos > 0$, 0 en caso contrario).
* **`duracion_dias_prevista`**: Estandarización de plazos temporales a días (convirtiendo meses a días $\times 30$ y días directos).


