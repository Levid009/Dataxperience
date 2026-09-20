# Contexto del Proyecto: Auditoría Predictiva de Contratación Pública en Obras de Infraestructura (Cundinamarca)

- Dominio: Compras públicas y auditoría de datos en Colombia (SECOP II).
- Fuente Oficial: Portal de Datos Abiertos de Colombia (datos.gov.co) - API Socrata / CSV.
- Sector: Obras Públicas / Infraestructura.
- Cobertura Geográfica: Cundinamarca (incluye municipios y gobernación).
- Objetivo Técnico:
  1. Extraer y procesar contratos públicos ejecutados o en ejecución.
  2. Construir métricas de desvío presupuestal:
     - Delta_Valor = Valor_Total_Con_Adiciones - Valor_Inicial
     - Ratio_Sobrecosto = (Valor_Total_Con_Adiciones - Valor_Inicial) / Valor_Inicial
     - Flag_Sobrecosto (1 si Ratio_Sobrecosto > 0, 0 en caso contrario).
  3. Realizar análisis estadístico riguroso (asimétricas, dispersión, Tukey IQR para anomalías).
  4. Entrenar un modelo predictivo para estimar la probabilidad de sobrecosto previo a la adjudicación.
- Stack: Python (pandas, numpy, sodapy/requests, matplotlib, seaborn, scikit-learn).