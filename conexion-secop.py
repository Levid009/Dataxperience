import os
import requests
import pandas as pd

# 1. Directorio de destino
os.makedirs("data/raw", exist_ok=True)
output_path = "data/raw/secop_obras_cundinamarca.csv"

# 2. Endpoint oficial SODA v2 de SECOP II Procesos de Contratación
URL = "https://www.datos.gov.co/resource/p6dx-8zbt.json"

# 3. Columnas exactas de la API de Socrata para SECOP II Procesos
selected_columns = [
    "id_del_portafolio",
    "referencia_del_proceso",
    "entidad",
    "departamento_entidad",
    "ciudad_entidad",
    "modalidad_de_contratacion",
    "tipo_de_contrato",
    "estado_del_procedimiento",
    "precio_base",
    "valor_total_adjudicacion",
    "proveedores_unicos_con",
    "duracion",
    "unidad_de_duracion",
    "fecha_de_publicacion_del"
]

# 4. Parámetros SoQL (filtro Cundinamarca, Obras, adjudicados con valor)
params = {
    "$select": ",".join(selected_columns),
    "$where": "departamento_entidad = 'Cundinamarca' AND tipo_de_contrato = 'Obra' AND valor_total_adjudicacion > 0",
    "$limit": 10000,
    "$order": "fecha_de_publicacion_del DESC"
}

headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) DataScienceAuditProject/1.0"
}

print("Consultando SECOP II - Procesos de Contratación (Obras en Cundinamarca)...")
try:
    response = requests.get(URL, params=params, headers=headers, timeout=60)
    response.raise_for_status()
    data = response.json()
    
    if not data:
        print("No se encontraron registros con los filtros indicados.")
    else:
        df = pd.DataFrame(data)
        df.to_csv(output_path, index=False, encoding="utf-8")
        print(f"Extracción exitosa: {len(df)} registros descargados.")
        print(f"Dataset guardado en: {output_path}")
        print("\nPrimeras 3 filas descargadas:")
        print(df[["entidad", "precio_base", "valor_total_adjudicacion", "modalidad_de_contratacion"]].head(3))

except requests.exceptions.RequestException as e:
    print(f"Error al consultar la API: {e}")
    if hasattr(e, "response") and e.response is not None:
        print("Detalle del error Socrata:", e.response.text)