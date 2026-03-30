import pandas as pd
import urllib.request
import os

print("Descargando el dataset COMPAS desde ProPublica...")
url = "https://raw.githubusercontent.com/propublica/compas-analysis/master/compas-scores-two-years.csv"
archivo_crudo = "compas_crudo.csv"

# Solo lo descarga si no lo tienes ya en la carpeta
if not os.path.exists(archivo_crudo):
    urllib.request.urlretrieve(url, archivo_crudo)

print("Limpiando y procesando datos...")
df_raw = pd.read_csv(archivo_crudo)

# 1. Seleccionar solo las columnas que usará el pipeline
columnas = ['sex', 'age', 'race', 'priors_count', 'c_charge_degree', 'two_year_recid', 'decile_score']
df = df_raw[columnas].copy()

# 2. Filtrar solo las razas de interés para tu caso de estudio principal
df = df[df['race'].isin(['African-American', 'Caucasian'])]

# 3. Limpiar valores nulos para que AIF360 no falle
df = df.dropna()

# 4. Guardar dataset limpio y estandarizado
df.to_csv('compas_limpio.csv', index=False)
print(f"¡Listo! Archivo 'compas_limpio.csv' generado con {len(df)} registros limpios.")