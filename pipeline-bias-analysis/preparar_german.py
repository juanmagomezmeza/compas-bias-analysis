import pandas as pd
import numpy as np
from sklearn.datasets import fetch_openml  

print("Descargando el German Credit Dataset desde OpenML...")
# Descargar el dataset oficial (ID 31 es el dataset de crédito alemán)
data = fetch_openml(data_id=31, as_frame=True, parser='auto')
df = data.frame

# 1. Limpiar el objetivo (Target): 1 = Crédito Aprobado (Good), 0 = Rechazado (Bad)
# Nota: En OpenML, el target suele venir en la columna 'class'
df['credito_aprobado'] = (df['class'] == 'good').astype(int)

# 2. Limpiar el atributo protegido:
# El género en este dataset histórico está codificado en 'personal_status'
df['sexo'] = df['personal_status'].apply(lambda x: 'Mujer' if 'female' in str(x).lower() else 'Hombre')

# 3. Simular el score del modelo del banco (1 a 10)
# Agregamos la penalización matemática para simular el sesgo a auditar
np.random.seed(42)

# Fórmula del score simulado:
# $$score = \frac{amount}{1000} - (2 \times sex_{Mujer}) + \epsilon$$
score_base = (df['duration'] / 12) + (df['credit_amount'] / 1000) - (df['sexo'] == 'Mujer').astype(int) * 2 + np.random.normal(0, 1, len(df))

# Normalizamos a escala 1-10 para tu pipeline
df['score_banco'] = pd.qcut(score_base.rank(method='first'), 10, labels=False) + 1

# 4. Guardar el CSV limpio para el Pipeline
columnas_finales = ['age', 'sexo', 'credit_amount', 'duration', 'purpose', 'credito_aprobado', 'score_banco']
df_limpio = df[columnas_finales].dropna()

# Guardar con el nombre que espera la configuración
df_limpio.to_csv('german_credit_limpio.csv', index=False)

print(f"¡Listo! Archivo 'german_credit_limpio.csv' generado con {len(df_limpio)} registros.")
print(f"Sesgo introducido: Las mujeres tienen un castigo promedio de 2 puntos en el score simulado.")