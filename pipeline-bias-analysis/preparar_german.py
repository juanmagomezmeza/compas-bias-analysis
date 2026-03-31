import pandas as pd
from sklearn.datasets 
import fetch_openml
import numpy as np

print("Descargando el German Credit Dataset desde OpenML...")
# Descargar el dataset oficial
data = fetch_openml(data_id=31, as_frame=True, parser='auto')
df = data.frame

# 1. Limpiar el objetivo (Target): 1 = Crédito Aprobado (Good), 0 = Rechazado (Bad)
df['credito_aprobado'] = (df['class'] == 'good').astype(int)

# 2. Limpiar el atributo protegido: El género viene mezclado en 'personal_status'
df['sexo'] = df['personal_status'].apply(lambda x: 'Mujer' if 'female' in str(x).lower() else 'Hombre')

# 3. Simular el score del modelo del banco (1 a 10)
# Le agregamos una penalización matemática a las mujeres para simular un modelo histórico sesgado
np.random.seed(42)
score_base = (df['credit_amount'] / 1000) - (df['sexo'] == 'Mujer').astype(int) * 2 + np.random.normal(0, 2, len(df))
df['score_banco'] = pd.qcut(score_base.rank(method='first'), 10, labels=False) + 1

# 4. Guardar el CSV limpio
columnas_finales = ['age', 'sexo', 'credit_amount', 'duration', 'purpose', 'credito_aprobado', 'score_banco']
df_limpio = df[columnas_finales].dropna()
df_limpio.to_csv('german_credit_limpio.csv', index=False)

print(f"¡Listo! Archivo 'german_credit_limpio.csv' generado con {len(df_limpio)} registros.")