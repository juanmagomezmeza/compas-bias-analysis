import pandas as pd
import numpy as np
from sklearn.datasets import fetch_openml

print("Descargando Communities and Crime Dataset (ID 1218)...")
# Descargar dataset
data = fetch_openml(data_id=1218, as_frame=True, parser='auto')
df = data.frame

# 1. Limpieza de Nulos (Este dataset tiene muchas columnas vacías)
# Seleccionamos columnas clave para evitar perder demasiados datos
cols_interes = [
    'racepctblack', 'racePctWhite', 'agePct12t29', 'pctWSocSec', 
    'PctUnemployed', 'HousVacant', 'ViolentCrimesPerPop'
]
df = df[cols_interes].dropna()

# 2. Binarizar Atributo Protegido (Raza)
# El dataset original tiene porcentajes. Creamos una columna binaria:
# 1 = Comunidad con alta población afroamericana (No privilegiado)
# 0 = Comunidad con baja población afroamericana (Privilegiado)
umbral_raza = df['racepctblack'].median()
df['es_afroamericano'] = (df['racepctblack'] > umbral_raza).astype(int)

# 3. Binarizar Objetivo (Crimen)
# 1 = Tasa de crimen ALTA (Evento punitivo/negativo)
# 0 = Tasa de crimen BAJA (Evento favorable)
# Nota: Para tu pipeline, el 0 (Bajo Crimen) será la "clase favorable"
umbral_crimen = df['ViolentCrimesPerPop'].median()
df['alto_crimen'] = (df['ViolentCrimesPerPop'] > umbral_crimen).astype(int)

# 4. Simular Score de la "Caja Negra" (1 a 10)
# Introducimos un sesgo artificial: el algoritmo castiga a las comunidades afroamericanas
# sumando puntos de riesgo solo por la raza.
np.random.seed(42)
score_base = (df['PctUnemployed'] * 10) + (df['es_afroamericano'] * 3) + np.random.normal(0, 1, len(df))

# Normalizar a escala 1-10
df['score_seguridad'] = pd.qcut(score_base.rank(method='first'), 10, labels=False) + 1

# 5. Guardar CSV final
# Mapeamos los nombres para que sean legibles en tus gráficos
df['raza_nombre'] = df['es_afroamericano'].map({1: 'Afroamericana', 0: 'Otras'})

df_final = df[['raza_nombre', 'agePct12t29', 'PctUnemployed', 'alto_crimen', 'score_seguridad']]
df_final.to_csv('communities_crime_limpio.csv', index=False)

print(f"¡Listo! Archivo 'communities_crime_limpio.csv' generado con {len(df_final)} registros.")
print("Sesgo: Se penalizó a las comunidades afroamericanas con +3 puntos de riesgo base.")