"""
Análisis de Sesgos Algorítmicos en COMPAS usando AIF360
Dataset: ProPublica COMPAS Analysis
Versión con AIF360 completo - Métricas avanzadas de fairness
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import warnings
import os
import urllib.request
warnings.filterwarnings('ignore')

# Importar AIF360
from aif360.datasets import CompasDataset
from aif360.metrics import BinaryLabelDatasetMetric, ClassificationMetric
from aif360.datasets import BinaryLabelDataset

# Configuración de estilo
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("Set2")
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 11

print("="*70)
print("ANÁLISIS DE SESGOS ALGORÍTMICOS - COMPAS con AIF360")
print("="*70)

# =============================================================================
# CARGAR DATASET DIRECTAMENTE CON PANDAS (como en tu script original)
# =============================================================================
print("\n[1/6] Cargando dataset COMPAS...")

data_url = "https://raw.githubusercontent.com/propublica/compas-analysis/master/compas-scores-two-years.csv"
data_file = "compas-scores-two-years.csv"

if not os.path.exists(data_file):
    print("Descargando dataset COMPAS desde GitHub...")
    urllib.request.urlretrieve(data_url, data_file)
    print("✓ Dataset descargado exitosamente")
else:
    print("✓ Dataset ya existe localmente")

# Cargar dataset directamente con pandas
df_raw = pd.read_csv(data_file)

# Preprocesar datos IGUAL que tu script original
df = df_raw[['sex', 'age', 'race', 'juv_fel_count', 'juv_misd_count', 
             'juv_other_count', 'priors_count', 'c_charge_degree', 
             'two_year_recid', 'decile_score', 'score_text']].copy()

# Limpiar datos
df = df[df['race'].isin(['African-American', 'Caucasian'])]

# Mapear valores para mejor interpretación
df['sex_original'] = df['sex'].copy()
df['race_original'] = df['race'].copy()
df['sex'] = df['sex'].map({'Male': 'Male', 'Female': 'Female'})
df['two_year_recid_label'] = df['two_year_recid'].map({1: 'Reincidió', 0: 'No Reincidió'})

# Crear predicciones basadas en decile_score (IGUAL que tu script)
df['predicted_high_risk'] = (df['decile_score'] >= 5).astype(int)

print(f"  Dimensiones del dataset: {df.shape}")
print(f"  Grupos raciales: {df['race'].unique()}")
print(f"  Rango de decile_score: {df['decile_score'].min()} - {df['decile_score'].max()}")
print(f"  Tasa de predicción alta: {df['predicted_high_risk'].mean()*100:.2f}%")

# =============================================================================
# CREAR DATASET AIF360 DESDE PANDAS DATAFRAME
# =============================================================================
print("\n[2/6] Creando dataset AIF360 y calculando métricas de fairness...")

# Preparar dataframe para AIF360 (SOLO columnas numéricas)
df_aif = pd.DataFrame()
df_aif['race_binary'] = df['race_original'].map({'Caucasian': 1, 'African-American': 0})
df_aif['sex_binary'] = df['sex_original'].map({'Male': 1, 'Female': 0})
df_aif['age'] = df['age']
df_aif['juv_fel_count'] = df['juv_fel_count']
df_aif['juv_misd_count'] = df['juv_misd_count']
df_aif['juv_other_count'] = df['juv_other_count']
df_aif['priors_count'] = df['priors_count']
df_aif['two_year_recid'] = df['two_year_recid']
df_aif['predicted_high_risk'] = df['predicted_high_risk']

# Verificar que todo es numérico
print(f"  Tipos de datos: {df_aif.dtypes.unique()}")

# Crear BinaryLabelDataset de AIF360
dataset_orig = BinaryLabelDataset(
    favorable_label=1,
    unfavorable_label=0,
    df=df_aif,
    label_names=['two_year_recid'],
    protected_attribute_names=['race_binary'],
    privileged_protected_attributes=[[1]],  # Caucasian = 1
    unprivileged_protected_attributes=[[0]]  # African-American = 0
)

# Definir grupos privilegiados y no privilegiados
privileged_groups = [{'race_binary': 1}]  # Caucasian
unprivileged_groups = [{'race_binary': 0}]  # African-American

# Calcular métricas de fairness con AIF360
metric_orig = BinaryLabelDatasetMetric(
    dataset_orig,
    unprivileged_groups=unprivileged_groups,
    privileged_groups=privileged_groups
)

print("\n" + "="*70)
print("MÉTRICAS DE FAIRNESS - DATASET ORIGINAL (AIF360)")
print("="*70)
print(f"Disparate Impact: {metric_orig.disparate_impact():.4f}")
print(f"  (Ideal = 1.0, >0.8 es aceptable)")
print(f"\nStatistical Parity Difference: {metric_orig.statistical_parity_difference():.4f}")
print(f"  (Ideal = 0.0, rango aceptable: -0.1 a 0.1)")
print(f"\nMean Difference: {metric_orig.mean_difference():.4f}")
print("="*70)

# Crear dataset con predicciones
dataset_pred = dataset_orig.copy(deepcopy=True)
dataset_pred.labels = df_aif['predicted_high_risk'].values.reshape(-1, 1)

# Métricas de clasificación con AIF360
classified_metric = ClassificationMetric(
    dataset_orig, dataset_pred,
    unprivileged_groups=unprivileged_groups,
    privileged_groups=privileged_groups
)

print("\n" + "="*70)
print("MÉTRICAS DE CLASIFICACIÓN (AIF360)")
print("="*70)
print(f"Equal Opportunity Difference: {classified_metric.equal_opportunity_difference():.4f}")
print(f"  (Diferencia en True Positive Rate entre grupos)")
print(f"\nAverage Odds Difference: {classified_metric.average_odds_difference():.4f}")
print(f"  (Promedio de diferencias en TPR y FPR)")
print(f"\nDisparate Impact (predicciones): {classified_metric.disparate_impact():.4f}")
print("="*70)

# =============================================================================
# FIGURA 6.1 - Distribución de riesgo por raza
# =============================================================================
print("\n[3/6] Generando Figura 6.1...")
plt.figure(figsize=(10, 6))
risk_race = pd.crosstab(df['race'], df['two_year_recid_label'], normalize='index') * 100

ax = risk_race.plot(kind='bar', stacked=False, color=['#2ecc71', '#e74c3c'], width=0.7)
plt.title('Figura 6.1 - Distribución de Riesgo de Reincidencia por Raza\n(Análisis con AIF360)', 
          fontsize=14, fontweight='bold', pad=20)
plt.xlabel('Grupo Racial', fontsize=12, fontweight='bold')
plt.ylabel('Porcentaje (%)', fontsize=12, fontweight='bold')
plt.xticks(rotation=0)
plt.legend(title='Reincidencia Real', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(axis='y', alpha=0.3)

for container in ax.containers:
    ax.bar_label(container, fmt='%.1f%%', padding=3)

plt.tight_layout()
plt.savefig('figura_6_1_riesgo_por_raza_aif360.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Figura 6.1 generada")

# =============================================================================
# FIGURA 6.2 - Distribución de riesgo por género
# =============================================================================
print("[4/6] Generando Figura 6.2...")
plt.figure(figsize=(10, 6))
risk_gender = pd.crosstab(df['sex'], df['two_year_recid_label'], normalize='index') * 100

ax = risk_gender.plot(kind='bar', stacked=False, color=['#2ecc71', '#e74c3c'], width=0.7)
plt.title('Figura 6.2 - Distribución de Riesgo de Reincidencia por Género\n(Análisis con AIF360)', 
          fontsize=14, fontweight='bold', pad=20)
plt.xlabel('Género', fontsize=12, fontweight='bold')
plt.ylabel('Porcentaje (%)', fontsize=12, fontweight='bold')
plt.xticks(rotation=0)
plt.legend(title='Reincidencia Real', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(axis='y', alpha=0.3)

for container in ax.containers:
    ax.bar_label(container, fmt='%.1f%%', padding=3)

plt.tight_layout()
plt.savefig('figura_6_2_riesgo_por_genero_aif360.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Figura 6.2 generada")

# =============================================================================
# FIGURA 6.3 - Comparación entre tasas de reincidencia real y predicha
# =============================================================================
print("[5/6] Generando Figura 6.3...")
plt.figure(figsize=(12, 6))

# Tasa real de reincidencia
real_recid = df.groupby('race')['two_year_recid'].apply(
    lambda x: (x == 1).sum() / len(x) * 100
)

# Predicción alto riesgo
pred_recid = df.groupby('race')['predicted_high_risk'].apply(
    lambda x: x.sum() / len(x) * 100
)

comparison_df = pd.DataFrame({
    'Reincidencia Real': real_recid,
    'Predicción Alto Riesgo': pred_recid
})

ax = comparison_df.plot(kind='bar', width=0.8, color=['#3498db', '#e67e22'])
plt.title('Figura 6.3 - Comparación entre Tasas de Reincidencia Real y Predicha\n(Análisis con AIF360)', 
          fontsize=14, fontweight='bold', pad=20)
plt.xlabel('Grupo Racial', fontsize=12, fontweight='bold')
plt.ylabel('Tasa (%)', fontsize=12, fontweight='bold')
plt.xticks(rotation=0)
plt.legend(title='Métrica', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(axis='y', alpha=0.3)

for container in ax.containers:
    ax.bar_label(container, fmt='%.1f%%', padding=3)

plt.tight_layout()
plt.savefig('figura_6_3_comparacion_real_predicha_aif360.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Figura 6.3 generada")

# =============================================================================
# FIGURA 6.4 - Matriz de confusión por grupo racial
# =============================================================================
print("[6/6] Generando Figura 6.4...")
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

races = df['race'].unique()
df['two_year_recid_binary'] = df['two_year_recid'].astype(int)

for idx, race in enumerate(races):
    df_race = df[df['race'] == race]
    
    cm = confusion_matrix(
        df_race['two_year_recid_binary'], 
        df_race['predicted_high_risk']
    )
    
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
    
    sns.heatmap(cm_normalized, annot=True, fmt='.1f', cmap='YlOrRd', 
                cbar_kws={'label': 'Porcentaje (%)'}, ax=axes[idx],
                xticklabels=['Pred: Bajo Riesgo', 'Pred: Alto Riesgo'],
                yticklabels=['Real: No Reincidió', 'Real: Reincidió'])
    
    axes[idx].set_title(f'{race}\n(n={len(df_race)})', fontweight='bold', fontsize=12)
    axes[idx].set_xlabel('Predicción', fontweight='bold')
    axes[idx].set_ylabel('Valor Real', fontweight='bold')

fig.suptitle('Figura 6.4 - Matriz de Confusión por Grupo Racial\n(Análisis con AIF360)', 
             fontsize=14, fontweight='bold', y=1.00)
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig('figura_6_4_matriz_confusion_aif360.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Figura 6.4 generada")

# =============================================================================
# FIGURA 6.5 - Distribución de puntuaciones de riesgo por raza
# =============================================================================
print("Generando Figura 6.5...")
plt.figure(figsize=(12, 7))

ax = sns.violinplot(data=df, x='race', y='decile_score', hue='two_year_recid_label',
                    split=True, inner='quartile', palette=['#2ecc71', '#e74c3c'])

plt.title('Figura 6.5 - Distribución de Puntuaciones de Riesgo por Raza\n(Análisis con AIF360)', 
          fontsize=14, fontweight='bold', pad=20)
plt.xlabel('Grupo Racial', fontsize=12, fontweight='bold')
plt.ylabel('Puntuación de Riesgo (Decile Score: 1-10)', fontsize=12, fontweight='bold')
plt.legend(title='Reincidencia Real', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(axis='y', alpha=0.3)

plt.axhline(y=5, color='black', linestyle='--', linewidth=1, alpha=0.5, 
            label='Umbral Alto Riesgo (≥5)')

plt.tight_layout()
plt.savefig('figura_6_5_distribucion_scores_aif360.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Figura 6.5 generada")

# =============================================================================
# MÉTRICAS DE CLASIFICACIÓN POR GRUPO RACIAL
# =============================================================================
print("\n" + "="*70)
print("MÉTRICAS DE CLASIFICACIÓN POR GRUPO RACIAL")
print("="*70)

for race in races:
    print(f"\n{race}:")
    df_race = df[df['race'] == race]
    
    fp = ((df_race['two_year_recid_binary'] == 0) & 
          (df_race['predicted_high_risk'] == 1)).sum()
    tn = ((df_race['two_year_recid_binary'] == 0) & 
          (df_race['predicted_high_risk'] == 0)).sum()
    fpr = fp / (fp + tn) * 100 if (fp + tn) > 0 else 0
    
    fn = ((df_race['two_year_recid_binary'] == 1) & 
          (df_race['predicted_high_risk'] == 0)).sum()
    tp = ((df_race['two_year_recid_binary'] == 1) & 
          (df_race['predicted_high_risk'] == 1)).sum()
    fnr = fn / (fn + tp) * 100 if (fn + tp) > 0 else 0
    
    ppr = df_race['predicted_high_risk'].mean() * 100
    
    print(f"  - Tasa de Falsos Positivos (FPR): {fpr:.2f}%")
    print(f"  - Tasa de Falsos Negativos (FNR): {fnr:.2f}%")
    print(f"  - Tasa de Predicción Positiva (PPR): {ppr:.2f}%")
    print(f"  - Score Promedio: {df_race['decile_score'].mean():.2f}")
    print(f"  - N (muestra): {len(df_race)}")

# =============================================================================
# MÉTRICAS DE EQUIDAD COMPARATIVAS
# =============================================================================
print("\n" + "="*70)
print("MÉTRICAS DE EQUIDAD COMPARATIVAS (AIF360)")
print("="*70)

df_aa = df[df['race'] == 'African-American']
df_cauc = df[df['race'] == 'Caucasian']

# FPR difference
fpr_aa = ((df_aa['two_year_recid_binary'] == 0) & (df_aa['predicted_high_risk'] == 1)).sum() / \
         ((df_aa['two_year_recid_binary'] == 0)).sum()
fpr_cauc = ((df_cauc['two_year_recid_binary'] == 0) & (df_cauc['predicted_high_risk'] == 1)).sum() / \
           ((df_cauc['two_year_recid_binary'] == 0)).sum()

print(f"\nFalse Positive Rate Difference:")
print(f"  African-American: {fpr_aa*100:.2f}%")
print(f"  Caucasian: {fpr_cauc*100:.2f}%")
print(f"  Diferencia: {(fpr_aa - fpr_cauc)*100:.2f} puntos porcentuales")
print(f"  AIF360 Equal Opportunity Diff: {classified_metric.equal_opportunity_difference():.4f}")

# FNR difference
fnr_aa = ((df_aa['two_year_recid_binary'] == 1) & (df_aa['predicted_high_risk'] == 0)).sum() / \
         ((df_aa['two_year_recid_binary'] == 1)).sum()
fnr_cauc = ((df_cauc['two_year_recid_binary'] == 1) & (df_cauc['predicted_high_risk'] == 0)).sum() / \
           ((df_cauc['two_year_recid_binary'] == 1)).sum()

print(f"\nFalse Negative Rate Difference:")
print(f"  African-American: {fnr_aa*100:.2f}%")
print(f"  Caucasian: {fnr_cauc*100:.2f}%")
print(f"  Diferencia: {(fnr_aa - fnr_cauc)*100:.2f} puntos porcentuales")

print("\n" + "="*70)
print("✓ ANÁLISIS COMPLETO CON AIF360")
print("="*70)
print("\nTodas las figuras han sido guardadas en:")
print(f"  {os.path.abspath('.')}")
print("\nArchivos generados:")
print("  • figura_6_1_riesgo_por_raza_aif360.png")
print("  • figura_6_2_riesgo_por_genero_aif360.png")
print("  • figura_6_3_comparacion_real_predicha_aif360.png")
print("  • figura_6_4_matriz_confusion_aif360.png")
print("  • figura_6_5_distribucion_scores_aif360.png")
print("\n" + "="*70)
print("MÉTRICAS AIF360 UTILIZADAS:")
print("="*70)
print("""
✓ Disparate Impact: Ratio de tasas de resultado positivo
✓ Statistical Parity Difference: Diferencia en tasas de predicción
✓ Equal Opportunity Difference: Diferencia en TPR entre grupos
✓ Average Odds Difference: Promedio de diferencias en TPR y FPR
""")
print("="*70)