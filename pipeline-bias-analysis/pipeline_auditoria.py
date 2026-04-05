"""
Framework de Auditoría de Sesgos y Gobernanza Algorítmica
Pipeline Agnóstico (Parametrizado vía CLI y config.json)
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
import sys
import json
import argparse

# Importar AIF360
from aif360.datasets import StandardDataset
from aif360.metrics import BinaryLabelDatasetMetric, ClassificationMetric
from aif360.algorithms.postprocessing import EqOddsPostprocessing, CalibratedEqOddsPostprocessing, RejectOptionClassification
from fpdf import FPDF

warnings.filterwarnings('ignore')

# Configuración de estilo
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("Set2")
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 11

# =============================================================================
# FUNCIÓN DE VALIDACIÓN DE ESQUEMA (SCHEMA) CORREGIDA
# =============================================================================
def validar_esquema_json(config):
    """
    Valida estrictamente la presencia de las claves requeridas y sus tipos de datos (Schema).
    No evalúa la lógica de negocio ni si las columnas existen, solo la estructura.
    """
    errores = []
    
    esquema_raiz = {
        "dataset_name": str,
        "data_path": str,
        "label_name": str,
        "favorable_classes": list,
        "protected_attribute_names": list,
        "privileged_classes": list,
        "features_to_keep": list,
        "plot_mapping": dict
    }
    
    esquema_mapping = {
        "target_label_col": str,
        "target_label_name": str,
        "protected_col": str,
        "priv_val_name": str,
        "unpriv_val_name": str,
        "risk_score_col": str,
        "risk_threshold": int
    }

    # 1. Validar raíz
    for clave, tipo_esperado in esquema_raiz.items():
        if clave not in config:
            errores.append(f"Estructura JSON: Falta la clave principal '{clave}'.")
        else:
            valor = config[clave]
            # Python considera a los booleanos como enteros, evitamos ese falso positivo
            if isinstance(valor, bool) and tipo_esperado != bool:
                errores.append(f"Estructura JSON: La clave '{clave}' no puede ser booleana.")
            elif not isinstance(valor, tipo_esperado):
                errores.append(f"Estructura JSON: La clave '{clave}' debe ser de tipo '{tipo_esperado.__name__}'.")

    # Si no existe plot_mapping o no es diccionario, cortamos acá para evitar errores en cascada
    if "plot_mapping" not in config or not isinstance(config["plot_mapping"], dict):
        return errores

    mapping = config["plot_mapping"]

    # 2. Validar interior de plot_mapping
    for clave, tipo_esperado in esquema_mapping.items():
        if clave not in mapping:
            errores.append(f"Estructura JSON: Falta la clave '{clave}' dentro de 'plot_mapping'.")
        else:
            valor = mapping[clave]
            if isinstance(valor, bool):
                errores.append(f"Estructura JSON: La clave 'plot_mapping.{clave}' no puede ser booleana.")
            elif not isinstance(valor, tipo_esperado):
                errores.append(f"Estructura JSON: La clave 'plot_mapping.{clave}' debe ser de tipo '{tipo_esperado.__name__}'.")

    # Validar secondary_col que es opcional
    sec_col = config.get('secondary_col') or mapping.get('secondary_col')
    if sec_col is not None and not isinstance(sec_col, str):
        errores.append("Estructura JSON: La clave 'secondary_col', si se provee, debe ser un string.")

    return errores

# =============================================================================
# CLASE DE GOBERNANZA Y AUDITORÍA
# =============================================================================
class AIGovernanceAuditor:
    def __init__(self, privileged_groups, unprivileged_groups):
        self.privileged_groups = privileged_groups
        self.unprivileged_groups = unprivileged_groups
        self.thresholds = {
            'disparate_impact': (0.8, 1.25),
            'equal_opportunity_diff': (-0.1, 0.1)
        }

    def evaluate_compliance(self, metric, is_classification=False):
        di = metric.disparate_impact()
        di_compliant = self.thresholds['disparate_impact'][0] <= di <= self.thresholds['disparate_impact'][1]
        status = "ACEPTADO" if di_compliant else "RECHAZADO"
        
        if is_classification:
            eod = metric.equal_opportunity_difference()
            eod_compliant = self.thresholds['equal_opportunity_diff'][0] <= eod <= self.thresholds['equal_opportunity_diff'][1]
            if not eod_compliant: 
                status = "RECHAZADO"
            return status, di, eod
            
        return status, di

    def print_certification_report(self, m_orig, m_eq, m_cal, m_roc, dataset_name, mitigation_applied):
        print("\n" + "="*70)
        print("REPORTE DE CERTIFICACIÓN DE GOBERNANZA ALGORÍTMICA")
        print("="*70)
        
        st_orig, di_orig, eod_orig = self.evaluate_compliance(m_orig, True)
        print(f"[CAJA NEGRA] SISTEMA ORIGINAL ({dataset_name}): {st_orig}")
        print(f"  > Disparate Impact: {di_orig:.4f} | Equal Opp. Diff: {eod_orig:.4f}")
        
        if mitigation_applied:
            st_eq, di_eq, eod_eq = self.evaluate_compliance(m_eq, True)
            print(f"\n[INTERVENCIÓN 1] EQUALIZED ODDS: {st_eq}")
            print(f"  > Disparate Impact: {di_eq:.4f} | Equal Opp. Diff: {eod_eq:.4f}")
            
            st_cal, di_cal, eod_cal = self.evaluate_compliance(m_cal, True)
            print(f"\n[INTERVENCIÓN 2] CALIBRATED EQUALIZED ODDS: {st_cal}")
            print(f"  > Disparate Impact: {di_cal:.4f} | Equal Opp. Diff: {eod_cal:.4f}")

            st_roc, di_roc, eod_roc = self.evaluate_compliance(m_roc, True)
            print(f"\n[INTERVENCIÓN 3] REJECT OPTION CLASSIFICATION (ROC): {st_roc}")
            print(f"  > Disparate Impact: {di_roc:.4f} | Equal Opp. Diff: {eod_roc:.4f}")
            
            estrategias_cumplen = []
            if st_eq == "ACEPTADO": estrategias_cumplen.append("Eq. Odds")
            if st_cal == "ACEPTADO": estrategias_cumplen.append("Cal. EqOdds")
            if st_roc == "ACEPTADO": estrategias_cumplen.append("ROC")
            
            if estrategias_cumplen:
                print(f"\n[INFO TÉCNICA] Las intervenciones {', '.join(estrategias_cumplen)} logran corregir matemáticamente el sesgo.")
            
            status_final = "RECHAZADO"
        else:
            print("\n[INFO] No se aplicó mitigación algorítmica por encontrarse dentro de los umbrales éticos.")
            status_final = st_orig
        
        print("\nDICTAMEN TÉCNICO FINAL:")
        if status_final == "ACEPTADO":
            print("\033[92m✓ APROBADO: El sistema base es apto para despliegue sin intervenciones.\033[0m")
        else:
            print("\033[91m⚠ RECHAZADO: El modelo original ('Caja Negra') presenta sesgos inaceptables. Su certificación es denegada, independientemente de la viabilidad del post-procesamiento.\033[0m")
        print("="*70)

# =============================================================================
# [PASO 4] CARGA DE CONFIGURACIÓN Y DATOS
# =============================================================================
print("\n[Paso 4] Cargando configuración y dataset...")

parser = argparse.ArgumentParser(description="Pipeline de Auditoría de Gobernanza Algorítmica")
parser.add_argument('--config', type=str, default='config.json', help='Ruta al archivo de configuración JSON')
args = parser.parse_args()
config_path = args.config

if not os.path.exists(config_path):
    print("\n❌ ERROR CRÍTICO: No se encontró el archivo de configuración.")
    sys.exit(1)

# FASE 0: Validar sintaxis estricta del JSON
try:
    with open(config_path, 'r') as f:
        config = json.load(f)
except json.JSONDecodeError as e:
    print("\n❌ ERROR CRÍTICO DE SINTAXIS JSON:")
    print(f"   El archivo '{config_path}' está mal escrito (ej. falta un valor, sobra una coma).")
    print(f"   Detalle: {e}")
    sys.exit(1)

print("\n[*] FASE 1: Validando Estructura (Schema) del JSON...")
errores_esquema = validar_esquema_json(config)

if errores_esquema:
    print("\n❌ EL JSON NO RESPETA LA ESTRUCTURA ESPERADA:")
    for err in errores_esquema:
        print(f"  -> {err}")
    print("\nDeteniendo ejecución. Corrige la estructura del JSON e intenta nuevamente.")
    sys.exit(1)

print("✓ Estructura de JSON válida.")

print("\n[*] FASE 2: Validando Reglas de Negocio y Valores Nulos...")
errores_valores = []
mapping = config['plot_mapping']

# 1. Validar que los campos de texto obligatorios no vengan vacíos ("")
campos_texto_obligatorios = [
    ("dataset_name", config['dataset_name']),
    ("data_path", config['data_path']),
    ("label_name", config['label_name']),
    ("plot_mapping.target_label_name", mapping['target_label_name'])
]

for nombre, valor in campos_texto_obligatorios:
    if not str(valor).strip():
        errores_valores.append(f"El valor de '{nombre}' no puede estar vacío.")

# 2. Validar que las listas obligatorias no estén vacías ([])
listas_obligatorias = [
    ("favorable_classes", config['favorable_classes']),
    ("protected_attribute_names", config['protected_attribute_names']),
    ("privileged_classes", config['privileged_classes']),
    ("features_to_keep", config['features_to_keep'])
]

for nombre, lista in listas_obligatorias:
    if len(lista) == 0:
        errores_valores.append(f"El array '{nombre}' debe contener al menos un elemento.")

# 3. Validación estricta del Umbral de Riesgo (ya garantizamos en la FASE 1 que es un int puro)
risk_thresh_val = mapping['risk_threshold']
if not (0 <= risk_thresh_val <= 10):
    errores_valores.append("el risk_threshold debe ser un entero del 0 al 10.")

if errores_valores:
    print("\n❌ SE DETECTARON VALORES INVÁLIDOS EN EL JSON:")
    for err in errores_valores:
        print(f"  -> {err}")
    print("\nDeteniendo ejecución. Corrige los valores e intenta nuevamente.")
    sys.exit(1)

print("✓ Valores de negocio válidos.")

# =============================================================================
# [PASO 4b] CARGA DEL DATASET Y VALIDACIÓN DE DATOS VS JSON
# =============================================================================
print("\n[*] FASE 3: Validando integridad con el dataset CSV...")
data_path = config['data_path']
if not os.path.exists(data_path):
    print(f"\n❌ ERROR CRÍTICO: No se encontró el dataset CSV en la ruta: {data_path}")
    sys.exit(1)

df_raw = pd.read_csv(data_path)
errores_datos = []

# Validar que las columnas existan
for col in config['features_to_keep']:
    if col not in df_raw.columns:
        errores_datos.append(f"La columna '{col}' (en features_to_keep) no existe en el CSV.")

p_col = mapping['protected_col']
t_col = mapping['target_label_col']
risk_col = mapping['risk_score_col']
sec_col = config.get('secondary_col') or mapping.get('secondary_col')

if p_col not in df_raw.columns: errores_datos.append(f"La columna protegida '{p_col}' no existe en el CSV.")
if t_col not in df_raw.columns: errores_datos.append(f"La columna objetivo '{t_col}' no existe en el CSV.")
if risk_col not in df_raw.columns: errores_datos.append(f"La columna de riesgo '{risk_col}' no existe en el CSV.")
if sec_col and sec_col not in df_raw.columns: errores_datos.append(f"La columna secundaria '{sec_col}' no existe en el CSV.")

# Validar categorías del atributo protegido
if p_col in df_raw.columns:
    valores_reales = df_raw[p_col].dropna().unique()
    priv_val = mapping['priv_val_name']
    unpriv_val = mapping['unpriv_val_name']
    
    if priv_val not in valores_reales:
        errores_datos.append(f"El grupo '{priv_val}' no existe en la columna '{p_col}'. Valores posibles: {list(valores_reales)}")
    if unpriv_val not in valores_reales:
        errores_datos.append(f"El grupo '{unpriv_val}' no existe en la columna '{p_col}'. Valores posibles: {list(valores_reales)}")

# Validar AIF360
aif_label = config['label_name']
if aif_label not in df_raw.columns:
    errores_datos.append(f"La columna de AIF360 '{aif_label}' no existe en el CSV.")
else:
    val_fav = config['favorable_classes'][0]
    if val_fav not in df_raw[aif_label].dropna().unique():
        errores_datos.append(f"El valor favorable '{val_fav}' no existe en la columna '{aif_label}'.")
    
aif_prot_attrs = config['protected_attribute_names']
if not aif_prot_attrs or aif_prot_attrs[0] not in df_raw.columns:
    errores_datos.append(f"La columna protegida de AIF360 '{aif_prot_attrs}' no existe en el CSV.")
else:
    priv_classes = config['privileged_classes']
    if priv_classes and isinstance(priv_classes[0], list) and len(priv_classes[0]) > 0:
        val_aif360 = priv_classes[0][0]
        valores_reales_aif = df_raw[aif_prot_attrs[0]].dropna().unique()
        if val_aif360 not in valores_reales_aif:
            errores_datos.append(f"El valor privilegiado '{val_aif360}' no existe en la columna '{aif_prot_attrs[0]}'.")
    else:
        errores_datos.append("El parámetro 'privileged_classes' está mal formateado. Formato esperado: [['Valor']].")

if errores_datos:
    print("\n❌ SE DETECTARON INCONSISTENCIAS CRÍTICAS ENTRE EL JSON Y EL DATASET:")
    for err in errores_datos:
        print(f"  -> {err}")
    print("\nDeteniendo ejecución. Corrige la configuración o los datos e intenta nuevamente.")
    sys.exit(1)
    
print("✓ Datos validados exitosamente.\n")
# =============================================================================

df = df_raw[config['features_to_keep']].copy()
df = df.dropna()

t_name = mapping['target_label_name']
risk_thresh = mapping['risk_threshold']

pos_label = t_name
neg_label = f"No {t_name}"

df['target_label_text'] = df[t_col].map({1: pos_label, 0: neg_label})
df['predicted_high_risk'] = (df[risk_col] >= risk_thresh).astype(int)
df['target_binary'] = df[t_col].astype(int)
df = df[df[p_col].isin([priv_val, unpriv_val])]

print("\n[*] Configurando Auditoría AIF360 dinámicamente...")
cat_cols = df[config['features_to_keep']].select_dtypes(include=['object', 'string', 'category']).columns.tolist()
for col in config['protected_attribute_names'] + [config['label_name']]:
    if col in cat_cols: cat_cols.remove(col)

dataset_orig = StandardDataset(
    df=df[config['features_to_keep']], label_name=config['label_name'],
    favorable_classes=config['favorable_classes'],
    protected_attribute_names=config['protected_attribute_names'],
    privileged_classes=[[priv_val]], 
    categorical_features=cat_cols
)

protected_attr_name = config['protected_attribute_names'][0]
privileged_groups = [{protected_attr_name: 1.0}]
unprivileged_groups = [{protected_attr_name: 0.0}]

dataset_pred = dataset_orig.copy(deepcopy=True)
dataset_pred.labels = df['predicted_high_risk'].values.reshape(-1, 1)

scores_norm = ((df[risk_col] - df[risk_col].min()) / (df[risk_col].max() - df[risk_col].min())).values.reshape(-1, 1)
dataset_orig.scores = scores_norm
dataset_pred.scores = scores_norm

# =============================================================================
# [PASO 5] CÁLCULO DE MÉTRICAS (AUDITORÍA INICIAL)
# =============================================================================
print("\n[Paso 5] Ejecutando Auditoría Inicial...")
gov_auditor = AIGovernanceAuditor(privileged_groups, unprivileged_groups)
classified_metric_orig = ClassificationMetric(dataset_orig, dataset_pred, unprivileged_groups=unprivileged_groups, privileged_groups=privileged_groups)

# =============================================================================
# [PASO 6] GENERACIÓN DE VISUALIZACIONES DESCRIPTIVAS
# =============================================================================
print("\n[Paso 6] Generando Figuras Descriptivas base y de Diagnóstico...")

# FIG 6.1
plt.figure(figsize=(10, 6))
risk_prot = pd.crosstab(df[p_col], df['target_label_text'], normalize='index') * 100
ax = risk_prot.plot(kind='bar', stacked=False, color=['#2ecc71', '#e74c3c'], width=0.7)
plt.title(f'Distribución de {t_name} Real por {p_col.capitalize()}', fontsize=14, fontweight='bold', pad=20)
plt.xlabel(f'Grupo ({p_col})', fontsize=12, fontweight='bold')
plt.ylabel('Porcentaje (%)', fontsize=12, fontweight='bold')
plt.xticks(rotation=0)
plt.legend(title=f'{t_name} Real', bbox_to_anchor=(1.05, 1), loc='upper left')
for container in ax.containers: ax.bar_label(container, fmt='%.1f%%', padding=3)
plt.tight_layout()
plt.savefig('figura_6_1_real_por_atributo.png', dpi=300, bbox_inches='tight')
plt.close()

# FIG 6.3
plt.figure(figsize=(12, 6))
real_target = df.groupby(p_col)['target_binary'].apply(lambda x: (x == 1).sum() / len(x) * 100)
pred_target = df.groupby(p_col)['predicted_high_risk'].apply(lambda x: x.sum() / len(x) * 100)
comparison_df = pd.DataFrame({f'{t_name} Real': real_target, 'Predicción de Riesgo': pred_target})
ax = comparison_df.plot(kind='bar', width=0.8, color=['#3498db', '#e67e22'])
plt.title(f'Comparación entre Tasas Reales y Predichas por {p_col.capitalize()}', fontsize=14, fontweight='bold', pad=20)
plt.xlabel(f'Grupo ({p_col})', fontsize=12, fontweight='bold')
plt.ylabel('Tasa (%)', fontsize=12, fontweight='bold')
plt.xticks(rotation=0)
plt.legend(title='Métrica', bbox_to_anchor=(1.05, 1), loc='upper left')
for container in ax.containers: ax.bar_label(container, fmt='%.1f%%', padding=3)
plt.tight_layout()
plt.savefig('figura_6_3_comparacion_real_predicha.png', dpi=300, bbox_inches='tight')
plt.close()

# FIG 6.4
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
groups = df[p_col].unique()
for idx, group in enumerate(groups[:2]): 
    df_group = df[df[p_col] == group]
    cm = confusion_matrix(df_group['target_binary'], df_group['predicted_high_risk'])
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
    sns.heatmap(cm_normalized, annot=True, fmt='.1f', cmap='YlOrRd', 
                cbar_kws={'label': 'Porcentaje (%)'}, ax=axes[idx],
                xticklabels=['Pred: Bajo', 'Pred: Alto'],
                yticklabels=[f'Real: {neg_label}', f'Real: {pos_label}'])
    axes[idx].set_title(f'{group}\n(n={len(df_group)})', fontweight='bold', fontsize=12)
    axes[idx].set_xlabel('Predicción', fontweight='bold')
    axes[idx].set_ylabel('Valor Real', fontweight='bold')
fig.suptitle(f'Matriz de Confusión por {p_col.capitalize()}', fontsize=14, fontweight='bold', y=1.00)
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig('figura_6_4_matriz_confusion.png', dpi=300, bbox_inches='tight')
plt.close()

# FIG 6.5 - Distribución de Scores (Violin Plot)
plt.figure(figsize=(12, 6))
paleta_violines = {
    pos_label: '#e74c3c', # Rojo: Riesgo
    neg_label: '#2ecc71'  # Verde: Favorable
}

sns.violinplot(
    data=df,
    x=p_col,
    y=risk_col,
    hue='target_label_text',
    split=True,
    inner="quart",
    palette=paleta_violines
)
plt.title(f'Distribución de Scores de {t_name} por {p_col.capitalize()}', fontsize=14, fontweight='bold', pad=20)
plt.xlabel(f'Grupo ({p_col})', fontsize=12, fontweight='bold')
plt.ylabel('Puntaje de Riesgo (Score)', fontsize=12, fontweight='bold')
plt.legend(title=f'{t_name} Real', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig('figura_6_5_distribucion_scores.png', dpi=300, bbox_inches='tight')
plt.close()

# FIG 6.6
df_unpriv = df[df[p_col] == unpriv_val]
df_priv = df[df[p_col] == priv_val]
fpr_unpriv = ((df_unpriv['target_binary'] == 0) & (df_unpriv['predicted_high_risk'] == 1)).sum() / ((df_unpriv['target_binary'] == 0)).sum() * 100
fpr_priv = ((df_priv['target_binary'] == 0) & (df_priv['predicted_high_risk'] == 1)).sum() / ((df_priv['target_binary'] == 0)).sum() * 100
fnr_unpriv = ((df_unpriv['target_binary'] == 1) & (df_unpriv['predicted_high_risk'] == 0)).sum() / ((df_unpriv['target_binary'] == 1)).sum() * 100
fnr_priv = ((df_priv['target_binary'] == 1) & (df_priv['predicted_high_risk'] == 0)).sum() / ((df_priv['target_binary'] == 1)).sum() * 100

error_data = pd.DataFrame({'Falsos Positivos (Sesgo en contra)': [fpr_unpriv, fpr_priv], 'Falsos Negativos (Sesgo a favor)': [fnr_unpriv, fnr_priv]}, index=[unpriv_val, priv_val])
plt.figure(figsize=(10, 6))
ax = error_data.plot(kind='bar', width=0.7, color=['#e74c3c', '#3498db'])
plt.title(f'Tasas de Error del Modelo por {p_col.capitalize()}', fontsize=14, fontweight='bold', pad=20)
plt.xlabel(f'Grupo ({p_col})', fontsize=12, fontweight='bold')
plt.ylabel('Tasa de Error (%)', fontsize=12, fontweight='bold')
plt.xticks(rotation=0)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
for container in ax.containers: ax.bar_label(container, fmt='%.1f%%', padding=3)
plt.tight_layout()
plt.savefig('figura_6_6_tasas_error.png', dpi=300, bbox_inches='tight')
plt.close()

# FIG 6.7a - Diagnóstico: DI
plt.figure(figsize=(10, 6))
di_value = classified_metric_orig.disparate_impact()

if np.isnan(di_value) or np.isinf(di_value):
    print("⚠️ ALERTA: División por cero en Disparate Impact. Forzando a 0.0 para graficar.")
    di_value = 0.0

bar_color = '#e67e22' if di_value > 1.25 or di_value < 0.8 else '#2ecc71'
plt.bar(['Disparate Impact\n(Predicciones)'], [di_value], color=bar_color, width=0.35)
plt.axhline(y=1.0, color='black', linestyle='-', linewidth=2, label='Equidad Ideal (1.0)')
plt.axhline(y=0.8, color='#e74c3c', linestyle='--', linewidth=1.5, label='Umbral Mínimo (0.8)')
plt.axhline(y=1.25, color='#e74c3c', linestyle='--', linewidth=1.5, label='Umbral Máximo (1.25)')
plt.axhspan(0.8, 1.25, color='#2ecc71', alpha=0.1)
plt.title('Diagnóstico: Medición de Disparate Impact', fontsize=14, fontweight='bold', pad=20)
plt.ylabel('Ratio', fontsize=12, fontweight='bold')
plt.ylim(0, max(di_value + 0.6, 2.0))
plt.legend(loc='upper right', framealpha=0.95)
plt.text(0, di_value + 0.03, f'{di_value:.4f}', ha='center', va='bottom', fontweight='bold', fontsize=12, color=bar_color)
plt.tight_layout()
plt.savefig('figura_6_7_disparate_impact.png', dpi=300, bbox_inches='tight')
plt.close()

# FIG 6.7b - Diagnóstico: EOD
plt.figure(figsize=(10, 6))
eod_value = classified_metric_orig.equal_opportunity_difference()

if np.isnan(eod_value) or np.isinf(eod_value):
    eod_value = 0.0

bar_color_eod = '#e67e22' if abs(eod_value) > 0.1 else '#2ecc71'
plt.bar(['Equal Opportunity Diff\n(Predicciones)'], [eod_value], color=bar_color_eod, width=0.35)
plt.axhline(y=0.0, color='black', linestyle='-', linewidth=2, label='Equidad Ideal (0.0)')
plt.axhline(y=-0.1, color='#e74c3c', linestyle='--', linewidth=1.5, label='Umbral Mínimo (-0.1)')
plt.axhline(y=0.1, color='#e74c3c', linestyle='--', linewidth=1.5, label='Umbral Máximo (0.1)')
plt.axhspan(-0.1, 0.1, color='#2ecc71', alpha=0.1)
plt.title('Diagnóstico: Medición de Equal Opportunity Difference', fontsize=14, fontweight='bold', pad=20)
plt.ylabel('Diferencia', fontsize=12, fontweight='bold')
y_max_eod = max(abs(eod_value) + 0.1, 0.2)
plt.ylim(-y_max_eod, y_max_eod)
plt.legend(loc='upper right', framealpha=0.95)
plt.text(0, eod_value + (0.02 if eod_value > 0 else -0.05), f'{eod_value:.4f}', ha='center', va='bottom', fontweight='bold', fontsize=12, color=bar_color_eod)
plt.tight_layout()
plt.savefig('figura_6_7_b_equal_opportunity.png', dpi=300, bbox_inches='tight')
plt.close()

# =============================================================================
# [PASO 7 y 8] DECISIÓN Y LABORATORIO DE POST-PROCESAMIENTO
# =============================================================================
print("\n[Paso 7] Validando presencia de sesgo...")
status_inicial, di_inicial, eod_inicial = gov_auditor.evaluate_compliance(classified_metric_orig, True)

mitigation_applied = False
classified_metric_eq = None
classified_metric_cal = None
classified_metric_roc = None

if status_inicial == "RECHAZADO":
    print("⚠ ALERTA: Las métricas exceden los umbrales éticos.")
    print("Iniciando laboratorio comparativo de Post-procesamiento...")
    mitigation_applied = True
    
    # --- RUTA A: Equalized Odds ---
    print("\n[Paso 8.A] Ejecutando Equalized Odds...")
    eq_odds = EqOddsPostprocessing(privileged_groups=privileged_groups, unprivileged_groups=unprivileged_groups, seed=42)
    eq_odds.fit(dataset_orig, dataset_pred)
    dataset_pred_eq = eq_odds.predict(dataset_pred)
    classified_metric_eq = ClassificationMetric(dataset_orig, dataset_pred_eq,
                                                unprivileged_groups=unprivileged_groups, privileged_groups=privileged_groups)
    di_eq = classified_metric_eq.disparate_impact()
    eod_eq = classified_metric_eq.equal_opportunity_difference()
    
    # --- RUTA B: Calibrated Equalized Odds ---
    print("\n[Paso 8.B] Ejecutando Calibrated Equalized Odds...")
    cal_eq_odds = CalibratedEqOddsPostprocessing(privileged_groups=privileged_groups, unprivileged_groups=unprivileged_groups, cost_constraint='fnr', seed=42)
    cal_eq_odds.fit(dataset_orig, dataset_pred)
    dataset_pred_cal = cal_eq_odds.predict(dataset_pred)
    classified_metric_cal = ClassificationMetric(dataset_orig, dataset_pred_cal,
                                                 unprivileged_groups=unprivileged_groups, privileged_groups=privileged_groups)
    di_cal = classified_metric_cal.disparate_impact()
    eod_cal = classified_metric_cal.equal_opportunity_difference()
    
    # --- RUTA C: Reject Option Classification ---
    print("\n[Paso 8.C] Ejecutando Reject Option Classification (ROC)...")
    roc = RejectOptionClassification(unprivileged_groups=unprivileged_groups, privileged_groups=privileged_groups)
    roc.fit(dataset_orig, dataset_pred)
    dataset_pred_roc = roc.predict(dataset_pred)
    classified_metric_roc = ClassificationMetric(dataset_orig, dataset_pred_roc,
                                                 unprivileged_groups=unprivileged_groups, privileged_groups=privileged_groups)
    di_roc = classified_metric_roc.disparate_impact()
    eod_roc = classified_metric_roc.equal_opportunity_difference()
    
    # --- GRÁFICOS COMPARATIVOS DE LOS ALGORITMOS ---
    print("\n[Paso 9] Generando gráficos de evaluación del Post-procesamiento...")
    algoritmos = ['Original', 'Eq. Odds', 'Calibrated\nEq. Odds', 'ROC']
    colores = ['#e74c3c', '#3498db', '#9b59b6', '#2ecc71']

    # FIG 6.8 (Evolución DI)
    plt.figure(figsize=(11, 6))
    bars = plt.bar(algoritmos, [di_inicial, di_eq, di_cal, di_roc], color=colores, width=0.5)
    plt.title('Comparativa de Mitigación: Disparate Impact (DI)', fontsize=14, fontweight='bold', pad=20)
    plt.axhline(y=1.0, color='black', linestyle='-', linewidth=2, label='Ideal (1.0)')
    plt.axhline(y=0.8, color='#e74c3c', linestyle='--', linewidth=1.5, label='Umbral (0.8 a 1.25)')
    plt.axhline(y=1.25, color='#e74c3c', linestyle='--', linewidth=1.5)
    plt.axhspan(0.8, 1.25, color='#2ecc71', alpha=0.1, label='Rango Aceptable')
    plt.ylabel('Ratio')
    for bar in bars:
        yval = bar.get_height()
        if not np.isnan(yval) and not np.isinf(yval):
            plt.text(bar.get_x() + bar.get_width()/2, yval + 0.02, f'{yval:.4f}', ha='center', va='bottom', fontweight='bold')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig('figura_6_8_evolucion_di.png', dpi=300, bbox_inches='tight')
    plt.close()

    # FIG 6.9 (Evolución EOD)
    plt.figure(figsize=(11, 6))
    bars = plt.bar(algoritmos, [eod_inicial, eod_eq, eod_cal, eod_roc], color=colores, width=0.5)
    plt.title('Comparativa de Mitigación: Equal Opportunity Diff (EOD)', fontsize=14, fontweight='bold', pad=20)
    plt.axhline(y=0.0, color='black', linestyle='-', linewidth=2, label='Ideal (0.0)')
    plt.axhline(y=-0.1, color='#e74c3c', linestyle='--', linewidth=1.5, label='Umbral (-0.1 a 0.1)')
    plt.axhline(y=0.1, color='#e74c3c', linestyle='--', linewidth=1.5)
    plt.axhspan(-0.1, 0.1, color='#2ecc71', alpha=0.1, label='Rango Aceptable')
    plt.ylabel('Diferencia')
    for bar in bars:
        yval = bar.get_height()
        if not np.isnan(yval) and not np.isinf(yval):
            offset = 0.01 if yval > 0 else -0.05
            plt.text(bar.get_x() + bar.get_width()/2, yval + offset, f'{yval:.4f}', ha='center', va='bottom', fontweight='bold')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig('figura_6_9_evolucion_eod.png', dpi=300, bbox_inches='tight')
    plt.close()

else:
    print("✓ CUMPLIMIENTO: No se detectó sesgo significativo. Saltando mitigación...")


# =============================================================================
# [PASO 10 y 11] COMPILACIÓN Y REPORTE PDF FINAL
# =============================================================================
print("\n[Paso 10] Evaluando métricas finales para el dictamen...")
gov_auditor.print_certification_report(classified_metric_orig, classified_metric_eq, classified_metric_cal, classified_metric_roc, config['dataset_name'], mitigation_applied)

class GovernancePDF(FPDF):
    def header(self):
        self.set_font('Arial', 'B', 15)
        self.cell(0, 10, 'Reporte de Auditoría de Gobernanza Algorítmica'.encode('latin-1', 'ignore').decode('latin-1'), 0, 1, 'C')
        self.set_font('Arial', '', 10)
        self.cell(0, 10, f"Sistema: {config['dataset_name']} | Auditoría automatizada".encode('latin-1', 'ignore').decode('latin-1'), 0, 1, 'C')
        self.ln(5)

    def chapter_title(self, title):
        if self.get_y() > 230: self.add_page()
        self.set_font('Arial', 'B', 12)
        self.set_fill_color(230, 230, 230)
        self.cell(0, 10, title.encode('latin-1', 'ignore').decode('latin-1'), 0, 1, 'L', fill=True)
        self.ln(4)

    def chapter_body(self, text):
        self.set_font('Arial', '', 10)
        self.multi_cell(0, 7, text.encode('latin-1', 'ignore').decode('latin-1'))
        self.ln(2)
        
    def add_figure(self, image_path, title, explanation):
        if os.path.exists(image_path):
            self.chapter_title(title)
            self.image(image_path, x=60, w=90) 
            self.ln(2)
            self.chapter_body(explanation)
            self.ln(5)

    def draw_summary_table(self, data, mitigated):
        self.set_font('Arial', 'B', 8)
        self.set_fill_color(52, 152, 219)
        self.set_text_color(255, 255, 255)
        
        w = [50, 24, 24, 24, 24, 28] if mitigated else [80, 50, 50]
        headers = ['Métrica', 'Original', 'Eq. Odds', 'Cal. EqOdds', 'ROC', 'Umbral Óptimo'] if mitigated else ['Métrica', 'Valor Obtenido', 'Umbral Óptimo']
        
        for i in range(len(headers)):
            self.cell(w[i], 10, headers[i].encode('latin-1', 'ignore').decode('latin-1'), 1, 0, 'C', fill=True)
        self.ln()
        
        self.set_font('Arial', '', 8)
        self.set_text_color(0, 0, 0)
        self.set_fill_color(245, 245, 245)
        fill = False
        for row in data:
            for i in range(len(row)):
                align = 'L' if i == 0 else 'C'
                self.cell(w[i], 10, str(row[i]).encode('latin-1', 'ignore').decode('latin-1'), 1, 0, align, fill=fill)
            self.ln()
            fill = not fill
        self.ln(5)

def generate_pdf_report(auditor, m_orig, m_eq, m_cal, m_roc, mitigated):
    print("\n[Paso 11] Generando PDF de reporte...")
    try:
        pdf = GovernancePDF()
        pdf.set_auto_page_break(auto=True, margin=15)
        pdf.add_page()

        pdf.chapter_title("1. Alcance de la Auditoría")
        pdf.chapter_body(f"Análisis integral de sesgo algorítmico sobre el sistema {config['dataset_name']} focalizado en el atributo protegido '{p_col}'.")

        figuras_pre = [
            ('figura_6_1_real_por_atributo.png', f"Figura 6.1 - {t_name} Real por Atributo Protegido", "Distribución base de los datos históricos."),
            ('figura_6_3_comparacion_real_predicha.png', "Figura 6.3 - Real vs Predicha", "Brecha de predicción entre grupos."),
            ('figura_6_4_matriz_confusion.png', "Figura 6.4 - Matrices de Confusión", "Desglose de errores por grupo."),
            ('figura_6_5_distribucion_scores.png', "Figura 6.5 - Distribución de Scores de Riesgo", f"Densidad de los puntajes asignados, divididos por grupo y resultado real (verde: {neg_label}, rojo: {pos_label})."),
            ('figura_6_6_tasas_error.png', "Figura 6.6 - Tasas de Error Dispares", "Comparativa de Falsos Positivos y Falsos Negativos."),
            ('figura_6_7_disparate_impact.png', "Figura 6.7a - Diagnóstico: Disparate Impact", "Cumplimiento de paridad estadística pre-mitigación."),
            ('figura_6_7_b_equal_opportunity.png', "Figura 6.7b - Diagnóstico: Equal Opportunity Diff", "Cumplimiento de igualdad de oportunidades pre-mitigación.")
        ]
        for path, title, desc in figuras_pre:
            pdf.add_figure(path, title, desc)

        pdf.chapter_title("2. Diagnóstico del Sistema (Caja Negra)")
        st_orig, di_orig, eod_orig = auditor.evaluate_compliance(m_orig, True)
        
        if mitigated:
            mensaje = (f"ALERTA DE SESGO: Se detectó que el modelo original presenta un sesgo estadístico. "
                       f"Impacto Dispar (DI) = {di_orig:.4f} | Diferencia Igualdad Oportunidades (EOD) = {eod_orig:.4f}. "
                       f"Al tratarse de una Caja Negra, se procede a aplicar 3 técnicas de mitigación de Post-procesamiento.")
            pdf.set_text_color(200, 0, 0)
        else:
            mensaje = (f"CUMPLIMIENTO: El modelo analizado NO presenta un sesgo estadístico significativo. "
                       f"Impacto Dispar (DI) = {di_orig:.4f} | Diferencia Igualdad Oportunidades (EOD) = {eod_orig:.4f}. "
                       f"Las métricas se encuentran dentro de los umbrales de tolerancia.")
            pdf.set_text_color(0, 150, 0)

        pdf.set_font('Arial', 'B', 10)
        pdf.multi_cell(0, 7, mensaje.encode('latin-1', 'ignore').decode('latin-1'))
        pdf.set_text_color(0, 0, 0)
        pdf.set_font('Arial', '', 10)
        pdf.ln(5)

        if mitigated:
            pdf.chapter_title("3. Laboratorio Comparativo de Mitigación (Post-procesamiento)")
            pdf.chapter_body("Resultados del impacto de Equalized Odds, Calibrated Equalized Odds y Reject Option Classification sobre las métricas finales:")
            figuras_post = [
                ('figura_6_8_evolucion_di.png', "Figura 6.8 - Evolución: Disparate Impact", "Efecto comparativo sobre el balance demográfico global de las predicciones."),
                ('figura_6_9_evolucion_eod.png', "Figura 6.9 - Evolución: Equal Opportunity Diff", "Efecto comparativo sobre la corrección de errores algorítmicos (falsos positivos/negativos).")
            ]
            for path, title, desc in figuras_post:
                pdf.add_figure(path, title, desc)

        pdf.chapter_title("4. Resumen Consolidado de Métricas")
        if mitigated:
            table_data = [
                ['Accuracy (Precisión)', f"{m_orig.accuracy():.2%}", f"{m_eq.accuracy():.2%}", f"{m_cal.accuracy():.2%}", f"{m_roc.accuracy():.2%}", 'Max'],
                ['Disp. Impact (Impacto Dispar)', f"{m_orig.disparate_impact():.4f}", f"{m_eq.disparate_impact():.4f}", f"{m_cal.disparate_impact():.4f}", f"{m_roc.disparate_impact():.4f}", '[0.8 - 1.25]'],
                ['Equal Opp. (Igualdad Oport.)', f"{m_orig.equal_opportunity_difference():.4f}", f"{m_eq.equal_opportunity_difference():.4f}", f"{m_cal.equal_opportunity_difference():.4f}", f"{m_roc.equal_opportunity_difference():.4f}", '[-0.1 - 0.1]']
            ]
        else:
            table_data = [
                ['Accuracy (Precisión)', f"{m_orig.accuracy():.2%}", 'Maximizar'],
                ['Disparate Impact (Impacto Dispar)', f"{m_orig.disparate_impact():.4f}", '[0.80 - 1.25]'],
                ['Equal Opportunity (Igualdad Oport.)', f"{m_orig.equal_opportunity_difference():.4f}", '[-0.10 - 0.10]']
            ]
            
        pdf.draw_summary_table(table_data, mitigated)

        if mitigated:
            estrategias_cumplen = []
            if auditor.evaluate_compliance(m_eq, True)[0] == "ACEPTADO": estrategias_cumplen.append("Eq. Odds")
            if auditor.evaluate_compliance(m_cal, True)[0] == "ACEPTADO": estrategias_cumplen.append("Calibrated EqOdds")
            if auditor.evaluate_compliance(m_roc, True)[0] == "ACEPTADO": estrategias_cumplen.append("ROC")
            
            if estrategias_cumplen:
                nota_mitigacion = f" (A pesar de que las intervenciones técnicas de {', '.join(estrategias_cumplen)} demostraron viabilidad para corregir el sesgo)."
            else:
                nota_mitigacion = " (Ninguna técnica de post-procesamiento logró revertir el sesgo)."
                
            st_final = f"RECHAZADO. El sistema base ('Caja Negra') incumple las normativas de equidad algorítmica desde su origen{nota_mitigacion}"
        else:
            st_orig_val, _, _ = auditor.evaluate_compliance(m_orig, True)
            st_final = f"{st_orig_val}. El modelo original se encuentra dentro de los umbrales éticos de gobernanza desde su diseño."

        pdf.chapter_title("5. Dictamen Final de Certificación")
        if mitigated:
            pdf.set_text_color(220, 20, 60)
        else:
            pdf.set_text_color(34, 139, 34)
            
        pdf.chapter_body(f"Resultado final de la auditoría: {st_final} Fin del reporte.")
        pdf.set_text_color(0, 0, 0)

        pdf.output(f"Reporte_Gobernanza_{config['dataset_name'].replace(' ', '_')}.pdf")
        print(f"\n[ÉXITO] Reporte generado: Reporte_Gobernanza_{config['dataset_name'].replace(' ', '_')}.pdf")

    except Exception as e:
        print(f"\n[ERROR] Falló la generación del PDF: {e}")

generate_pdf_report(gov_auditor, classified_metric_orig, classified_metric_eq, classified_metric_cal, classified_metric_roc, mitigation_applied)