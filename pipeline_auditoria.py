"""
Framework de Auditoría de Sesgos y Gobernanza Algorítmica
Pipeline Agnóstico (Parametrizado vía settings.json y config.json)
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
import json
import urllib.request
import argparse

# Importar AIF360
from aif360.datasets import StandardDataset
from aif360.metrics import BinaryLabelDatasetMetric, ClassificationMetric
from aif360.algorithms.preprocessing import Reweighing
from aif360.algorithms.postprocessing import EqOddsPostprocessing
from fpdf import FPDF

warnings.filterwarnings('ignore')

# Configuración de estilo
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("Set2")
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 11

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
        status = "CUMPLE" if di_compliant else "FUERA DE RANGO"
        
        if is_classification:
            eod = metric.equal_opportunity_difference()
            eod_compliant = self.thresholds['equal_opportunity_diff'][0] <= eod <= self.thresholds['equal_opportunity_diff'][1]
            if not eod_compliant: 
                status = "FUERA DE RANGO"
            return status, di, eod
            
        return status, di

    def print_certification_report(self, original_metric, mitigated_metric, dataset_name):
        print("\n" + "="*70)
        print("REPORTE DE CERTIFICACIÓN DE GOBERNANZA ALGORÍTMICA")
        print("="*70)
        
        st_orig, di_orig, eod_orig = self.evaluate_compliance(original_metric, True)
        st_mit, di_mit, eod_mit = self.evaluate_compliance(mitigated_metric, True)
        
        print(f"[CAJA NEGRA] SISTEMA ORIGINAL ({dataset_name}): {st_orig}")
        print(f"  > Disparate Impact: {di_orig:.4f} | Equal Opp. Diff: {eod_orig:.4f}")
        print(f"\n[PROPUESTA] SISTEMA MITIGADO (Post-procesamiento): {st_mit}")
        print(f"  > Disparate Impact: {di_mit:.4f} | Equal Opp. Diff: {eod_mit:.4f}")
        
        print("\nDICTAMEN TÉCNICO:")
        if st_mit == "CUMPLE":
            print("✓ APROBADO: El sistema mitigado es apto para despliegue bajo supervisión humana.")
        else:
            print("⚠ RECHAZADO: Se requiere mayor ajuste algorítmico o revisión de la arquitectura.")
        print("="*70)

# =============================================================================
# [1/9] CARGA DE CONFIGURACIÓN Y DATOS (VÍA ARGUMENTOS DE TERMINAL)
# =============================================================================
print("\n[1/9] Cargando configuración y dataset...")

# 1. Configurar el lector de argumentos de la terminal
parser = argparse.ArgumentParser(description="Pipeline de Auditoría de Gobernanza Algorítmica")
parser.add_argument('--config', type=str, default='config.json', 
                    help='Ruta al archivo de configuración JSON a utilizar')
args = parser.parse_args()

config_path = args.config

# 2. Verificar que el archivo objetivo exista
if not os.path.exists(config_path):
    raise FileNotFoundError(f"ERROR: No se encontró el archivo de configuración '{config_path}'.")

# 3. Cargar la configuración final
with open(config_path, 'r') as f:
    config = json.load(f)

print(f"[*] Archivo de configuración cargado: {config_path}")
print(f"Dataset activo: {config['dataset_name']}")

# Lógica específica para descargar COMPAS si no existe
if config['data_path'] == "compas-scores-two-years.csv" and not os.path.exists(config['data_path']):
    print("Descargando dataset de prueba...")
    urllib.request.urlretrieve("https://raw.githubusercontent.com/propublica/compas-analysis/master/compas-scores-two-years.csv", config['data_path'])

df_raw = pd.read_csv(config['data_path'])
df = df_raw[config['features_to_keep']].copy()
df = df.dropna()

# Extraer variables de ploteo
p_col = config['plot_mapping']['protected_col']
sec_col = config['plot_mapping'].get('secondary_col', None)
t_col = config['plot_mapping']['target_label_col']
t_name = config['plot_mapping']['target_label_name']
priv_val = config['plot_mapping']['priv_val_name']
unpriv_val = config['plot_mapping']['unpriv_val_name']
risk_col = config['plot_mapping']['risk_score_col']
risk_thresh = config['plot_mapping']['risk_threshold']

# Preprocesamiento genérico para gráficos
df['target_label_text'] = df[t_col].map({1: f'Positivo ({t_name})', 0: f'Negativo (No {t_name})'})
df['predicted_high_risk'] = (df[risk_col] >= risk_thresh).astype(int)
df['target_binary'] = df[t_col].astype(int)

# Filtrar solo los grupos de interés
df = df[df[p_col].isin([priv_val, unpriv_val])]

# =============================================================================
# [2/9] CONFIGURACIÓN DE AIF360 Y GRUPOS
# =============================================================================
print("\n[2/9] Configurando Auditoría AIF360 dinámicamente...")

# 1. Separar SOLO las columnas originales (dejamos fuera las etiquetas de texto de los gráficos)
df_for_aif = df[config['features_to_keep']].copy()

# 2. Detectar automáticamente qué columnas son de texto/categóricas
cat_cols = df_for_aif.select_dtypes(include=['object', 'string', 'category']).columns.tolist()

# 3. Remover los atributos protegidos y el target de esa lista
for col in config['protected_attribute_names'] + [config['label_name']]:
    if col in cat_cols:
        cat_cols.remove(col)

# 4. Instanciar el dataset avisándole cuáles son las columnas categóricas a transformar
dataset_orig = StandardDataset(
    df=df_for_aif,
    label_name=config['label_name'],
    favorable_classes=config['favorable_classes'],
    protected_attribute_names=config['protected_attribute_names'],
    privileged_classes=config['privileged_classes'],
    categorical_features=cat_cols
)

# En AIF360 usando StandardDataset, la clase privilegiada siempre se convierte internamente a 1.0
protected_attr_name = config['protected_attribute_names'][0]
privileged_groups = [{protected_attr_name: 1.0}]
unprivileged_groups = [{protected_attr_name: 0.0}]

dataset_pred = dataset_orig.copy(deepcopy=True)
dataset_pred.labels = df['predicted_high_risk'].values.reshape(-1, 1)

# =============================================================================
# [3/9] AUDITORÍA INICIAL (CAJA NEGRA)
# =============================================================================
print("\n[3/9] Ejecutando Auditoría Inicial...")
gov_auditor = AIGovernanceAuditor(privileged_groups, unprivileged_groups)

metric_orig = BinaryLabelDatasetMetric(dataset_orig, 
                                       unprivileged_groups=unprivileged_groups,
                                       privileged_groups=privileged_groups)

classified_metric_orig = ClassificationMetric(dataset_orig, dataset_pred,
                                              unprivileged_groups=unprivileged_groups,
                                              privileged_groups=privileged_groups)

print(f"Disparate Impact (Datos): {metric_orig.disparate_impact():.4f}")
print(f"Disparate Impact (Predicciones): {classified_metric_orig.disparate_impact():.4f}")
print(f"Equal Opportunity Difference: {classified_metric_orig.equal_opportunity_difference():.4f}")

# =============================================================================
# [4/9] GENERACIÓN DE VISUALIZACIONES DESCRIPTIVAS DINÁMICAS
# =============================================================================
print("\n[4/9] Generando Figuras Descriptivas...")

# FIG 6.1: Distribución Real por Atributo Protegido
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

# FIG 6.2: Distribución Real por Atributo Secundario
if sec_col in df.columns:
    plt.figure(figsize=(10, 6))
    risk_sec = pd.crosstab(df[sec_col], df['target_label_text'], normalize='index') * 100
    ax = risk_sec.plot(kind='bar', stacked=False, color=['#2ecc71', '#e74c3c'], width=0.7)
    plt.title(f'Distribución de {t_name} Real por {sec_col.capitalize()}', fontsize=14, fontweight='bold', pad=20)
    plt.xlabel(f'Grupo ({sec_col})', fontsize=12, fontweight='bold')
    plt.ylabel('Porcentaje (%)', fontsize=12, fontweight='bold')
    plt.xticks(rotation=0)
    plt.legend(title=f'{t_name} Real', bbox_to_anchor=(1.05, 1), loc='upper left')
    for container in ax.containers: ax.bar_label(container, fmt='%.1f%%', padding=3)
    plt.tight_layout()
    plt.savefig('figura_6_2_real_por_secundario.png', dpi=300, bbox_inches='tight')
    plt.close()

# FIG 6.3: Comparación Real vs Predicha
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

# FIG 6.4: Matriz de Confusión Dinámica
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
groups = df[p_col].unique()
for idx, group in enumerate(groups[:2]): 
    df_group = df[df[p_col] == group]
    cm = confusion_matrix(df_group['target_binary'], df_group['predicted_high_risk'])
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
    sns.heatmap(cm_normalized, annot=True, fmt='.1f', cmap='YlOrRd', 
                cbar_kws={'label': 'Porcentaje (%)'}, ax=axes[idx],
                xticklabels=['Pred: Bajo', 'Pred: Alto'],
                yticklabels=['Real: Negativo', 'Real: Positivo'])
    axes[idx].set_title(f'{group}\n(n={len(df_group)})', fontweight='bold', fontsize=12)
    axes[idx].set_xlabel('Predicción', fontweight='bold')
    axes[idx].set_ylabel('Valor Real', fontweight='bold')
fig.suptitle(f'Matriz de Confusión por {p_col.capitalize()}', fontsize=14, fontweight='bold', y=1.00)
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig('figura_6_4_matriz_confusion.png', dpi=300, bbox_inches='tight')
plt.close()

# FIG 6.6: Tasas de Error
df_unpriv = df[df[p_col] == unpriv_val]
df_priv = df[df[p_col] == priv_val]

fpr_unpriv = ((df_unpriv['target_binary'] == 0) & (df_unpriv['predicted_high_risk'] == 1)).sum() / ((df_unpriv['target_binary'] == 0)).sum() * 100
fpr_priv = ((df_priv['target_binary'] == 0) & (df_priv['predicted_high_risk'] == 1)).sum() / ((df_priv['target_binary'] == 0)).sum() * 100
fnr_unpriv = ((df_unpriv['target_binary'] == 1) & (df_unpriv['predicted_high_risk'] == 0)).sum() / ((df_unpriv['target_binary'] == 1)).sum() * 100
fnr_priv = ((df_priv['target_binary'] == 1) & (df_priv['predicted_high_risk'] == 0)).sum() / ((df_priv['target_binary'] == 1)).sum() * 100

error_data = pd.DataFrame({
    'Falsos Positivos (Sesgo en contra)': [fpr_unpriv, fpr_priv],
    'Falsos Negativos (Sesgo a favor)': [fnr_unpriv, fnr_priv]
}, index=[unpriv_val, priv_val])

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

# FIG 6.7: Disparate Impact
plt.figure(figsize=(10, 6))
di_value = classified_metric_orig.disparate_impact()
bar_color = '#e67e22' if di_value > 1.25 or di_value < 0.8 else '#2ecc71'
plt.bar(['Disparate Impact\n(Predicciones)'], [di_value], color=bar_color, width=0.35)
plt.axhline(y=1.0, color='black', linestyle='-', linewidth=2, label='Equidad Ideal (1.0)')
plt.axhline(y=0.8, color='#e74c3c', linestyle='--', linewidth=1.5, label='Umbral Mínimo (0.8)')
plt.axhline(y=1.25, color='#e74c3c', linestyle='--', linewidth=1.5, label='Umbral Máximo (1.25)')
plt.axhspan(0.8, 1.25, color='#2ecc71', alpha=0.1)
plt.title('Medición del Disparate Impact en Predicciones', fontsize=14, fontweight='bold', pad=20)
plt.ylabel('Ratio', fontsize=12, fontweight='bold')
plt.ylim(0, max(di_value + 0.6, 2.0)) 
plt.legend(loc='upper right', framealpha=0.95)
plt.text(0, di_value + 0.03, f'{di_value:.4f}', ha='center', va='bottom', fontweight='bold', fontsize=12, color=bar_color)
plt.tight_layout()
plt.savefig('figura_6_7_disparate_impact.png', dpi=300, bbox_inches='tight')
plt.close()

# =============================================================================
# [5/9] MITIGACIÓN 1: PRE-PROCESAMIENTO (REWEIGHING)
# =============================================================================
print("\n[5/9] Ejecutando Mitigación de Pre-procesamiento (Reweighing)...")
RW = Reweighing(unprivileged_groups=unprivileged_groups, privileged_groups=privileged_groups)
dataset_transf_train = RW.fit_transform(dataset_orig)

metric_transf = BinaryLabelDatasetMetric(dataset_transf_train,
                                         unprivileged_groups=unprivileged_groups,
                                         privileged_groups=privileged_groups)
di_mitigado = metric_transf.disparate_impact()

# FIG 6.8: Comparativa Reweighing
plt.figure(figsize=(10, 6))
di_antes = classified_metric_orig.disparate_impact() 
comparison_data = pd.DataFrame({
    'Disparate Impact\n(Original)': [di_antes],
    'Disparate Impact\n(Mitigado)': [di_mitigado]
})
ax = comparison_data.plot(kind='bar', color=['#e74c3c', '#2ecc71'], width=0.5)
plt.title('Eficacia de Mitigación: Reweighing', fontsize=14, fontweight='bold', pad=20)
plt.ylabel('Ratio', fontsize=12, fontweight='bold')
plt.ylim(0, max(di_antes, di_mitigado) + 0.3)
plt.xticks(rotation=0)
plt.axhline(y=1.0, color='black', linestyle='-', linewidth=2, label='Equidad Ideal (1.0)')
plt.axhline(y=0.8, color='black', linestyle='--', linewidth=1, alpha=0.5)
plt.axhline(y=1.25, color='black', linestyle='--', linewidth=1, alpha=0.5)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
for container in ax.containers: ax.bar_label(container, fmt='%.4f', padding=3, fontweight='bold')
plt.tight_layout()
plt.savefig('figura_6_8_mitigacion_reweighing.png', dpi=300, bbox_inches='tight')
plt.close()

# =============================================================================
# [6/9] MITIGACIÓN 2: POST-PROCESAMIENTO (EQUALIZED ODDS)
# =============================================================================
print("\n[6/9] Ejecutando Mitigación de Post-procesamiento (EqOdds)...")
eq_odds = EqOddsPostprocessing(privileged_groups=privileged_groups,
                               unprivileged_groups=unprivileged_groups,
                               seed=42)
eq_odds.fit(dataset_orig, dataset_pred)
dataset_pred_mitigado = eq_odds.predict(dataset_pred)

classified_metric_mitigada = ClassificationMetric(dataset_orig, dataset_pred_mitigado,
                                                  unprivileged_groups=unprivileged_groups,
                                                  privileged_groups=privileged_groups)
eod_antes = classified_metric_orig.equal_opportunity_difference()
eod_despues = classified_metric_mitigada.equal_opportunity_difference()

# FIG 6.9: Comparativa EqOdds
plt.figure(figsize=(10, 6))
comparison_data_eod = pd.DataFrame({
    'Original (Sesgado)': [eod_antes],
    'Mitigado (Post-proc)': [eod_despues]
})
ax = comparison_data_eod.plot(kind='bar', color=['#e74c3c', '#2ecc71'], width=0.5)
plt.title('Eficacia de Mitigación: Equal Opportunity Difference', fontsize=14, fontweight='bold', pad=20)
plt.ylabel('Diferencia en Tasa Verdaderos Positivos', fontsize=12, fontweight='bold')
y_max = max(abs(eod_antes), abs(eod_despues)) + 0.1
plt.ylim(-y_max, y_max)
plt.xticks(rotation=0)
plt.axhline(y=0.0, color='black', linestyle='-', linewidth=2, label='Equidad Ideal (0.0)')
plt.axhspan(-0.1, 0.1, color='#2ecc71', alpha=0.1, label='Rango Aceptable')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
for container in ax.containers: ax.bar_label(container, labels=[f'{val:.4f}' for val in container.datavalues], padding=3, fontweight='bold')
plt.tight_layout()
plt.savefig('figura_6_9_mitigacion_eqodds.png', dpi=300, bbox_inches='tight')
plt.close()

# =============================================================================
# [7/9] CERTIFICACIÓN FINAL
# =============================================================================
print("\n[7/9] Evaluando métricas finales...")
gov_auditor.print_certification_report(classified_metric_orig, classified_metric_mitigada, config['dataset_name'])

# =============================================================================
# [8/9 & 9/9] GENERACIÓN DE REPORTE DE GOBERNANZA EN PDF 
# =============================================================================
class GovernancePDF(FPDF):
    def header(self):
        self.set_font('Arial', 'B', 15)
        title = 'Reporte de Auditoría de Gobernanza Algorítmica'.encode('latin-1', 'ignore').decode('latin-1')
        self.cell(0, 10, title, 0, 1, 'C')
        self.set_font('Arial', '', 10)
        subtitle = f"Sistema: {config['dataset_name']} | Auditoría automatizada".encode('latin-1', 'ignore').decode('latin-1')
        self.cell(0, 10, subtitle, 0, 1, 'C')
        self.ln(5)

    def chapter_title(self, title):
        if self.get_y() > 230: self.add_page()
        self.set_font('Arial', 'B', 12)
        self.set_fill_color(230, 230, 230)
        safe_title = title.encode('latin-1', 'ignore').decode('latin-1')
        self.cell(0, 10, safe_title, 0, 1, 'L', fill=True)
        self.ln(4)

    def chapter_body(self, text):
        self.set_font('Arial', '', 10)
        safe_text = text.encode('latin-1', 'ignore').decode('latin-1')
        self.multi_cell(0, 7, safe_text)
        self.ln(2)

    def add_figure(self, image_path, title, explanation):
        if os.path.exists(image_path):
            self.chapter_title(title)
            self.image(image_path, x=65, w=80) 
            self.ln(2)
            self.chapter_body(explanation)
            self.ln(5)

    def draw_summary_table(self, data):
        self.set_font('Arial', 'B', 9)
        self.set_fill_color(52, 152, 219)
        self.set_text_color(255, 255, 255)
        w = [60, 42, 42, 42] 
        headers = ['Métrica', 'Modelo Orig.', 'Modelo Mitig.', 'Umbral Óptimo']
        for i in range(len(headers)):
            self.cell(w[i], 10, headers[i].encode('latin-1', 'ignore').decode('latin-1'), 1, 0, 'C', fill=True)
        self.ln()
        self.set_font('Arial', '', 9)
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

def generate_pdf_report(auditor, m_orig, m_mit):
    print("\n[9/9] Generando PDF de reporte...")
    try:
        pdf = GovernancePDF()
        pdf.set_auto_page_break(auto=True, margin=15)
        pdf.add_page()

        pdf.chapter_title("1. Alcance de la Auditoría")
        pdf.chapter_body(f"Análisis integral de sesgo algorítmico sobre el sistema {config['dataset_name']} focalizado en el atributo protegido '{config['plot_mapping']['protected_col']}'.")

        figuras = [
            ('figura_6_1_real_por_atributo.png', "Figura 6.1 - Realidad por Atributo Protegido", "Distribución base de los datos históricos."),
            ('figura_6_3_comparacion_real_predicha.png', "Figura 6.3 - Real vs Predicha", "Brecha de predicción entre grupos."),
            ('figura_6_4_matriz_confusion.png', "Figura 6.4 - Matrices de Confusión", "Desglose de errores por grupo."),
            ('figura_6_6_tasas_error.png', "Figura 6.6 - Tasas de Error Dispares", "Comparativa de Falsos Positivos y Falsos Negativos."),
            ('figura_6_7_disparate_impact.png', "Figura 6.7 - Medición del Disparate Impact", "Cumplimiento legal pre-mitigación."),
            ('figura_6_8_mitigacion_reweighing.png', "Figura 6.8 - Mitigación: Reweighing", "Resultado del pre-procesamiento."),
            ('figura_6_9_mitigacion_eqodds.png', "Figura 6.9 - Mitigación: Equal Opportunity", "Resultado del post-procesamiento.")
        ]

        for path, title, desc in figuras:
            pdf.add_figure(path, title, desc)

        acc_orig = m_orig.accuracy()
        acc_mit = m_mit.accuracy()
        table_data = [
            ['Precisión (Accuracy)', f"{acc_orig:.2%}", f"{acc_mit:.2%}", 'Maximizar'],
            ['Disparate Impact (DI)', f"{m_orig.disparate_impact():.4f}", f"{m_mit.disparate_impact():.4f}", '[0.80 - 1.25]'],
            ['Equal Opportunity Diff', f"{m_orig.equal_opportunity_difference():.4f}", f"{m_mit.equal_opportunity_difference():.4f}", '[-0.10 - 0.10]']
        ]
        
        pdf.chapter_title("2. Resumen de Métricas")
        pdf.draw_summary_table(table_data)

        st_mit, _, _ = auditor.evaluate_compliance(m_mit, True)
        pdf.chapter_title("3. Dictamen Final de Certificación")
        pdf.chapter_body(f"Resultado final de gobernanza: {st_mit}. Fin del reporte.")

        pdf.output(f"Reporte_Gobernanza_{config['dataset_name'].replace(' ', '_')}.pdf")
        print(f"\n[ÉXITO] Reporte generado: Reporte_Gobernanza_{config['dataset_name'].replace(' ', '_')}.pdf")

    except Exception as e:
        print(f"\n[ERROR] Falló la generación del PDF: {e}")

generate_pdf_report(gov_auditor, classified_metric_orig, classified_metric_mitigada)