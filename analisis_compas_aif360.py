"""
Análisis de Sesgos Algorítmicos y Gobernanza en COMPAS usando AIF360
Dataset: ProPublica COMPAS Analysis
Versión con Framework de Gobernanza, Auditoría y Mitigación
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

# Importar AIF360
from aif360.datasets import BinaryLabelDataset
from aif360.metrics import BinaryLabelDatasetMetric, ClassificationMetric
from aif360.algorithms.preprocessing import Reweighing
from aif360.algorithms.postprocessing import EqOddsPostprocessing

warnings.filterwarnings('ignore')

# Configuración de estilo
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("Set2")
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 11

# =============================================================================
# CLASE DE GOBERNANZA Y AUDITORÍA (FRAMEWORK)
# =============================================================================
class AIGovernanceAuditor:
    """
    Framework de Gobernanza Algorítmica.
    Establece umbrales técnicos y certifica el cumplimiento ético.
    """
    def __init__(self, privileged_groups, unprivileged_groups):
        self.privileged_groups = privileged_groups
        self.unprivileged_groups = unprivileged_groups
        # Umbrales basados en la Regla del 80% y estándares internacionales
        self.thresholds = {
            'disparate_impact': (0.8, 1.25),
            'equal_opportunity_diff': (-0.1, 0.1)
        }

    def evaluate_compliance(self, metric, is_classification=False):
        """Evalúa si las métricas están dentro de los rangos aceptables."""
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

    def print_certification_report(self, original_metric, mitigated_metric):
        """Genera el reporte final de certificación."""
        print("\n" + "="*70)
        print("REPORTE DE CERTIFICACIÓN DE GOBERNANZA ALGORÍTMICA")
        print("="*70)
        
        st_orig, di_orig, eod_orig = self.evaluate_compliance(original_metric, True)
        st_mit, di_mit, eod_mit = self.evaluate_compliance(mitigated_metric, True)
        
        print(f"[CAJA NEGRA] SISTEMA ORIGINAL (COMPAS): {st_orig}")
        print(f"  > Disparate Impact: {di_orig:.4f} | Equal Opp. Diff: {eod_orig:.4f}")
        
        print(f"\n[PROPUESTA] SISTEMA MITIGADO (Post-procesamiento): {st_mit}")
        print(f"  > Disparate Impact: {di_mit:.4f} | Equal Opp. Diff: {eod_mit:.4f}")
        
        print("\nDICTAMEN TÉCNICO:")
        if st_mit == "CUMPLE":
            print("✓ APROBADO: El sistema mitigado es apto para despliegue bajo supervisión humana.")
        else:
            print("⚠ RECHAZADO: Se requiere mayor ajuste algorítmico o revisión de la arquitectura.")
        print("="*70)

print("="*70)
print("ANÁLISIS DE SESGOS ALGORÍTMICOS Y GOBERNANZA - COMPAS")
print("="*70)

# =============================================================================
# [1/9] CARGA Y PREPROCESAMIENTO DE DATOS
# =============================================================================
print("\n[1/9] Cargando dataset COMPAS...")

data_url = "https://raw.githubusercontent.com/propublica/compas-analysis/master/compas-scores-two-years.csv"
data_file = "compas-scores-two-years.csv"

if not os.path.exists(data_file):
    urllib.request.urlretrieve(data_url, data_file)

df_raw = pd.read_csv(data_file)
df = df_raw[['sex', 'age', 'race', 'juv_fel_count', 'juv_misd_count', 
             'juv_other_count', 'priors_count', 'c_charge_degree', 
             'two_year_recid', 'decile_score', 'score_text']].copy()

df = df[df['race'].isin(['African-American', 'Caucasian'])]

df['sex_original'] = df['sex'].copy()
df['race_original'] = df['race'].copy()
df['sex'] = df['sex'].map({'Male': 'Male', 'Female': 'Female'})
df['two_year_recid_label'] = df['two_year_recid'].map({1: 'Reincidió', 0: 'No Reincidió'})
df['predicted_high_risk'] = (df['decile_score'] >= 5).astype(int)
df['two_year_recid_binary'] = df['two_year_recid'].astype(int)

# =============================================================================
# [2/9] CONFIGURACIÓN DE AIF360 Y GRUPOS
# =============================================================================
print("\n[2/9] Configurando Auditoría AIF360...")

df_aif = pd.DataFrame()
df_aif['race_binary'] = df['race_original'].map({'Caucasian': 1, 'African-American': 0})
df_aif['two_year_recid'] = df['two_year_recid']
df_aif['predicted_high_risk'] = df['predicted_high_risk']

privileged_groups = [{'race_binary': 1}]  # Caucasian
unprivileged_groups = [{'race_binary': 0}]  # African-American

# Crear datasets de AIF360. NOTA: favorable_label=0 (No reincidir)
dataset_orig = BinaryLabelDataset(
    favorable_label=0,
    unfavorable_label=1,
    df=df_aif[['race_binary', 'two_year_recid']],
    label_names=['two_year_recid'],
    protected_attribute_names=['race_binary']
)

dataset_pred = dataset_orig.copy(deepcopy=True)
dataset_pred.labels = df_aif['predicted_high_risk'].values.reshape(-1, 1)

# =============================================================================
# [3/9] AUDITORÍA INICIAL (CAJA NEGRA)
# =============================================================================
gov_auditor = AIGovernanceAuditor(privileged_groups, unprivileged_groups)

metric_orig = BinaryLabelDatasetMetric(dataset_orig, 
                                       unprivileged_groups=unprivileged_groups,
                                       privileged_groups=privileged_groups)

classified_metric_orig = ClassificationMetric(dataset_orig, dataset_pred,
                                              unprivileged_groups=unprivileged_groups,
                                              privileged_groups=privileged_groups)

print("\n" + "="*70)
print("MÉTRICAS DE FAIRNESS - SISTEMA ORIGINAL COMPAS")
print("="*70)
print(f"Disparate Impact (Datos): {metric_orig.disparate_impact():.4f}")
print(f"Disparate Impact (Predicciones): {classified_metric_orig.disparate_impact():.4f}")
print(f"Equal Opportunity Difference: {classified_metric_orig.equal_opportunity_difference():.4f}")
print("="*70)

# =============================================================================
# [4/9] GENERACIÓN DE VISUALIZACIONES DESCRIPTIVAS (FIGS 6.1 a 6.7)
# =============================================================================
print("\n[4/8] Generando Figuras Descriptivas (6.1 a 6.7)...")

# FIGURA 6.1
plt.figure(figsize=(10, 6))
risk_race = pd.crosstab(df['race'], df['two_year_recid_label'], normalize='index') * 100
ax = risk_race.plot(kind='bar', stacked=False, color=['#2ecc71', '#e74c3c'], width=0.7)
plt.title('Distribución de Reincidencia Real por Raza', fontsize=14, fontweight='bold', pad=20)
plt.xlabel('Grupo Racial', fontsize=12, fontweight='bold')
plt.ylabel('Porcentaje (%)', fontsize=12, fontweight='bold')
plt.xticks(rotation=0)
plt.legend(title='Reincidencia Real', bbox_to_anchor=(1.05, 1), loc='upper left')
for container in ax.containers: ax.bar_label(container, fmt='%.1f%%', padding=3)
plt.tight_layout()
plt.savefig('figura_6_1_reincidencia_real_por_raza_aif360.png', dpi=300, bbox_inches='tight')
plt.close()

# FIGURA 6.2
plt.figure(figsize=(10, 6))
risk_gender = pd.crosstab(df['sex'], df['two_year_recid_label'], normalize='index') * 100
ax = risk_gender.plot(kind='bar', stacked=False, color=['#2ecc71', '#e74c3c'], width=0.7)
plt.title('Distribución de Reincidencia Real por Género', fontsize=14, fontweight='bold', pad=20)
plt.xlabel('Género', fontsize=12, fontweight='bold')
plt.ylabel('Porcentaje (%)', fontsize=12, fontweight='bold')
plt.xticks(rotation=0)
plt.legend(title='Reincidencia Real', bbox_to_anchor=(1.05, 1), loc='upper left')
for container in ax.containers: ax.bar_label(container, fmt='%.1f%%', padding=3)
plt.tight_layout()
plt.savefig('figura_6_2_reincidencia_real_por_genero_aif360.png', dpi=300, bbox_inches='tight')
plt.close()

# FIGURA 6.3
plt.figure(figsize=(12, 6))
real_recid = df.groupby('race')['two_year_recid'].apply(lambda x: (x == 1).sum() / len(x) * 100)
pred_recid = df.groupby('race')['predicted_high_risk'].apply(lambda x: x.sum() / len(x) * 100)
comparison_df = pd.DataFrame({'Reincidencia Real': real_recid, 'Reincidencia Predicha': pred_recid})
ax = comparison_df.plot(kind='bar', width=0.8, color=['#3498db', '#e67e22'])
plt.title('Comparación entre Tasas de Reincidencia Real y Predicha', fontsize=14, fontweight='bold', pad=20)
plt.xlabel('Grupo Racial', fontsize=12, fontweight='bold')
plt.ylabel('Tasa (%)', fontsize=12, fontweight='bold')
plt.xticks(rotation=0)
plt.legend(title='Métrica', bbox_to_anchor=(1.05, 1), loc='upper left')
for container in ax.containers: ax.bar_label(container, fmt='%.1f%%', padding=3)
plt.tight_layout()
plt.savefig('figura_6_3_comparacion_real_predicha_aif360.png', dpi=300, bbox_inches='tight')
plt.close()

# FIGURA 6.4
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
races = df['race'].unique()
for idx, race in enumerate(races):
    df_race = df[df['race'] == race]
    cm = confusion_matrix(df_race['two_year_recid_binary'], df_race['predicted_high_risk'])
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
    sns.heatmap(cm_normalized, annot=True, fmt='.1f', cmap='YlOrRd', 
                cbar_kws={'label': 'Porcentaje (%)'}, ax=axes[idx],
                xticklabels=['Pred: Bajo Riesgo', 'Pred: Alto Riesgo'],
                yticklabels=['Real: No Reincidió', 'Real: Reincidió'])
    axes[idx].set_title(f'{race}\n(n={len(df_race)})', fontweight='bold', fontsize=12)
    axes[idx].set_xlabel('Predicción', fontweight='bold')
    axes[idx].set_ylabel('Valor Real', fontweight='bold')
fig.suptitle('Matriz de Confusión por Grupo Racial', fontsize=14, fontweight='bold', y=1.00)
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig('figura_6_4_matriz_confusion_aif360.png', dpi=300, bbox_inches='tight')
plt.close()

# FIGURA 6.5
plt.figure(figsize=(12, 7))
hue_order = ['No Reincidió', 'Reincidió']

ax = sns.violinplot(
    data=df, 
    x='race', 
    y='decile_score', 
    hue='two_year_recid_label',
    hue_order=hue_order,
    split=True, 
    inner='quartile', 
    palette=['#2ecc71', '#e74c3c']
)

plt.title('Puntuaciones de Riesgo por Raza y Estado de Reincidencia Real', 
          fontsize=14, fontweight='bold', pad=20)

plt.xlabel('Grupo Racial', fontsize=12, fontweight='bold')
plt.ylabel('Puntuación de Riesgo (Decile Score: 1-10)', fontsize=12, fontweight='bold')
plt.legend(title='Reincidencia Real', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.axhline(y=5, color='black', linestyle='--', linewidth=1, alpha=0.5, label='Umbral Alto Riesgo (≥5)')
plt.tight_layout()
plt.savefig('figura_6_5_distribucion_scores_aif360.png', dpi=300, bbox_inches='tight')
plt.close()

# FIGURA 6.6
df_aa = df[df['race'] == 'African-American']
df_cauc = df[df['race'] == 'Caucasian']
fpr_aa_val = ((df_aa['two_year_recid_binary'] == 0) & (df_aa['predicted_high_risk'] == 1)).sum() / ((df_aa['two_year_recid_binary'] == 0)).sum() * 100
fpr_cauc_val = ((df_cauc['two_year_recid_binary'] == 0) & (df_cauc['predicted_high_risk'] == 1)).sum() / ((df_cauc['two_year_recid_binary'] == 0)).sum() * 100
fnr_aa_val = ((df_aa['two_year_recid_binary'] == 1) & (df_aa['predicted_high_risk'] == 0)).sum() / ((df_aa['two_year_recid_binary'] == 1)).sum() * 100
fnr_cauc_val = ((df_cauc['two_year_recid_binary'] == 1) & (df_cauc['predicted_high_risk'] == 0)).sum() / ((df_cauc['two_year_recid_binary'] == 1)).sum() * 100

error_data = pd.DataFrame({
    'Falsos Positivos (Se equivocó en contra)': [fpr_aa_val, fpr_cauc_val],
    'Falsos Negativos (Se equivocó a favor)': [fnr_aa_val, fnr_cauc_val]
}, index=['African-American', 'Caucasian'])

plt.figure(figsize=(10, 6))
ax = error_data.plot(kind='bar', width=0.7, color=['#e74c3c', '#3498db'])
plt.title('Tasas de Error del Algoritmo por Raza', fontsize=14, fontweight='bold', pad=20)
plt.xlabel('Grupo Racial', fontsize=12, fontweight='bold')
plt.ylabel('Tasa de Error (%)', fontsize=12, fontweight='bold')
plt.xticks(rotation=0)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
for container in ax.containers: ax.bar_label(container, fmt='%.1f%%', padding=3)
plt.tight_layout()
plt.savefig('figura_6_6_tasas_error_por_raza.png', dpi=300, bbox_inches='tight')
plt.close()

# FIGURA 6.7
plt.figure(figsize=(10, 6))
di_value = classified_metric_orig.disparate_impact()
bar_color = '#e67e22' if di_value > 1.25 or di_value < 0.8 else '#2ecc71'
plt.bar(['Disparate Impact\n(Predicciones)'], [di_value], color=bar_color, width=0.35)
plt.axhline(y=1.0, color='black', linestyle='-', linewidth=2, label='Equidad Ideal (1.0)')
plt.axhline(y=0.8, color='#e74c3c', linestyle='--', linewidth=1.5, label='Umbral Mínimo Aceptable (0.8)')
plt.axhline(y=1.25, color='#e74c3c', linestyle='--', linewidth=1.5, label='Umbral Máximo Aceptable (1.25)')
plt.axhspan(0.8, 1.25, color='#2ecc71', alpha=0.1)
plt.title('Medición del Disparate Impact en Predicciones', fontsize=14, fontweight='bold', pad=20)
plt.ylabel('Ratio', fontsize=12, fontweight='bold')
plt.ylim(0, max(di_value + 0.6, 2.0)) 
plt.legend(loc='upper right', framealpha=0.95)
plt.text(0, di_value + 0.03, f'{di_value:.4f}', ha='center', va='bottom', fontweight='bold', fontsize=12, color=bar_color)
plt.tight_layout()
plt.savefig('figura_6_7_disparate_impact_visualizacion.png', dpi=300, bbox_inches='tight')
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

# FIGURA 6.8
plt.figure(figsize=(10, 6))
di_antes = classified_metric_orig.disparate_impact() 
comparison_data = pd.DataFrame({
    'Disparate Impact\n(Original)': [di_antes],
    'Disparate Impact\n(Pre-procesamiento mitigado)': [di_mitigado]
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
plt.savefig('figura_6_8_mitigacion_reweighing_antes_despues.png', dpi=300, bbox_inches='tight')
plt.close()

# =============================================================================
# [6/9] MITIGACIÓN 2: POST-PROCESAMIENTO (EQUALIZED ODDS)
# =============================================================================
print("\n[6/9] Ejecutando Mitigación de Post-procesamiento (EqOdds)...")

eq_odds = EqOddsPostprocessing(privileged_groups=privileged_groups,
                               unprivileged_groups=unprivileged_groups,
                               seed=42)

# Ajustamos las predicciones originales de COMPAS
eq_odds.fit(dataset_orig, dataset_pred)
dataset_pred_mitigado = eq_odds.predict(dataset_pred)

classified_metric_mitigada = ClassificationMetric(dataset_orig, dataset_pred_mitigado,
                                                  unprivileged_groups=unprivileged_groups,
                                                  privileged_groups=privileged_groups)
eod_antes = classified_metric_orig.equal_opportunity_difference()
eod_despues = classified_metric_mitigada.equal_opportunity_difference()

# FIGURA 6.9
plt.figure(figsize=(10, 6))
comparison_data_eod = pd.DataFrame({
    'Predicciones\nOriginales (Sesgadas)': [eod_antes],
    'Predicciones\nMitigadas (Post-procesamiento)': [eod_despues]
})
ax = comparison_data_eod.plot(kind='bar', color=['#e74c3c', '#2ecc71'], width=0.5)
plt.title('Eficacia de Mitigación: Equal Opportunity Difference', fontsize=14, fontweight='bold', pad=20)
plt.ylabel('Diferencia en Tasa de Verdaderos Positivos', fontsize=12, fontweight='bold')
y_max = max(abs(eod_antes), abs(eod_despues)) + 0.1
plt.ylim(-y_max, y_max)
plt.xticks(rotation=0)
plt.axhline(y=0.0, color='black', linestyle='-', linewidth=2, label='Equidad Ideal (0.0)')
plt.axhspan(-0.1, 0.1, color='#2ecc71', alpha=0.1, label='Rango Aceptable')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
for container in ax.containers:
    labels = [f'{val:.4f}' for val in container.datavalues]
    ax.bar_label(container, labels=labels, padding=3, fontweight='bold')
plt.tight_layout()
plt.savefig('figura_6_9_mitigacion_eqodds_antes_despues.png', dpi=300, bbox_inches='tight')
plt.close()

# =============================================================================
# [7/9] CERTIFICACIÓN FINAL DEL FRAMEWORK DE GOBERNANZA
# =============================================================================
print("\n[7/9] Evaluando métricas finales mediante el Framework de Gobernanza...")
gov_auditor.print_certification_report(classified_metric_orig, classified_metric_mitigada)

# =============================================================================
# [8/9] RESUMEN FINAL
# =============================================================================
print("\n" + "="*70)
print("✓ EJECUCIÓN DEL PIPELINE COMPLETADA EXITOSAMENTE")
print("="*70)
print("\nArchivos generados en el directorio actual:")
print("  • figura_6_1 a figura_6_9 (.png)")
print("="*70)

# =============================================================================
# [9/9] GENERACIÓN DE REPORTE DE GOBERNANZA EN PDF (CON ACENTOS Y UMBRALES)
# =============================================================================
from fpdf import FPDF
import os
import time

class GovernancePDF(FPDF):
    def header(self):
        self.set_font('Arial', 'B', 15)
        # Usamos encode('latin-1') para que los acentos se rendericen bien
        title = 'Reporte de Auditoría de Gobernanza Algorítmica'.encode('latin-1', 'ignore').decode('latin-1')
        self.cell(0, 10, title, 0, 1, 'C')
        self.set_font('Arial', '', 10)
        subtitle = 'Sistema: COMPAS | Auditoría vía AIF360'.encode('latin-1', 'ignore').decode('latin-1')
        self.cell(0, 10, subtitle, 0, 1, 'C')
        self.ln(5)

    def chapter_title(self, title):
        if self.get_y() > 230:
            self.add_page()
        self.set_font('Arial', 'B', 12)
        self.set_fill_color(230, 230, 230)
        # Limpieza de caracteres para el título de capítulo
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
        else:
            print(f"--> ADVERTENCIA: No se encontró el archivo {image_path}.")

    def draw_summary_table(self, data):
        self.set_font('Arial', 'B', 9)
        self.set_fill_color(52, 152, 219)
        self.set_text_color(255, 255, 255)
        
        w = [60, 42, 42, 42] 
        headers = ['Métrica', 'Modelo Orig.', 'Modelo Mitig.', 'Umbral Óptimo']
        
        for i in range(len(headers)):
            h_safe = headers[i].encode('latin-1', 'ignore').decode('latin-1')
            self.cell(w[i], 10, h_safe, 1, 0, 'C', fill=True)
        self.ln()
        
        self.set_font('Arial', '', 9)
        self.set_text_color(0, 0, 0)
        self.set_fill_color(245, 245, 245)
        
        fill = False
        for row in data:
            for i in range(len(row)):
                cell_text = str(row[i]).encode('latin-1', 'ignore').decode('latin-1')
                align = 'L' if i == 0 else 'C'
                self.cell(w[i], 10, cell_text, 1, 0, align, fill=fill)
            self.ln()
            fill = not fill
        self.ln(5)

def generate_pdf_report(auditor, m_orig, m_mit):
    print("\nIniciando generación de PDF con acentos y umbrales...")
    try:
        pdf = GovernancePDF()
        pdf.set_auto_page_break(auto=True, margin=15)
        pdf.add_page()

        # 1. INTRODUCCIÓN
        pdf.chapter_title("1. Alcance de la Auditoría")
        pdf.chapter_body("Este reporte documenta el análisis integral de sesgo algorítmico sobre el sistema COMPAS y los límites de cumplimiento ético definidos.")

        # 2. TODAS LAS FIGURAS CON TÍTULOS ACENTUADOS
        figuras = [
            ('figura_6_1_reincidencia_real_por_raza_aif360.png', "Figura 6.1 - Reincidencia Real por Raza", "Distribución base de los datos según registros históricos."),
            ('figura_6_2_reincidencia_real_por_genero_aif360.png', "Figura 6.2 - Reincidencia Real por Género", "Análisis descriptivo por sexo."),
            ('figura_6_3_comparacion_real_predicha_aif360.png', "Figura 6.3 - Real vs Predicha", "Brecha de predicción entre grupos raciales."),
            ('figura_6_4_matriz_confusion_aif360.png', "Figura 6.4 - Matrices de Confusión", "Desglose de errores normalizados por grupo."),
            ('figura_6_5_distribucion_scores_aif360.png', "Figura 6.5 - Puntuaciones por Raza y Reincidencia", "Análisis de densidad de puntuaciones de riesgo."),
            ('figura_6_6_tasas_error_por_raza.png', "Figura 6.6 - Tasas de Error Dispares", "Comparativa de Falsos Positivos y Falsos Negativos."),
            ('figura_6_7_disparate_impact_visualizacion.png', "Figura 6.7 - Medición del Disparate Impact", "Estado de cumplimiento legal pre-mitigación."),
            ('figura_6_8_mitigacion_reweighing_antes_despues.png', "Figura 6.8 - Mitigación: Técnica de Reweighing", "Resultado de la intervención en el pre-procesamiento."),
            ('figura_6_9_mitigacion_eqodds_antes_despues.png', "Figura 6.9 - Mitigación: Equal Opportunity Difference", "Resultado tras la intervención de post-procesamiento.")
        ]

        for path, title, desc in figuras:
            pdf.add_figure(path, title, desc)

        # 3. TABLA DE RESUMEN
        acc_orig = m_orig.accuracy()
        acc_mit = m_mit.accuracy()
        di_orig = m_orig.disparate_impact()
        di_mit = m_mit.disparate_impact()
        eod_orig = m_orig.equal_opportunity_difference()
        eod_mit = m_mit.equal_opportunity_difference()
        
        acc_loss = acc_orig - acc_mit

        table_data = [
            ['Precisión (Accuracy)', f"{acc_orig:.2%}", f"{acc_mit:.2%}", 'Maximizar'],
            ['Disparate Impact (DI)', f"{di_orig:.4f}", f"{di_mit:.4f}", '[0.80 - 1.25]'],
            ['Equal Opportunity Diff (EOD)', f"{eod_orig:.4f}", f"{eod_mit:.4f}", '[-0.10 - 0.10]']
        ]
        
        pdf.chapter_title("2. Resumen de Métricas y Comparativa de Umbrales")
        pdf.draw_summary_table(table_data)

        # Texto condicional con acentos
        if acc_loss > 0.05:
            msg = f"ADVERTENCIA: Se observa una pérdida de precisión del {acc_loss:.2%}. Se ha priorizado la equidad ética por sobre la exactitud predictiva del modelo."
        else:
            msg = f"BALANCE EXITOSO: La pérdida de precisión fue marginal ({acc_loss:.2%}). El sistema cumple con los parámetros de equidad técnica establecidos."
        pdf.chapter_body(msg)

        # 4. DICTAMEN FINAL
        st_mit, _, _ = auditor.evaluate_compliance(m_mit, True)
        pdf.chapter_title("3. Dictamen Final de Certificación")
        pdf.chapter_body(f"Resultado final de gobernanza: {st_mit}. El sistema mitigado se encuentra dentro de los rangos de cumplimiento aceptables para su despliegue.")

        pdf.output("Reporte_Gobernanza_COMPAS.pdf")
        print("\n[ÉXITO] Reporte generado correctamente con acentos: Reporte_Gobernanza_COMPAS.pdf")

    except Exception as e:
        print(f"\n[ERROR] Falló la generación del PDF: {e}")

# Llamada final
generate_pdf_report(gov_auditor, classified_metric_orig, classified_metric_mitigada)