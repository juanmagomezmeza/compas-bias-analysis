# Framework de Auditoría de Sesgos y Gobernanza Algorítmica ⚖️🤖

Este proyecto es un **Pipeline de Auditoría Agnóstico** diseñado para evaluar, mitigar y certificar sesgos en sistemas de toma de decisiones automatizadas. Utiliza la librería **AIF360 (AI Fairness 360)** de IBM bajo un enfoque de "caja negra", permitiendo auditar modelos predictivos mediante archivos de configuración parametrizados.

Originalmente desarrollado para el análisis del sistema **COMPAS** (justicia penal), el framework es capaz de procesar cualquier conjunto de datos tabular (ej. Riesgo Crediticio, Selección de RRHH) mediante su arquitectura desacoplada.

## 🌟 Características Principales

* **Arquitectura Agnóstica:** Funciona con cualquier dataset mediante archivos `config.json`.
* **Pipeline de 3 Etapas:** Diagnóstico de sesgo, Mitigación (Reweighing / Eq. Odds) y Certificación.
* **Generación de Reportes:** Emite automáticamente un **Reporte de Gobernanza en PDF** con diagnósticos narrativos y dictámenes técnicos ("APROBADO" / "RECHAZADO").
* **Visualización Dinámica:** Genera 9 gráficos estadísticos adaptados automáticamente a los atributos protegidos definidos.

---

## 🛠️ Estructura del Proyecto

```text
├── pipeline_auditoria.py   # Núcleo del framework (Auditoría y Reporte PDF)
├── preparar_compas.py      # Script ETL: Procesa datos de ProPublica (Justicia)
├── preparar_german.py      # Script ETL: Procesa datos de OpenML (Finanzas)
├── config_compas.json      # Configuración para el caso criminal
├── config_german.json      # Configuración para el caso financiero
└── requirements.txt        # Dependencias del sistema

## 📦 Requisitos

- Python 3.10 o superior
- Pip actualizado
- Sistema operativo:
  - Windows 10/11
  - Linux
  - macOS

---

## 🚀 Instalación del proyecto

### 1. Clonar el repositorio

git clone https://github.com/juanmagomezmeza/compas-bias-analysis.git

cd compas-bias-analysis

### 2. Crear y activar el entorno virtual

## IMPORTANTE: Ejecutar todo desde la carpeta raiz!!!

# Windows
python -m venv venv
.\venv\Scripts\activate

# Linux / MacOS
python3 -m venv venv
source venv/bin/activate

### 3. Instalar dependencias
pip install -r requirements.txt

### 3. Ejecución del proyecto
# 1. Preparar los datos (ETL)
python3 preparar_compas.py  # Genera compas_limpio.csv
python3 preparar_german.py  # Genera german_credit_limpio.csv

# 2. Ejecutar la Auditoría
Utiliza el flag --config para indicar qué dataset y reglas de negocio auditar:
python3 pipeline_auditoria.py --config config_compas.json # Para auditar COMPAS
python3 pipeline_auditoria.py --config config_german.json # Para auditar German Credit

## 📊 Entregables de la Auditoría
Al finalizar, el sistema genera automáticamente:

Reporte_Gobernanza_[Nombre].pdf: Documento formal que incluye un análisis narrativo del sesgo detectado, métricas de equidad y el veredicto final.

Figuras PNG: 9 visualizaciones que incluyen matrices de confusión comparativas, tasas de error (FPR/FNR) y el impacto de las técnicas de mitigación.

## 📚 Referencias

AIF360 Documentation: https://github.com/Trusted-AI/AIF360
COMPAS Dataset & Fairness Research: ProPublica (2016)
Dataset German Credit: UCI Machine Learning Repository

## 📄 Licencia
Este proyecto fue desarrollado como parte de una Tesis de Grado sobre ética y sesgo en la IA. Se permite su uso para fines educativos, académicos y de investigación técnica.

