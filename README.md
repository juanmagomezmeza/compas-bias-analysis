# Análisis de Sesgo Algorítmico con AIF360, Pandas y Scikit-Learn

Este proyecto analiza el sesgo algorítmico utilizando el dataset **COMPAS** e implementando métricas de justicia algorítmica mediante la librería **AIF360 (AI Fairness 360)** de IBM.

Incluye generación de visualizaciones, análisis estadístico y evaluación de métricas de equidad como:
- Distribución de riesgo por raza
- Distribución de riesgo por género
- Matriz de confusión por grupo poblacional
- Comparación entre riesgo real y predicho
- Distribución de scores de riesgo
- Métricas de equidad como *Disparate Impact*, *Statistical Parity*, *Equal Opportunity*, etc.

---

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

# IMPORTANTE: Ejecutar todo desde la carpeta raiz!!!

# Windows
python -m venv venv
.\venv\Scripts\activate

# Linux / MacOS
python3 -m venv venv
source venv/bin/activate

### 3. Instalar dependencias
pip install -r requirements.txt

### 3. Ejecución del proyecto
python analisis_compas_aif360.py

## 📊 Visualizaciones generadas

El proyecto genera automáticamente gráficos como:

Boxplots
Violinplots
Histogramas
Mapas de calor
Matrices de confusión
Distribuciones de riesgo por grupo poblacional
Estos se guardan en la carpeta figures/.

## 📚 Referencias

AIF360 Documentation: https://github.com/Trusted-AI/AIF360
COMPAS Dataset & Fairness Research: ProPublica (2016)

## 📄 Licencia
Este proyecto puede utilizarse con fines educativos, de investigación y académicos.

