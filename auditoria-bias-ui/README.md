# ⚖️ Gobernanza Algorítmica UI

**Gobernanza Algorítmica UI** es una herramienta web desarrollada como parte integral de mi tesis de grado.
Su función principal es facilitar la configuración de auditorías de sesgo en modelos de Inteligencia Artificial, específicamente diseñada para interactuar con frameworks de equidad como **AIF360**.

La aplicación permite a investigadores y auditores cargar datasets en formato CSV, mapear variables críticas y exportar un archivo `config.json` estandarizado para pipelines de análisis en Python.

---

## ✨ Características Principales

* 🚀 **Parsing Dinámico:** Extracción automática de encabezados de archivos CSV mediante `PapaParse`.
* ⚖️ **Enfoque en Equidad:** Configuración optimizada para el análisis del algoritmo **COMPAS** y otros casos de estudio de sesgo algorítmico.
* 🛡️ **Validación en Tiempo Real:** Interfaz blindada con `React Hook Form` que impide la generación de archivos incompletos.
* 🎨 **Interfaz Profesional:** Diseño moderno en "Dark Mode" construido con `Tailwind CSS v4` y `Lucide React`.
* 📥 **Exportación Directa:** Generación de JSON con tipado correcto (conversión automática de strings a números donde el script de Python lo requiere).

---

## 🛠️ Stack Tecnológico

* **Core:** [React 18](https://react.dev/) + [Vite](https://vitejs.dev/)
* **Estilos:** [Tailwind CSS v4](https://tailwindcss.com/)
* **Gestión de Formularios:** [React Hook Form](https://react-hook-form.com/)
* **Procesamiento de Datos:** [PapaParse](https://www.papaparse.com/)
* **Iconografía:** [Lucide React](https://lucide.dev/)

---

## 🚀 Instalación y Ejecución

### Requisitos
* **Node.js:** v20.x o superior (Probado en v25.x)
* **npm:** Gestor de paquetes incluido en Node.js

### Pasos para el entorno local

1.  **Clonar el repositorio:**
    ```bash
    git clone https://github.com/juanmagomezmeza/compas-bias-analysis.git
    cd auditoria-bias-ui
    ```

2.  **Instalar dependencias:**
    ```bash
    npm install
    ```

3.  **Correr el servidor de desarrollo:**
    ```bash
    npm run dev
    ```

4.  **Abrir en el navegador:**
    Navegar a `http://localhost:5173`.

---

## 📖 Flujo de Trabajo

1.  **Carga del Dataset:** Subir el archivo `.csv` (ej: `compas_limpio.csv`).
2.  **Mapeo de Atributos:** Seleccionar la Variable Objetivo (`label_name`), el Atributo Protegido (`protected_col`) y el valor privilegiado.
3.  **Configuración de Visualización:** Definir las etiquetas que aparecerán en los reportes gráficos y el umbral de riesgo (`risk_threshold`).
4.  **Descarga:** Una vez que el formulario es válido (campos en verde/completos), descargar el `config.json`.

---

## 📂 Estructura del Proyecto

```text
auditoria-bias-ui/
 ├── src/
 │    ├── App.jsx          # Componente principal y lógica de negocio
 │    ├── index.css        # Directivas de Tailwind CSS v4
 │    └── main.jsx         # Punto de entrada de la aplicación
 ├── tailwind.config.js    # Configuración de estilos
 ├── postcss.config.js     # Configuración de procesador CSS
 └── package.json          # Dependencias y scripts