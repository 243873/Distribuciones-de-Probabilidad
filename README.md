# Analizador Estadístico con IA

Aplicación interactiva desarrollada en **Streamlit** para visualizar distribuciones de probabilidad, ejecutar pruebas de hipótesis Z y obtener retroalimentación mediante el asistente de IA **Google Gemini**.

---

## Estructura del Proyecto

```
proyecto_probabilidad/
├── app.py               # Punto de entrada principal
├── data_loader.py       # Carga de CSV y generación de datos sintéticos
├── diagnostics.py       # Análisis estadístico descriptivo y pruebas de normalidad
├── stats_engine.py      # Prueba Z, validaciones e intervalos de confianza
├── plots.py             # Gráficas (histograma+KDE, boxplot, curva Z)
├── gemini_helper.py     # Integración con API de Google Gemini
├── utils.py             # Utilidades (aplicar estilos CSS)
├── styles.css           # Estilos visuales personalizados
├── requirements.txt     # Dependencias del proyecto
└── .env                 # Variables de entorno (NO subir a Git)
```

---

## Instalación

### 1. Clonar el repositorio

```bash
git clone https://github.com/243873/Distribuciones-de-Probabilidad.git
cd Distribuciones-de-Probabilidad
```

### 2. Crear entorno virtual

```bash
python -m venv venv
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate
```

### 3. Instalar dependencias

```bash
pip install -r requirements.txt
```

### 4. Configurar API Key de Gemini

Crea un archivo `.env` en la raíz del proyecto:

```
GEMINI_API_KEY=tu_clave_aqui
```

> Obtén tu clave en: https://aistudio.google.com/app/apikey

---

## Ejecución Local

```bash
streamlit run app.py
```

---

## Despliegue en Streamlit Cloud

1. Sube el proyecto a GitHub (sin el archivo `.env`)
2. Ve a [share.streamlit.io](https://share.streamlit.io) y conecta el repositorio
3. En **Settings → Secrets**, agrega:

```toml
GEMINI_API_KEY = "tu_clave_aqui"
```

---

## Módulos y Funcionalidades

### ① Carga de Datos
- Generación sintética con 4 distribuciones: Normal, Uniforme, Exponencial, Sesgada (Chi²)
- Semilla configurable para reproducibilidad
- Carga de CSV con detección automática de separador y encoding
- Manejo de valores faltantes (NaN)

### ② Visualización
- Histograma con KDE y líneas de media, mediana y ±1σ
- Boxplot con anotaciones de cuartiles
- Diagnóstico automático:
  - Normalidad: Shapiro-Wilk (n < 50) o D'Agostino-Pearson (n ≥ 50)
  - Sesgo: criterio Hair et al. (2010), umbral |0.5|
  - Outliers: método IQR de Tukey (1977)
- Reflexión del estudiante con comparación automática

### ③ Prueba Z
- Varianza poblacional conocida, n ≥ 30
- Hipótesis H₀ y H₁ configurables con notación LaTeX
- Tipos: Bilateral, Cola Izquierda, Cola Derecha
- Nivel de significancia α: 0.01, 0.05, 0.10
- Gráfica de región crítica con zona de rechazo sombreada
- Intervalo de confianza coherente con tipo de prueba

### ④ Asistente IA (Gemini)
- Análisis estructurado en 5 secciones académicas
- Comparación decisión sistema vs decisión del estudiante
- Evaluación de supuestos de la prueba Z
- Recomendación de prueba alternativa si es necesario

---

## Referencias

- Hair, J.F. et al. (2010). *Multivariate Data Analysis* (7th ed.). Pearson.
- Tukey, J.W. (1977). *Exploratory Data Analysis*. Addison-Wesley.
- Montgomery, D.C. & Runger, G.C. (2014). *Applied Statistics and Probability for Engineers* (6th ed.). Wiley.