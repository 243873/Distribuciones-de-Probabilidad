# app.py

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
import google.generativeai as genai
import os
from dotenv import load_dotenv

# ==================================================
# CONFIGURACIÓN GENERAL
# ==================================================
st.set_page_config(
    page_title="Analizador Estadístico con IA",
    layout="wide"
)

st.title("📊 Analizador Estadístico y Pruebas de Hipótesis")

# ==================================================
# CONFIGURAR GEMINI
# ==================================================
load_dotenv()

API_KEY = os.getenv("GEMINI_API_KEY")

if API_KEY:
    genai.configure(api_key=API_KEY)

# ==================================================
# FUNCIÓN IA CON FALLBACK
# ==================================================
def analizar_ia(prompt):

    if not API_KEY:
        return "❌ No se encontró GEMINI_API_KEY en archivo .env"

    modelos = [
        "gemini-2.5-flash",
        "gemini-2.5-pro",
        "gemini-1.5-pro"
    ]

    ultimo_error = ""

    for modelo in modelos:
        try:
            model = genai.GenerativeModel(modelo)
            r = model.generate_content(prompt)
            return r.text
        except Exception as e:
            ultimo_error = str(e)

    return f"❌ Error al usar Gemini:\n{ultimo_error}"

# ==================================================
# CARGA DE DATOS
# ==================================================
st.sidebar.header("1️⃣ Fuente de Datos")

modo = st.sidebar.radio(
    "Selecciona origen:",
    ["Generación Sintética", "Cargar CSV"]
)

if modo == "Generación Sintética":

    n = st.sidebar.slider("Tamaño muestra (n)", 30, 1000, 100)
    mu_real = st.sidebar.number_input("Media real", value=50.0)
    sigma_real = st.sidebar.number_input("Desviación estándar", value=10.0)

    data = np.random.normal(mu_real, sigma_real, n)

    df = pd.DataFrame({
        "Variable_X": data
    })

else:
    archivo = st.sidebar.file_uploader("Subir CSV", type=["csv"])

    if archivo is None:
        st.warning("Sube un archivo CSV para continuar.")
        st.stop()

    df = pd.read_csv(archivo)

# ==================================================
# SELECCIÓN VARIABLE
# ==================================================
columnas_numericas = df.select_dtypes(include=np.number).columns.tolist()

if len(columnas_numericas) == 0:
    st.error("El archivo no contiene columnas numéricas.")
    st.stop()

variable = st.sidebar.selectbox(
    "Selecciona variable numérica:",
    columnas_numericas
)

st.subheader("Vista previa de datos")
st.dataframe(df.head())

# ==================================================
# VISUALIZACIÓN
# ==================================================
st.header("📈 Visualización de Distribución")

col1, col2 = st.columns(2)

with col1:
    fig, ax = plt.subplots()
    sns.histplot(df[variable], kde=True, ax=ax, color="skyblue")
    ax.set_title("Histograma + KDE")
    st.pyplot(fig)

with col2:
    fig, ax = plt.subplots()
    sns.boxplot(x=df[variable], ax=ax, color="lightgreen")
    ax.set_title("Boxplot")
    st.pyplot(fig)

# ==================================================
# REFLEXIÓN DEL ESTUDIANTE
# ==================================================
st.header("🧠 Reflexión del Estudiante")

normalidad = st.radio(
    "¿La distribución parece normal?",
    ["Sí", "No", "Incierto"]
)

sesgo = st.radio(
    "¿Existe sesgo?",
    ["Sin sesgo", "Sesgo izquierda", "Sesgo derecha"]
)

outliers = st.radio(
    "¿Hay outliers?",
    ["Sí", "No"]
)

# ==================================================
# PRUEBA Z
# ==================================================
st.header("🧪 Prueba de Hipótesis Z")

colA, colB = st.columns(2)

with colA:
    mu0 = st.number_input("Hipótesis nula H0 (media)", value=50.0)
    sigma_pob = st.number_input("Desviación poblacional σ", value=10.0)

with colB:
    alpha = st.selectbox(
        "Nivel significancia α",
        [0.01, 0.05, 0.10],
        index=1
    )

    tipo = st.selectbox(
        "Tipo prueba",
        ["Bilateral", "Cola Izquierda", "Cola Derecha"]
    )

# Mostrar hipótesis
if tipo == "Bilateral":
    st.latex(r"H_0:\mu=\mu_0 \quad H_1:\mu \ne \mu_0")
elif tipo == "Cola Izquierda":
    st.latex(r"H_0:\mu=\mu_0 \quad H_1:\mu < \mu_0")
else:
    st.latex(r"H_0:\mu=\mu_0 \quad H_1:\mu > \mu_0")

# ==================================================
# CÁLCULOS
# ==================================================
x = df[variable]

n = len(x)
media = x.mean()

z = (media - mu0) / (sigma_pob / np.sqrt(n))

if tipo == "Bilateral":
    p = 2 * (1 - stats.norm.cdf(abs(z)))
    crit = stats.norm.ppf(1 - alpha / 2)
    reject = abs(z) > crit

elif tipo == "Cola Izquierda":
    p = stats.norm.cdf(z)
    crit = stats.norm.ppf(alpha)
    reject = z < crit

else:
    p = 1 - stats.norm.cdf(z)
    crit = stats.norm.ppf(1 - alpha)
    reject = z > crit

decision_auto = "Rechazar H0" if reject else "No rechazar H0"

# ==================================================
# RESULTADOS
# ==================================================
st.subheader("📌 Resultados")

st.write(f"**Media muestral:** {media:.4f}")
st.write(f"**Tamaño muestra:** {n}")
st.write(f"**Estadístico Z:** {z:.4f}")
st.write(f"**Valor crítico:** {crit:.4f}")
st.write(f"**p-value:** {p:.6f}")
st.write(f"**Decisión automática:** {decision_auto}")

# ==================================================
# DECISIÓN DEL ESTUDIANTE
# ==================================================
decision_estudiante = st.radio(
    "¿Cuál sería tu decisión?",
    ["Rechazar H0", "No rechazar H0"]
)

# ==================================================
# CURVA NORMAL
# ==================================================
st.header("📉 Región Crítica")

fig, ax = plt.subplots(figsize=(10,4))

xs = np.linspace(-4, 4, 1000)
ys = stats.norm.pdf(xs)

ax.plot(xs, ys)

if tipo == "Bilateral":
    ax.fill_between(xs, ys, where=(xs > crit), alpha=0.5)
    ax.fill_between(xs, ys, where=(xs < -crit), alpha=0.5)

elif tipo == "Cola Izquierda":
    ax.fill_between(xs, ys, where=(xs < crit), alpha=0.5)

else:
    ax.fill_between(xs, ys, where=(xs > crit), alpha=0.5)

ax.axvline(z, linestyle="--", label=f"Z={z:.2f}")
ax.legend()

st.pyplot(fig)

# ==================================================
# MÓDULO IA
# ==================================================
st.header("🤖 Asistente IA (Gemini)")

if st.button("Analizar con IA"):

    prompt = f"""
Actúa como profesor estricto de estadística.

Resumen estadístico:

Media muestral = {media:.4f}
Media hipotética = {mu0}
n = {n}
Sigma poblacional = {sigma_pob}
Alpha = {alpha}
Tipo prueba = {tipo}

Resultado:
Z = {z:.4f}
p-value = {p:.6f}

Decisión automática:
{decision_auto}

Decisión del estudiante:
{decision_estudiante}

Observaciones del estudiante:
Normalidad = {normalidad}
Sesgo = {sesgo}
Outliers = {outliers}

Responde:

1. ¿Se rechaza H0?
2. ¿El estudiante acertó?
3. Explica usando Z y p-value.
4. ¿Son razonables los supuestos de prueba Z?
5. Conclusión práctica breve.
"""

    with st.spinner("Consultando Gemini..."):
        respuesta = analizar_ia(prompt)

    st.subheader("📘 Respuesta IA")
    st.write(respuesta)