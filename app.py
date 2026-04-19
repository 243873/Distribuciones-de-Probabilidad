# app.py
# Analizador Estadístico con IA

import streamlit as st
import pandas as pd
import numpy as np

from data_loader import cargar_csv, generar_datos
from diagnostics import analizar_distribucion
from stats_engine import prueba_z, validar_z, intervalo_confianza
from plots import histograma_kde, boxplot_chart, curva_z
from gemini_helper import analizar_ia
from utils import aplicar_estilos

# ------------------------------------------------------
# CONFIG
# ------------------------------------------------------
st.set_page_config(
    page_title="Analizador Estadístico con IA",
    layout="wide",
    initial_sidebar_state="expanded"
)

aplicar_estilos()

# ------------------------------------------------------
# SESSION STATE
# ------------------------------------------------------
if "df" not in st.session_state:
    st.session_state.df = None

if "variable" not in st.session_state:
    st.session_state.variable = None

# ------------------------------------------------------
# HEADER
# ------------------------------------------------------
st.title("Analizador Estadístico y Pruebas de Hipótesis")

# ------------------------------------------------------
# MENU
# ------------------------------------------------------
st.sidebar.title("Menú")

menu = st.sidebar.radio(
    "Navegación",
    [
        "Carga de Datos",
        "Visualización",
        "Prueba Z",
        "Asistente IA"
    ]
)

# ======================================================
# CARGA DE DATOS
# ======================================================
if menu == "Carga de Datos":

    st.header("Carga de Datos")

    modo = st.radio(
        "Selecciona origen",
        [
            "Generación Sintética",
            "Cargar CSV"
        ]
    )

    if modo == "Generación Sintética":

        c1, c2, c3 = st.columns(3)

        with c1:
            n = st.slider("Tamaño muestra", 30, 2000, 100)

        with c2:
            mu = st.number_input("Media real", value=50.0)

        with c3:
            sigma = st.number_input("Desviación estándar", value=10.0, min_value=0.01)

        df = generar_datos(n, mu, sigma)
        st.session_state.df = df
        st.session_state.variable = "Variable_X"

    else:

        archivo = st.file_uploader("Sube archivo CSV", type=["csv"])

        if archivo is not None:
            df = cargar_csv(archivo)

            if df is not None:
                st.session_state.df = df

    if st.session_state.df is not None:

        df = st.session_state.df

        cols = df.select_dtypes(include=np.number).columns.tolist()

        if len(cols) == 0:
            st.error("No existen columnas numéricas.")
            st.stop()

        variable = st.selectbox(
            "Selecciona variable numérica",
            cols
        )

        st.session_state.variable = variable

        st.subheader("Vista previa")
        st.dataframe(df, use_container_width=True, height=450)

# ======================================================
# VISUALIZACION
# ======================================================
elif menu == "Visualización":
    st.subheader("Criterios usados")

    st.latex(r"Skewness > 0 \rightarrow Sesgo\ derecha")

    st.latex(r"Skewness < 0 \rightarrow Sesgo\ izquierda")

    st.latex(r"Outliers:\ x<Q1-1.5IQR \quad o \quad x>Q3+1.5IQR")

    if st.session_state.df is None:
        st.warning("Primero carga datos.")
        st.stop()

    df = st.session_state.df
    variable = st.session_state.variable
    x = df[variable]

    st.header("Visualización de Distribución")

    st.pyplot(histograma_kde(x))
    st.pyplot(boxplot_chart(x))

    # Diagnóstico automático
    diag = analizar_distribucion(x)

    st.header("Reflexión del Estudiante")

    normalidad_user = st.radio(
        "¿La distribución parece normal?",
        ["Sí", "No", "Incierto"]
    )

    sesgo_user = st.radio(
        "¿Hay sesgo?",
        ["Sin sesgo", "Izquierda", "Derecha"]
    )

    outliers_user = st.radio(
        "¿Hay outliers?",
        ["Sí", "No"]
    )

    st.header("Diagnóstico Automático")

    c1, c2, c3 = st.columns(3)

    with c1:
        st.metric("Skewness", f"{diag['skew']:.3f}")
        st.metric("Kurtosis", f"{diag['kurtosis']:.3f}")

    with c2:
        st.metric("p Normalidad", f"{diag['p_normal']:.4f}")
        st.metric(
            "¿Normal?",
            "Sí" if diag["normal"] else "No"
        )

    with c3:
        st.metric("Sesgo", diag["sesgo"])
        st.metric("Outliers", diag["outliers"])

    # comparación
    st.subheader("Comparación")

    st.write(
        f"Tu respuesta normalidad: {normalidad_user} | Sistema: {'Sí' if diag['normal'] else 'No'}"
    )

    st.write(
        f"Tu respuesta sesgo: {sesgo_user} | Sistema: {diag['sesgo']}"
    )

    st.write(
        f"Tu respuesta outliers: {outliers_user} | Sistema: {'Sí' if diag['outliers']>0 else 'No'}"
    )

# ======================================================
# PRUEBA Z
# ======================================================
elif menu == "Prueba Z":

    if st.session_state.df is None:
        st.warning("Primero carga datos.")
        st.stop()

    df = st.session_state.df
    variable = st.session_state.variable
    x = df[variable]

    st.header("Prueba Z")

    c1, c2 = st.columns(2)

    with c1:
        mu0 = st.number_input("Hipótesis nula H0 (μ)", value=50.0)
        sigma = st.number_input("Desviación poblacional σ", value=10.0)

    with c2:
        alpha = st.selectbox("Nivel α", [0.01, 0.05, 0.10], index=1)
        tipo = st.selectbox(
            "Tipo de prueba",
            ["Bilateral", "Cola Izquierda", "Cola Derecha"]
        )

    n = len(x)

    errores = validar_z(n, sigma)

    if errores:
        for e in errores:
            st.error(e)
        st.stop()

    media = x.mean()

    z, p, crit, reject = prueba_z(
        media,
        mu0,
        sigma,
        n,
        alpha,
        tipo
    )

    decision = "Rechazar H0" if reject else "No rechazar H0"

    st.subheader("Resultados")

    st.write(f"n = {n}")
    st.write(f"Media muestral = {media:.4f}")
    st.write(f"Z calculado = {z:.4f}")
    st.write(f"Valor crítico = {crit:.4f}")
    st.write(f"p-value = {p:.6f}")
    st.write(f"Decisión = {decision}")

    if reject:
        st.success(f"Como p-value ({p:.4f}) < α ({alpha}), se rechaza H0.")
    else:
        st.info(f"Como p-value ({p:.4f}) ≥ α ({alpha}), no se rechaza H0.")

    # IC
    li, ls = intervalo_confianza(media, sigma, n, alpha)
    st.write(f"IC {(1-alpha)*100:.0f}% = [{li:.4f}, {ls:.4f}]")

    st.subheader("Región Crítica")
    st.pyplot(curva_z(z, crit, tipo))

    st.subheader("Fórmula Intervalo de Confianza")

    st.latex(
    r"IC = \bar{x} \pm z_{\alpha/2}\frac{\sigma}{\sqrt{n}}"
    )

    st.latex(
    rf"IC = {media:.4f} \pm {round(abs(crit),4)}\frac{{{sigma:.4f}}}{{\sqrt{{{n}}}}}"
    )

    st.latex(
    rf"IC = [{li:.4f}, {ls:.4f}]"
    )

    # Guardar state
    st.session_state.resultado = {
        "media": media,
        "mu0": mu0,
        "sigma": sigma,
        "n": n,
        "alpha": alpha,
        "tipo": tipo,
        "z": z,
        "p": p,
        "decision": decision
    }
    st.subheader("Fórmula Aplicada")

    st.latex(r"Z = \frac{\bar{x} - \mu_0}{\sigma / \sqrt{n}}")

    st.latex(
        rf"Z = \frac{{{media:.4f} - {mu0:.4f}}}{{{sigma:.4f}/\sqrt{{{n}}}}}"
    )

    st.latex(
        rf"Z = {z:.4f}"
    )

# ======================================================
# IA
# ======================================================
elif menu == "Asistente IA":

    if "resultado" not in st.session_state:
        st.warning("Primero realiza la prueba Z.")
        st.stop()

    st.header("Asistente IA")

    decision_estudiante = st.radio(
        "¿Cuál sería tu decisión?",
        ["Rechazar H0", "No rechazar H0"]
    )

    if st.button("Analizar con IA"):

        with st.spinner("Consultando Gemini..."):
            respuesta = analizar_ia(
                st.session_state.resultado,
                decision_estudiante
            )

        st.subheader("Respuesta IA")
        st.write(respuesta)