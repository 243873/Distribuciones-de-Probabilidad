# app.py — Analizador Estadístico con IA
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
for key, default in [
    ("df", None),
    ("variable", None),
    ("resultado", None),
    ("diagnostico", None),
]:
    if key not in st.session_state:
        st.session_state[key] = default

# ------------------------------------------------------
# HEADER
# ------------------------------------------------------
st.title(" Analizador Estadístico y Pruebas de Hipótesis")

# ------------------------------------------------------
# SIDEBAR — Menú y Flujo
# ------------------------------------------------------
st.sidebar.title("Menú de Navegación")

menu = st.sidebar.radio(
    "Sección",
    [
        "① Carga de Datos",
        "② Visualización",
        "③ Prueba Z",
        "④ Asistente IA"
    ]
)

# Indicador de flujo 
st.sidebar.markdown("---")
st.sidebar.markdown("### Flujo recomendado")
pasos = {
    "① Carga de Datos": st.session_state.df is not None,
    "② Visualización": st.session_state.diagnostico is not None,
    "③ Prueba Z": st.session_state.resultado is not None,
    "④ Asistente IA": st.session_state.resultado is not None,
}
for paso, completado in pasos.items():
    icono = "✅" if completado else "⏳"
    st.sidebar.write(f"{icono} {paso}")

# ======================================================
# CARGA DE DATOS
# ======================================================
if menu == "① Carga de Datos":

    st.header("① Carga de Datos")
    st.info("Selecciona el origen de tus datos. Puedes generar datos sintéticos o subir un archivo CSV propio.")

    modo = st.radio(
        "Selecciona origen",
        ["Generación Sintética", "Cargar CSV"]
    )

    if modo == "Generación Sintética":

        c1, c2, c3, c4 = st.columns(4)

        with c1:
            n = st.slider("Tamaño muestra (n)", 30, 2000, 100)
        with c2:
            mu = st.number_input("Media real (μ)", value=50.0)
        with c3:
            sigma = st.number_input("Desv. estándar (σ)", value=10.0, min_value=0.01)
        with c4:
            distribucion = st.selectbox(
                "Distribución",
                ["Normal", "Uniforme", "Exponencial", "Sesgada (Chi²)"]
            )

        semilla = st.number_input("Semilla aleatoria (reproducibilidad)", value=42, step=1)

        if st.button("Generar Datos"):
            df = generar_datos(n, mu, sigma, distribucion, int(semilla))
            st.session_state.df = df
            st.session_state.variable = "Variable_X"
            st.session_state.resultado = None
            st.session_state.diagnostico = None
            st.success(f"Datos generados: {n} observaciones con distribución {distribucion}.")

    else:
        archivo = st.file_uploader("Sube archivo CSV", type=["csv"])

        if archivo is not None:
            with st.spinner("Leyendo CSV..."):
                df, info = cargar_csv(archivo)

            if df is not None:
                st.session_state.df = df
                st.session_state.resultado = None
                st.session_state.diagnostico = None

                if info["nan_eliminados"] > 0:
                    st.warning(
                        f"Se eliminaron {info['nan_eliminados']} filas con valores faltantes."
                    )
                st.success(f"CSV cargado: {info['filas']} filas × {info['columnas']} columnas.")

    # Selección de variable y vista previa
    if st.session_state.df is not None:

        df = st.session_state.df
        cols = df.select_dtypes(include=np.number).columns.tolist()

        if len(cols) == 0:
            st.error("No existen columnas numéricas en el archivo.")
            st.stop()

        variable = st.selectbox("Selecciona variable numérica a analizar", cols)
        st.session_state.variable = variable

        st.subheader("Vista previa")
        st.dataframe(df[[variable]].describe().T, use_container_width=True)
        st.dataframe(df, use_container_width=True, height=350)

# ======================================================
# VISUALIZACIÓN
# ======================================================
elif menu == "② Visualización":

    if st.session_state.df is None:
        st.warning(" Primero carga datos en la sección ① Carga de Datos.")
        st.stop()

    df = st.session_state.df
    variable = st.session_state.variable
    x = df[variable].dropna()

    st.header("② Visualización de Distribución")
    st.caption(f"Variable analizada: **{variable}** | n = {len(x)}")

    # Gráficas
    with st.spinner("Generando gráficas..."):
        st.pyplot(histograma_kde(x, variable))
        st.pyplot(boxplot_chart(x, variable))

    st.markdown("---")

    # Criterios documentados JUNTO al diagnóstico
    st.header("Diagnóstico Automático")

    with st.expander(" Ver criterios estadísticos utilizados", expanded=False):
        st.markdown("**Prueba de normalidad:**")
        st.latex(r"H_0: \text{Los datos provienen de una distribución normal}")
        st.markdown("- n < 50 → Shapiro-Wilk | n ≥ 50 → D'Agostino-Pearson")
        st.markdown("- Si p-value > 0.05 → No se rechaza normalidad")

        st.markdown("**Sesgo (Skewness):**")
        st.latex(r"Skewness > 0.5 \Rightarrow \text{Sesgo a la derecha}")
        st.latex(r"Skewness < -0.5 \Rightarrow \text{Sesgo a la izquierda}")
        st.markdown("*(Hair et al., 2010: |skew| < 2 como aceptable para normalidad)*")

        st.markdown("**Outliers (Método IQR):**")
        st.latex(r"x < Q_1 - 1.5 \cdot IQR \quad \text{ó} \quad x > Q_3 + 1.5 \cdot IQR")

    with st.spinner("Calculando diagnóstico..."):
        diag = analizar_distribucion(x)
        st.session_state.diagnostico = diag

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Skewness", f"{diag['skew']:.3f}")
        st.metric("Kurtosis", f"{diag['kurtosis']:.3f}")
    with c2:
        st.metric("p-value normalidad", f"{diag['p_normal']:.4f}")
        st.metric("Prueba usada", diag["prueba_normalidad"])
    with c3:
        st.metric("¿Normal?", " Sí" if diag["normal"] else "❌ No")
        st.metric("Sesgo detectado", diag["sesgo"])
    with c4:
        st.metric("Outliers detectados", diag["outliers"])
        st.metric("Media", f"{x.mean():.3f}")

    st.markdown("---")

    # Reflexión del estudiante — activada con botón
    st.header("Reflexión del Estudiante")
    st.caption("Responde ANTES de ver la comparación. Luego presiona el botón.")

    col1, col2, col3 = st.columns(3)
    with col1:
        normalidad_user = st.radio("¿La distribución parece normal?", ["Sí", "No", "Incierto"])
    with col2:
        sesgo_user = st.radio("¿Hay sesgo?", ["Sin sesgo", "Izquierda", "Derecha"])
    with col3:
        outliers_user = st.radio("¿Hay outliers?", ["Sí", "No"])

    if st.button(" Confirmar mis respuestas y comparar"):
        st.subheader("Comparación de Respuestas")

        # Normalidad
        sistema_normal = "Sí" if diag["normal"] else "No"
        match_n = (normalidad_user == sistema_normal) or normalidad_user == "Incierto"
        st.write(
            f"**Normalidad** → Tu respuesta: `{normalidad_user}` | Sistema: `{sistema_normal}` "
            + ("✅" if normalidad_user == sistema_normal else "⚠️ Difieren" if normalidad_user != "Incierto" else "")
        )

        # Sesgo
        st.write(
            f"**Sesgo** → Tu respuesta: `{sesgo_user}` | Sistema: `{diag['sesgo']}` "
            + ("✅" if sesgo_user == diag["sesgo"] else "⚠️ Difieren")
        )

        # Outliers
        sistema_outliers = "Sí" if diag["outliers"] > 0 else "No"
        st.write(
            f"**Outliers** → Tu respuesta: `{outliers_user}` | Sistema: `{sistema_outliers}` "
            + ("✅" if outliers_user == sistema_outliers else "⚠️ Difieren")
        )

# ======================================================
# PRUEBA Z
# ======================================================
elif menu == "③ Prueba Z":

    if st.session_state.df is None:
        st.warning("⚠️ Primero carga datos en la sección ① Carga de Datos.")
        st.stop()

    df = st.session_state.df
    variable = st.session_state.variable
    x = df[variable].dropna()
    n = len(x)
    media = x.mean()
    std_muestral = x.std()

    st.header("③ Prueba Z de Hipótesis")
    st.caption(f"Variable: **{variable}** | n = {n} | Media muestral = {media:.4f}")

    st.subheader("Definición de Hipótesis")

    c1, c2 = st.columns(2)
    with c1:
        mu0 = st.number_input("Hipótesis nula H₀: μ =", value=50.0)
        tipo = st.selectbox(
            "Tipo de prueba (H₁)",
            ["Bilateral", "Cola Izquierda", "Cola Derecha"]
        )
    with c2:
        st.info(
            f"σ muestral de tus datos: **{std_muestral:.4f}**\n\n"
            "Usa este valor como referencia al ingresar σ poblacional."
        )
        sigma = st.number_input("Desviación poblacional σ (conocida)", value=round(std_muestral, 2))
        alpha = st.selectbox("Nivel de significancia (α)", [0.01, 0.05, 0.10], index=1)

    # Mostrar hipótesis formalmente
    st.subheader("Hipótesis Planteadas")
    col1, col2 = st.columns(2)
    with col1:
        st.latex(rf"H_0: \mu = {mu0}")
    with col2:
        simbolo = r"\neq" if tipo == "Bilateral" else "<" if tipo == "Cola Izquierda" else ">"
        st.latex(rf"H_1: \mu {simbolo} {mu0}")

    # Validación
    errores = validar_z(n, sigma)
    if errores:
        for e in errores:
            st.error(e)
        st.stop()

    # Cálculo
    z, p, crit, reject = prueba_z(media, mu0, sigma, n, alpha, tipo)
    decision = "Rechazar H₀" if reject else "No rechazar H₀"

    st.markdown("---")
    st.subheader("Resultados de la Prueba")

    r1, r2, r3, r4 = st.columns(4)
    r1.metric("Z calculado", f"{z:.4f}")
    r2.metric("Valor crítico", f"{crit:.4f}")
    r3.metric("p-value", f"{p:.6f}")
    r4.metric("Decisión", decision)

    if reject:
        st.success(f" Como p-value ({p:.4f}) < α ({alpha}), **se rechaza H₀**.")
    else:
        st.info(f"ℹ Como p-value ({p:.4f}) ≥ α ({alpha}), **no se rechaza H₀**.")

    # Fórmulas aplicadas
    st.subheader("Fórmula Aplicada")
    st.latex(r"Z = \frac{\bar{x} - \mu_0}{\sigma / \sqrt{n}}")
    st.latex(
        rf"Z = \frac{{{media:.4f} - {mu0:.4f}}}{{{sigma:.4f} / \sqrt{{{n}}}}}"
    )
    st.latex(rf"Z = {z:.4f}")

    # Intervalo de confianza
    li, ls = intervalo_confianza(media, sigma, n, alpha, tipo)
    st.subheader(f"Intervalo de Confianza al {(1-alpha)*100:.0f}%")

    if tipo == "Bilateral":
        st.latex(r"IC = \bar{x} \pm z_{\alpha/2} \cdot \frac{\sigma}{\sqrt{n}}")
        st.latex(rf"IC = [{li:.4f},\ {ls:.4f}]")
    elif tipo == "Cola Izquierda":
        st.latex(r"IC = \left(-\infty,\ \bar{x} + z_{\alpha} \cdot \frac{\sigma}{\sqrt{n}}\right)")
        st.latex(rf"IC = \left(-\infty,\ {ls:.4f}\right)")
    else:
        st.latex(r"IC = \left(\bar{x} - z_{\alpha} \cdot \frac{\sigma}{\sqrt{n}},\ +\infty\right)")
        st.latex(rf"IC = \left({li:.4f},\ +\infty\right)")

    # Gráfica de región crítica
    st.subheader("Región Crítica y Zona de Rechazo")
    with st.spinner("Generando gráfica..."):
        st.pyplot(curva_z(z, crit, tipo, alpha, reject))

    # Guardar en session state
    st.session_state.resultado = {
        "media": media,
        "mu0": mu0,
        "sigma": sigma,
        "n": n,
        "alpha": alpha,
        "tipo": tipo,
        "z": z,
        "p": p,
        "decision": decision,
        "std_muestral": std_muestral,
        "normal": st.session_state.diagnostico["normal"] if st.session_state.diagnostico else None,
        "sesgo": st.session_state.diagnostico["sesgo"] if st.session_state.diagnostico else "No analizado",
        "outliers": st.session_state.diagnostico["outliers"] if st.session_state.diagnostico else "No analizado",
    }

# ======================================================
# ASISTENTE IA
# ======================================================
elif menu == "④ Asistente IA":

    if st.session_state.resultado is None:
        st.warning(" Primero realiza la Prueba Z en la sección ③.")
        st.stop()

    res = st.session_state.resultado

    st.header("④ Asistente IA — Gemini")
    st.caption("El asistente evalúa tu decisión estadística y la compara con el resultado del sistema.")

    # Resumen del caso
    with st.expander(" Ver resumen del caso enviado a la IA", expanded=True):
        c1, c2, c3 = st.columns(3)
        c1.metric("n", res["n"])
        c1.metric("Media muestral", f"{res['media']:.4f}")
        c2.metric("μ₀ (H₀)", res["mu0"])
        c2.metric("σ poblacional", res["sigma"])
        c3.metric("Z calculado", f"{res['z']:.4f}")
        c3.metric("p-value", f"{res['p']:.6f}")
        st.write(f"**Tipo de prueba:** {res['tipo']} | **α:** {res['alpha']}")
        st.write(f"**Decisión del sistema:** `{res['decision']}`")

    st.markdown("---")

    decision_estudiante = st.radio(
        "¿Cuál es tu decisión estadística?",
        ["Rechazar H₀", "No rechazar H₀"],
        horizontal=True
    )

    if st.button(" Analizar con IA"):
        with st.spinner("Consultando Gemini..."):
            respuesta = analizar_ia(res, decision_estudiante)

        st.subheader("Respuesta del Asistente IA")
        st.markdown(respuesta)

        # Comparación visual explícita
        st.markdown("---")
        st.subheader("Comparación Final")
        col1, col2, col3 = st.columns(3)
        col1.metric("Tu decisión", decision_estudiante)
        col2.metric("Decisión del sistema", res["decision"])
        col3.metric(
            "¿Coinciden?",
            " Sí" if decision_estudiante == res["decision"] else " No"
        )