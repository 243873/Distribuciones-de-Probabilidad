import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
import google.generativeai as genai

# Configuración de página
st.set_page_config(page_title="Analizador Estadístico con IA", layout="wide")

# Configurar API de Gemini (Usa st.secrets en producción)
# genai.configure(api_key=st.secrets["GEMINI_API_KEY"])

st.title("📊 Análisis Estadístico y Pruebas de Hipótesis")
st.write("Navega por las secciones usando el menú superior:")

# --- CREACIÓN DEL MENÚ SUPERIOR (TABS) ---
tab1, tab2, tab3, tab4 = st.tabs([
    "📥 1. Carga de Datos", 
    "📈 2. Visualización", 
    "🧪 3. Prueba Z", 
    "🤖 4. Asistente IA"
])

# Variable global para almacenar los datos
df = None

# ==========================================
# SECCIÓN 1: CARGA DE DATOS
# ==========================================
with tab1:
    st.header("Carga de Datos")
    data_source = st.radio("Origen de datos:", ("Generación Sintética", "Cargar CSV"), horizontal=True)

    if data_source == "Generación Sintética":
        n = st.slider("Tamaño de muestra (n)", 30, 1000, 100)
        mu_real = st.number_input("Media real (oculta al usuario)", value=50.0)
        sigma_real = st.number_input("Desviación estándar real", value=10.0)
        
        # Generar datos normales con algo de ruido/outliers
        data = np.random.normal(loc=mu_real, scale=sigma_real, size=n)
        df = pd.DataFrame({'Variable_X': data})
        st.success("Datos sintéticos generados correctamente.")
        st.dataframe(df, height=250, use_container_width=True)
        
    else:
        uploaded_file = st.file_uploader("Sube tu CSV", type=["csv"])
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
            st.success("Archivo cargado correctamente.")
            st.dataframe(df, height=250, use_container_width=True)
        else:
            st.info("Sube un CSV para continuar con el análisis.")

# Validación de seguridad: Si no hay datos, no ejecutamos el resto de pestañas
if df is not None and not df.empty:
    
    columnas_numericas = df.select_dtypes(include=np.number).columns
    
    # ==========================================
    # SECCIÓN 2: VISUALIZACIÓN
    # ==========================================
    with tab2:
        st.header("Visualización de la Distribución")
        variable = st.selectbox("Selecciona la variable a analizar", columnas_numericas)
        
        col1, col2 = st.columns(2)
        fig_size = (6, 4) # Tamaño fijo para igualar proporciones
        
        with col1:
            fig, ax = plt.subplots(figsize=fig_size)
            sns.histplot(df[variable], kde=True, ax=ax, color='skyblue')
            ax.set_title("Histograma y KDE")
            st.pyplot(fig)

        with col2:
            fig, ax = plt.subplots(figsize=fig_size)
            sns.boxplot(x=df[variable], ax=ax, color='lightgreen')
            ax.set_title("Boxplot (Detección de Outliers)")
            st.pyplot(fig)

        st.markdown("### Reflexión del Estudiante")
        c1, c2, c3 = st.columns(3)
        normalidad = c1.radio("¿La distribución parece normal?", ["Sí", "No", "Incierto"])
        sesgo = c2.radio("¿Hay presencia de sesgo?", ["Sin sesgo", "Sesgo a la izquierda", "Sesgo a la derecha"])
        outliers = c3.radio("¿Observas outliers extremos en el boxplot?", ["Sí", "No"])

    # ==========================================
    # SECCIÓN 3: PRUEBA DE HIPÓTESIS
    # ==========================================
    with tab3:
        st.header("Prueba de Hipótesis (Prueba Z)")

        col_params1, col_params2 = st.columns(2)
        with col_params1:
            mu_hipotetica = st.number_input("Hipótesis Nula (Media H0)", value=50.0)
            sigma_pob = st.number_input("Desviación Estándar Poblacional (σ)", value=10.0)
        with col_params2:
            alpha = st.selectbox("Nivel de Significancia (α)", [0.01, 0.05, 0.10], index=1)
            tipo_prueba = st.selectbox("Tipo de Prueba", ["Bilateral", "Cola Izquierda", "Cola Derecha"])

        # Cálculos estadísticos
        n_muestra = len(df[variable])
        media_muestra = df[variable].mean()
        z_stat = (media_muestra - mu_hipotetica) / (sigma_pob / np.sqrt(n_muestra))

        # Calcular p-value y valores críticos según el tipo de prueba
        if tipo_prueba == "Bilateral":
            p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))
            z_critico = stats.norm.ppf(1 - alpha/2)
            rechazar_h0 = abs(z_stat) > z_critico
        elif tipo_prueba == "Cola Izquierda":
            p_value = stats.norm.cdf(z_stat)
            z_critico = stats.norm.ppf(alpha)
            rechazar_h0 = z_stat < z_critico
        else: # Cola Derecha
            p_value = 1 - stats.norm.cdf(z_stat)
            z_critico = stats.norm.ppf(1 - alpha)
            rechazar_h0 = z_stat > z_critico

        st.subheader("Resultados de la Prueba")
        r1, r2, r3 = st.columns(3)
        r1.metric("Estadístico Z", f"{z_stat:.4f}")
        r2.metric("p-value", f"{p_value:.4f}")
        
        if rechazar_h0:
            r3.error("🚨 Se rechaza H0")
        else:
            r3.success("✅ No hay evidencia para rechazar H0")

        # --- Gráfico de Zonas de Rechazo ---
        fig_z, ax_z = plt.subplots(figsize=(8, 3))
        x_z = np.linspace(-4, 4, 1000)
        y_z = stats.norm.pdf(x_z, 0, 1)
        ax_z.plot(x_z, y_z, label='Distribución Normal Estándar')

        # Sombreado dinámico según la prueba
        if tipo_prueba == "Bilateral":
            ax_z.fill_between(x_z, y_z, where=(x_z > z_critico), color='red', alpha=0.5, label='Rechazo')
            ax_z.fill_between(x_z, y_z, where=(x_z < -z_critico), color='red', alpha=0.5)
        elif tipo_prueba == "Cola Izquierda":
            ax_z.fill_between(x_z, y_z, where=(x_z < z_critico), color='red', alpha=0.5, label='Rechazo')
        else:
            ax_z.fill_between(x_z, y_z, where=(x_z > z_critico), color='red', alpha=0.5, label='Rechazo')

        # Marcar el Z calculado
        ax_z.axvline(z_stat, color='black', linestyle='--', label=f'Z calculado ({z_stat:.2f})')
        ax_z.legend()
        st.pyplot(fig_z)

    # ==========================================
    # SECCIÓN 4: ASISTENTE IA
    # ==========================================
    with tab4:
        st.header("Asistente Estadístico (Gemini)")
        st.write("Consulta a la Inteligencia Artificial para interpretar los resultados de la prueba Z que acabas de realizar.")

        if st.button("Analizar resultados con IA", type="primary"):
            with st.spinner("Consultando a Gemini..."):
                prompt = f"""
                Actúa como un profesor experto en estadística. Se realizó una prueba Z con los siguientes parámetros:
                - Media muestral: {media_muestra:.4f}
                - Media hipotética (H0): {mu_hipotetica}
                - Tamaño de muestra (n): {n_muestra}
                - Desviación estándar poblacional: {sigma_pob}
                - Nivel de significancia (alpha): {alpha}
                - Tipo de prueba: {tipo_prueba}
                
                El estadístico Z calculado fue {z_stat:.4f} y el p-value fue {p_value:.4f}.
                
                Responde a lo siguiente de forma clara y directa:
                1. ¿Se rechaza H0?
                2. Explica la decisión estadística basándote en el Z y el p-value.
                3. ¿Qué implicaciones tiene esto de forma práctica?
                4. ¿Fueron razonables los supuestos (n > 30 y varianza conocida)?
                """
                
                try:
                    model = genai.GenerativeModel('gemini-1.5-flash')
                    response = model.generate_content(prompt)
                    st.success("Análisis completado")
                    st.markdown(response.text)
                except Exception as e:
                    st.error(f"Error al conectar con la API de Gemini: {e}")