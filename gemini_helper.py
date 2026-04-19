# gemini_helper.py
import os
import google.generativeai as genai
import streamlit as st


def _obtener_api_key() -> str | None:
    """
    Obtiene la API key de Gemini con prioridad:
    1. st.secrets (Streamlit Cloud)
    2. Variables de entorno / .env (local)
    """
    # Streamlit Cloud
    try:
        key = st.secrets.get("GEMINI_API_KEY")
        if key:
            return key
    except Exception:
        pass

    # Local con dotenv o variable de entorno
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass

    return os.getenv("GEMINI_API_KEY")


def analizar_ia(resultados: dict, decision_estudiante: str) -> str:
    """
    Envía un resumen estadístico a Gemini y retorna análisis educativo.

    El prompt incluye:
    - Parámetros de la prueba Z
    - Diagnóstico de distribución (si disponible)
    - Decisión del sistema vs decisión del estudiante
    - Solicitud de evaluación de supuestos
    - Solicitud de alternativas si supuestos no se cumplen

    Parámetros
    ----------
    resultados        : dict con métricas de la prueba Z y diagnóstico
    decision_estudiante: 'Rechazar H₀' o 'No rechazar H₀'

    Retorna
    -------
    str con la respuesta de Gemini (o mensaje de error)
    """
    api_key = _obtener_api_key()

    if not api_key:
        return (
            " **API Key no encontrada.**\n\n"
            "Para usar el asistente IA:\n"
            "- **Local**: Agrega `GEMINI_API_KEY=tu_clave` en el archivo `.env`\n"
            "- **Streamlit Cloud**: Agrega la clave en *Settings → Secrets*"
        )

    try:
        genai.configure(api_key=api_key)

        # Contexto de distribución (si se pasó diagnóstico)
        contexto_distribucion = ""
        if resultados.get("normal") is not None:
            contexto_distribucion = f"""
DIAGNÓSTICO DE DISTRIBUCIÓN (previo a la prueba):
- ¿Los datos son normales?: {"Sí" if resultados['normal'] else "No"}
- Sesgo detectado: {resultados.get('sesgo', 'No analizado')}
- Outliers detectados: {resultados.get('outliers', 'No analizado')}
- Desviación estándar muestral: {resultados.get('std_muestral', 'N/D')}
"""

        prompt = f"""
Eres un profesor universitario de estadística aplicada. Tu objetivo es:
1. Evaluar si la decisión estadística es correcta.
2. Comparar la decisión del sistema con la del estudiante.
3. Evaluar los supuestos de la prueba.
4. Dar una conclusión práctica.

Responde siempre en ESPAÑOL, de forma clara y estructurada usando los siguientes encabezados exactos:

## 1. ¿Se rechaza H₀?
## 2. Evaluación de la decisión del estudiante
## 3. Evaluación de supuestos
## 4. ¿Qué prueba alternativa recomendarías?
## 5. Conclusión práctica

---

PARÁMETROS DE LA PRUEBA Z:
- Tamaño de muestra: n = {resultados['n']}
- Media muestral: x̄ = {resultados['media']:.4f}
- Media hipotética bajo H₀: μ₀ = {resultados['mu0']}
- Desviación estándar poblacional (conocida): σ = {resultados['sigma']}
- Nivel de significancia: α = {resultados['alpha']}
- Tipo de prueba: {resultados['tipo']}

RESULTADOS CALCULADOS:
- Estadístico Z = {resultados['z']:.4f}
- p-value = {resultados['p']:.6f}
- Decisión automática del sistema: {resultados['decision']}
- Decisión del estudiante: {decision_estudiante}

{contexto_distribucion}

Responde de forma educativa, señalando errores conceptuales si el estudiante se equivocó.
Si los supuestos de la prueba Z no se cumplen del todo, indícalo con claridad.
Mantén un tono académico pero accesible para estudiantes de licenciatura.
"""

        model = genai.GenerativeModel("gemini-2.5-flash")
        respuesta = model.generate_content(prompt)
        return respuesta.text

    except Exception as e:
        return (
            f" **Error al consultar Gemini:**\n\n`{e}`\n\n"
            "Verifica que tu API key sea válida y que tengas conexión a internet."
        )