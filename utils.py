# utils.py
import streamlit as st
import os


def aplicar_estilos():
    """
    Carga y aplica el archivo styles.css al tema de Streamlit.
    Compatible con ejecución local y Streamlit Cloud.
    """
    css_path = os.path.join(os.path.dirname(__file__), "styles.css")

    try:
        with open(css_path, encoding="utf-8") as f:
            css = f.read()
        st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)
    except FileNotFoundError:
        # En producción sin acceso al sistema de archivos
        st.markdown(
            "<style>h1,h2,h3{color:#003049;}</style>",
            unsafe_allow_html=True
        )