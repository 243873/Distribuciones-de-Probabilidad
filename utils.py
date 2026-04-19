import streamlit as st

def aplicar_estilos():

    with open("styles.css", encoding="utf-8") as f:
        css = f.read()

    st.markdown(
        f"<style>{css}</style>",
        unsafe_allow_html=True
    )