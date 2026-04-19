import pandas as pd
import numpy as np
import streamlit as st

def cargar_csv(archivo):
    try:
        return pd.read_csv(archivo)
    except Exception as e:
        st.error(f"Error al leer CSV: {e}")
        return None

def generar_datos(n, mu, sigma):
    datos = np.random.normal(mu, sigma, n)

    return pd.DataFrame({
        "Variable_X": datos
    })