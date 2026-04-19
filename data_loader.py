# data_loader.py
import pandas as pd
import numpy as np
import streamlit as st


def generar_datos(n: int, mu: float, sigma: float, distribucion: str = "Normal", semilla: int = 42) -> pd.DataFrame:
    """
    Genera datos sintéticos con distribución seleccionable y semilla fija.

    Parámetros
    ----------
    n           : Tamaño de muestra
    mu          : Media objetivo (centro)
    sigma       : Desviación estándar objetivo
    distribucion: 'Normal', 'Uniforme', 'Exponencial', 'Sesgada (Chi²)'
    semilla     : Semilla para reproducibilidad

    Retorna
    -------
    DataFrame con columna 'Variable_X'
    """
    np.random.seed(semilla)

    if distribucion == "Normal":
        datos = np.random.normal(mu, sigma, n)

    elif distribucion == "Uniforme":
        # Uniforme entre [mu - sqrt(3)*sigma, mu + sqrt(3)*sigma]
        # garantiza misma media y desv. estándar aprox.
        rango = np.sqrt(3) * sigma
        datos = np.random.uniform(mu - rango, mu + rango, n)

    elif distribucion == "Exponencial":
        # Exponencial con media mu (lambda = 1/mu)
        datos = np.random.exponential(scale=mu, size=n)

    elif distribucion == "Sesgada (Chi²)":
        # Chi² con k grados, luego reescalada a media mu, sigma deseados
        k = 4
        raw = np.random.chisquare(k, n)
        datos = (raw - raw.mean()) / raw.std() * sigma + mu

    else:
        datos = np.random.normal(mu, sigma, n)

    return pd.DataFrame({"Variable_X": datos})


def cargar_csv(archivo) -> tuple[pd.DataFrame | None, dict]:
    """
    Lee un CSV con manejo robusto de encoding, separadores y valores faltantes.

    Retorna
    -------
    (DataFrame limpio, dict con info de carga)
    """
    info = {"filas": 0, "columnas": 0, "nan_eliminados": 0}

    try:
        # Intentar UTF-8 primero, luego latin-1
        try:
            df = pd.read_csv(archivo, encoding="utf-8", sep=None, engine="python")
        except UnicodeDecodeError:
            archivo.seek(0)
            df = pd.read_csv(archivo, encoding="latin-1", sep=None, engine="python")

        filas_original = len(df)

        # Eliminar filas completamente vacías
        df = df.dropna(how="all")

        # Contar NaN en columnas numéricas antes de limpiar
        nan_count = df.select_dtypes(include=np.number).isna().sum().sum()
        df = df.dropna(subset=df.select_dtypes(include=np.number).columns.tolist())

        info["filas"] = len(df)
        info["columnas"] = len(df.columns)
        info["nan_eliminados"] = filas_original - len(df)

        return df, info

    except Exception as e:
        st.error(f"Error al leer CSV: {e}")
        return None, info