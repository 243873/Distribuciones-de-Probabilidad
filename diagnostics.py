# diagnostics.py
import scipy.stats as stats
import numpy as np

def analizar_distribucion(x) -> dict:
    """
    Analiza distribución de una serie numérica.

    Selección automática de prueba de normalidad:
    - n < 50 → Shapiro-Wilk (más potente para muestras pequeñas)
    - n ≥ 50 → D'Agostino-Pearson (robusto para muestras grandes)

    Criterio de sesgo: |skew| > 0.5 (Hair et al., 2010)
    Criterio de outliers: método IQR estándar (Tukey, 1977)

    Parámetros
    ----------
    x : array-like de valores numéricos

    Retorna
    -------
    dict con métricas estadísticas
    """
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]  # Eliminar NaN por seguridad

    n = len(x)

    # Skewness y Kurtosis
    skew = float(stats.skew(x))
    kurtosis = float(stats.kurtosis(x))  # Kurtosis exceso (0 = normal)

    # Normalidad — selección automática de prueba
    try:
        if n < 50:
            stat_n, p_normal = stats.shapiro(x)
            prueba_normalidad = "Shapiro-Wilk"
        else:
            stat_n, p_normal = stats.normaltest(x)
            prueba_normalidad = "D'Agostino-Pearson"
    except Exception:
        p_normal = 0.0
        stat_n = 0.0
        prueba_normalidad = "Error en prueba"

    normal = p_normal > 0.05

    # Sesgo categórico (Hair et al., 2010 — umbral |0.5|)
    if skew > 0.5:
        sesgo = "Derecha"
    elif skew < -0.5:
        sesgo = "Izquierda"
    else:
        sesgo = "Sin sesgo"

    # Outliers — método IQR (Tukey, 1977)
    q1 = float(np.percentile(x, 25))
    q3 = float(np.percentile(x, 75))
    iqr = q3 - q1
    li = q1 - 1.5 * iqr
    ls = q3 + 1.5 * iqr
    outliers = int(((x < li) | (x > ls)).sum())

    return {
        "skew": skew,
        "kurtosis": kurtosis,
        "p_normal": float(p_normal),
        "stat_normalidad": float(stat_n),
        "prueba_normalidad": prueba_normalidad,
        "normal": normal,
        "sesgo": sesgo,
        "outliers": outliers,
        "q1": q1,
        "q3": q3,
        "iqr": iqr,
        "limite_inferior_iqr": li,
        "limite_superior_iqr": ls,
        "n": n,
    }