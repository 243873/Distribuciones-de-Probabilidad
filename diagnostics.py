import scipy.stats as stats
import numpy as np

def analizar_distribucion(x):

    skew = stats.skew(x)
    kurtosis = stats.kurtosis(x)

    try:
        p_normal = stats.normaltest(x).pvalue
    except:
        p_normal = 0.0

    normal = p_normal > 0.05

    if skew > 0.5:
        sesgo = "Derecha"
    elif skew < -0.5:
        sesgo = "Izquierda"
    else:
        sesgo = "Sin sesgo"

    q1 = np.percentile(x,25)
    q3 = np.percentile(x,75)
    iqr = q3-q1

    li = q1 - 1.5*iqr
    ls = q3 + 1.5*iqr

    outliers = ((x<li)|(x>ls)).sum()

    return {
        "skew": skew,
        "kurtosis": kurtosis,
        "p_normal": p_normal,
        "normal": normal,
        "sesgo": sesgo,
        "outliers": outliers
    }