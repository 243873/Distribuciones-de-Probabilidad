import numpy as np
import scipy.stats as stats

def validar_z(n, sigma):

    errores = []

    if n < 30:
        errores.append("La prueba Z requiere n ≥ 30")

    if sigma <= 0:
        errores.append("σ debe ser mayor que 0")

    return errores


def prueba_z(media, mu0, sigma, n, alpha, tipo):

    z = (media - mu0) / (sigma / np.sqrt(n))

    if tipo == "Bilateral":
        p = 2 * (1 - stats.norm.cdf(abs(z)))
        crit = stats.norm.ppf(1 - alpha/2)
        reject = abs(z) > crit

    elif tipo == "Cola Izquierda":
        p = stats.norm.cdf(z)
        crit = stats.norm.ppf(alpha)
        reject = z < crit

    else:
        p = 1 - stats.norm.cdf(z)
        crit = stats.norm.ppf(1-alpha)
        reject = z > crit

    return z, p, crit, reject


def intervalo_confianza(media, sigma, n, alpha):

    z_alpha = stats.norm.ppf(1-alpha/2)

    margen = z_alpha * sigma / np.sqrt(n)

    return media - margen, media + margen