import numpy as np
import scipy.stats as stats

def prueba_z(media, mu0, sigma, n, alpha, tipo):
    z = (media - mu0) / (sigma / np.sqrt(n))

    if tipo == "Bilateral":
        p = 2 * (1 - stats.norm.cdf(abs(z)))
        crit = stats.norm.ppf(1 - alpha/2)
        reject = abs(z) > crit

    elif tipo == "Izquierda":
        p = stats.norm.cdf(z)
        crit = stats.norm.ppf(alpha)
        reject = z < crit

    else:
        p = 1 - stats.norm.cdf(z)
        crit = stats.norm.ppf(1-alpha)
        reject = z > crit

    return z, p, crit, reject