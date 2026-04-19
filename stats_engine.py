# stats_engine.py
import numpy as np
import scipy.stats as stats


def validar_z(n: int, sigma: float) -> list[str]:
    """
    Valida los supuestos mínimos para aplicar prueba Z.

    Supuestos verificados:
    - n ≥ 30 (Teorema Central del Límite garantiza aprox. normal de x̄)
    - σ > 0 (varianza poblacional positiva)

    Retorna lista de errores (vacía si todo es válido).
    """
    errores = []

    if n < 30:
        errores.append(
            f"La prueba Z requiere n ≥ 30 (Teorema Central del Límite). "
            f"Tu muestra tiene n = {n}. Considera usar la prueba t de Student."
        )

    if sigma <= 0:
        errores.append("La desviación poblacional σ debe ser estrictamente mayor que 0.")

    return errores


def prueba_z(
    media: float,
    mu0: float,
    sigma: float,
    n: int,
    alpha: float,
    tipo: str
) -> tuple[float, float, float, bool]:
    """
    Calcula estadístico Z, p-value, valor crítico y decisión.

    Fórmula: Z = (x̄ - μ₀) / (σ / √n)

    Parámetros
    ----------
    media : Media muestral (x̄)
    mu0   : Media hipotética bajo H₀
    sigma : Desviación estándar poblacional conocida
    n     : Tamaño de muestra
    alpha : Nivel de significancia
    tipo  : 'Bilateral', 'Cola Izquierda', 'Cola Derecha'

    Retorna
    -------
    (z_calculado, p_value, valor_critico, se_rechaza_H0)
    """
    z = (media - mu0) / (sigma / np.sqrt(n))

    if tipo == "Bilateral":
        p = 2 * (1 - stats.norm.cdf(abs(z)))
        crit = stats.norm.ppf(1 - alpha / 2)
        reject = abs(z) > crit

    elif tipo == "Cola Izquierda":
        p = stats.norm.cdf(z)
        crit = stats.norm.ppf(alpha)       # Negativo, p.ej. -1.645
        reject = z < crit

    else:  # Cola Derecha
        p = 1 - stats.norm.cdf(z)
        crit = stats.norm.ppf(1 - alpha)   # Positivo, p.ej. 1.645
        reject = z > crit

    return float(z), float(p), float(crit), bool(reject)


def intervalo_confianza(
    media: float,
    sigma: float,
    n: int,
    alpha: float,
    tipo: str = "Bilateral"
) -> tuple[float, float]:
    """
    Calcula intervalo de confianza coherente con el tipo de prueba.

    - Bilateral  : IC simétrico [x̄ ± z_{α/2} · σ/√n]
    - Cola Izq.  : IC unilateral superior (-∞, x̄ + z_α · σ/√n]
    - Cola Der.  : IC unilateral inferior [x̄ - z_α · σ/√n, +∞)

    Retorna (límite_inferior, límite_superior)
    con ±inf para intervalos abiertos.
    """
    error_estandar = sigma / np.sqrt(n)

    if tipo == "Bilateral":
        z_alpha = stats.norm.ppf(1 - alpha / 2)
        margen = z_alpha * error_estandar
        return media - margen, media + margen

    elif tipo == "Cola Izquierda":
        z_alpha = stats.norm.ppf(1 - alpha)
        return float("-inf"), media + z_alpha * error_estandar

    else:  # Cola Derecha
        z_alpha = stats.norm.ppf(1 - alpha)
        return media - z_alpha * error_estandar, float("inf")