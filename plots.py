# plots.py
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import numpy as np
import scipy.stats as stats

# Paleta institucional
COLOR_PRIMARIO   = "#003049"
COLOR_ACENTO     = "#D62828"
COLOR_DESTAQUE   = "#F77F00"
COLOR_RECHAZO    = "#D62828"
COLOR_ACEPTACION = "#6DB1BF"
COLOR_Z_CALC     = "#F77F00"
COLOR_Z_CRIT     = "#D62828"


def histograma_kde(x, nombre_variable: str = "Variable"):
    """
    Histograma con KDE superpuesto.
    Incluye líneas de media, mediana y ±1σ.
    """
    fig, ax = plt.subplots(figsize=(12, 5))
    fig.patch.set_facecolor("#FAFAFA")
    ax.set_facecolor("#FAFAFA")

    # Histograma + KDE
    sns.histplot(x, kde=True, color=COLOR_DESTAQUE, alpha=0.6, ax=ax,
                 edgecolor="white", linewidth=0.5)

    media = x.mean()
    mediana = x.median() if hasattr(x, "median") else np.median(x)
    std = x.std() if hasattr(x, "std") else np.std(x)

    # Líneas de referencia
    ax.axvline(media, color=COLOR_PRIMARIO, linestyle="--", linewidth=2,
               label=f"Media = {media:.2f}")
    ax.axvline(mediana, color=COLOR_ACENTO, linestyle=":", linewidth=2,
               label=f"Mediana = {mediana:.2f}")
    ax.axvline(media + std, color="#888888", linestyle="-.", linewidth=1,
               label=f"+1σ = {media+std:.2f}")
    ax.axvline(media - std, color="#888888", linestyle="-.", linewidth=1,
               label=f"−1σ = {media-std:.2f}")

    ax.set_title(f"Histograma + KDE — {nombre_variable}", fontsize=14,
                 fontweight="bold", color=COLOR_PRIMARIO, pad=12)
    ax.set_xlabel("Valor", fontsize=11)
    ax.set_ylabel("Frecuencia", fontsize=11)
    ax.legend(fontsize=9, framealpha=0.7)
    ax.grid(alpha=0.2, linestyle="--")

    # Anotación de n
    ax.text(0.98, 0.97, f"n = {len(x)}", transform=ax.transAxes,
            ha="right", va="top", fontsize=9, color=COLOR_PRIMARIO,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7))

    fig.tight_layout()
    return fig


def boxplot_chart(x, nombre_variable: str = "Variable"):
    """
    Boxplot con anotaciones de Q1, Q3, mediana y outliers.
    """
    fig, ax = plt.subplots(figsize=(12, 3.5))
    fig.patch.set_facecolor("#FAFAFA")
    ax.set_facecolor("#FAFAFA")

    sns.boxplot(x=x, color=COLOR_DESTAQUE, ax=ax,
                linewidth=1.5, flierprops=dict(
                    marker="o", markerfacecolor=COLOR_ACENTO,
                    markersize=5, alpha=0.6
                ))

    q1     = np.percentile(x, 25)
    q3     = np.percentile(x, 75)
    mediana = np.median(x)
    iqr    = q3 - q1

    # Anotaciones de cuartiles
    for val, etiqueta, pos_y in [
        (q1, f"Q1={q1:.2f}", 0.75),
        (mediana, f"Med={mediana:.2f}", 0.25),
        (q3, f"Q3={q3:.2f}", 0.75),
    ]:
        ax.annotate(
            etiqueta, xy=(val, 0), xytext=(val, pos_y),
            ha="center", fontsize=8, color=COLOR_PRIMARIO,
            arrowprops=dict(arrowstyle="-", color=COLOR_PRIMARIO, alpha=0.4),
        )

    ax.set_title(f"Boxplot — {nombre_variable}  |  IQR = {iqr:.3f}",
                 fontsize=13, fontweight="bold", color=COLOR_PRIMARIO, pad=10)
    ax.set_xlabel("Valor", fontsize=10)
    ax.grid(alpha=0.2, linestyle="--", axis="x")

    fig.tight_layout()
    return fig


def curva_z(z_calc: float, crit: float, tipo: str, alpha: float, reject: bool):
    """
    Curva de densidad normal estándar con:
    - Zona de rechazo sombreada en rojo
    - Zona de no rechazo sombreada en azul
    - Línea vertical del Z calculado (naranja)
    - Línea(s) vertical(es) del/los valor(es) crítico(s) (rojo punteado)
    - Anotaciones de texto con valores
    - Leyenda completa
    - Indicación visual de la decisión
    """
    fig, ax = plt.subplots(figsize=(12, 5))
    fig.patch.set_facecolor("#FAFAFA")
    ax.set_facecolor("#FAFAFA")

    xs = np.linspace(-4.5, 4.5, 1000)
    ys = stats.norm.pdf(xs)

    # --- Zona de aceptación (fondo azul claro) ---
    ax.fill_between(xs, ys, color=COLOR_ACEPTACION, alpha=0.15, label="Zona de no rechazo")

    # --- Zona(s) de rechazo ---
    if tipo == "Bilateral":
        ax.fill_between(xs, ys, where=(xs > crit),
                        color=COLOR_RECHAZO, alpha=0.4, label=f"Zona de rechazo (α/2={alpha/2})")
        ax.fill_between(xs, ys, where=(xs < -crit),
                        color=COLOR_RECHAZO, alpha=0.4)
        # Líneas críticas bilaterales
        ax.axvline( crit, color=COLOR_Z_CRIT, linestyle="-.", linewidth=1.8,
                    label=f"Z crítico = ±{crit:.3f}")
        ax.axvline(-crit, color=COLOR_Z_CRIT, linestyle="-.", linewidth=1.8)

    elif tipo == "Cola Izquierda":
        ax.fill_between(xs, ys, where=(xs < crit),
                        color=COLOR_RECHAZO, alpha=0.4, label=f"Zona de rechazo (α={alpha})")
        ax.axvline(crit, color=COLOR_Z_CRIT, linestyle="-.", linewidth=1.8,
                   label=f"Z crítico = {crit:.3f}")

    else:  # Cola Derecha
        ax.fill_between(xs, ys, where=(xs > crit),
                        color=COLOR_RECHAZO, alpha=0.4, label=f"Zona de rechazo (α={alpha})")
        ax.axvline(crit, color=COLOR_Z_CRIT, linestyle="-.", linewidth=1.8,
                   label=f"Z crítico = {crit:.3f}")

    # --- Línea del Z calculado ---
    ax.axvline(z_calc, color=COLOR_Z_CALC, linestyle="--", linewidth=2.5,
               label=f"Z calculado = {z_calc:.3f}")

    # --- Curva normal encima de todo ---
    ax.plot(xs, ys, color=COLOR_PRIMARIO, linewidth=2.2)

    # --- Anotación de decisión ---
    decision_texto = "✓ Se RECHAZA H₀" if reject else "✗ No se rechaza H₀"
    color_decision = COLOR_RECHAZO if reject else COLOR_ACEPTACION
    ax.text(
        0.5, 0.92, decision_texto,
        transform=ax.transAxes, ha="center", va="top",
        fontsize=13, fontweight="bold", color=color_decision,
        bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                  edgecolor=color_decision, linewidth=1.5, alpha=0.9)
    )

    # --- Anotación del Z calculado sobre la curva ---
    y_z = stats.norm.pdf(z_calc)
    ax.annotate(
        f"Z = {z_calc:.3f}",
        xy=(z_calc, y_z),
        xytext=(z_calc + (0.5 if z_calc < 0 else -0.5), y_z + 0.05),
        fontsize=9, color=COLOR_Z_CALC, fontweight="bold",
        arrowprops=dict(arrowstyle="->", color=COLOR_Z_CALC, lw=1.2),
        ha="center"
    )

    ax.set_title(
        f"Distribución Normal Estándar — Prueba Z {tipo}  |  α = {alpha}",
        fontsize=13, fontweight="bold", color=COLOR_PRIMARIO, pad=12
    )
    ax.set_xlabel("Z", fontsize=11)
    ax.set_ylabel("Densidad de probabilidad", fontsize=11)
    ax.legend(fontsize=9, loc="upper left", framealpha=0.8)
    ax.grid(alpha=0.2, linestyle="--")
    ax.set_xlim(-4.5, 4.5)
    ax.set_ylim(0, None)

    fig.tight_layout()
    return fig