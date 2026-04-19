import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import scipy.stats as stats

def histograma_kde(x):

    fig, ax = plt.subplots(figsize=(12,5))

    sns.histplot(
        x,
        kde=True,
        color="#F77F00",
        ax=ax
    )

    ax.grid(alpha=0.2)
    ax.set_title("Histograma + KDE")

    return fig


def boxplot_chart(x):

    fig, ax = plt.subplots(figsize=(12,3))

    sns.boxplot(
        x=x,
        color="#FCBF49",
        ax=ax
    )

    ax.grid(alpha=0.2)
    ax.set_title("Boxplot")

    return fig


def curva_z(z, crit, tipo):

    fig, ax = plt.subplots(figsize=(12,5))

    xs = np.linspace(-4,4,1000)
    ys = stats.norm.pdf(xs)

    ax.plot(xs, ys, color="#003049", linewidth=2)

    if tipo == "Bilateral":
        ax.fill_between(xs, ys, where=(xs>crit), alpha=0.5, color="#D62828")
        ax.fill_between(xs, ys, where=(xs<-crit), alpha=0.5, color="#D62828")

    elif tipo == "Cola Izquierda":
        ax.fill_between(xs, ys, where=(xs<crit), alpha=0.5, color="#D62828")

    else:
        ax.fill_between(xs, ys, where=(xs>crit), alpha=0.5, color="#D62828")

    ax.axvline(z, linestyle="--", color="#F77F00", linewidth=2)

    ax.grid(alpha=0.2)

    return fig