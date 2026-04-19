import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import scipy.stats as stats

def histograma(data):
    fig, ax = plt.subplots()
    sns.histplot(data, kde=True, ax=ax)
    return fig

def boxplot(data):
    fig, ax = plt.subplots()
    sns.boxplot(x=data, ax=ax)
    return fig

def curva_z(z, crit, tipo):
    fig, ax = plt.subplots(figsize=(8,4))
    x = np.linspace(-4,4,1000)
    y = stats.norm.pdf(x)

    ax.plot(x,y)

    if tipo == "Bilateral":
        ax.fill_between(x,y,where=(x>crit),alpha=.5)
        ax.fill_between(x,y,where=(x<-crit),alpha=.5)

    elif tipo == "Izquierda":
        ax.fill_between(x,y,where=(x<crit),alpha=.5)

    else:
        ax.fill_between(x,y,where=(x>crit),alpha=.5)

    ax.axvline(z, linestyle="--")
    return fig