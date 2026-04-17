import pandas as pd
import numpy as np

np.random.seed(42)

n = 120
media_real = 52.5
sigma_real = 10

datos = np.random.normal(loc=media_real, scale=sigma_real, size=n)
datos = np.append(datos, [95.0, 5.0, 98.5])

df = pd.DataFrame({
    "id_muestra": range(1, len(datos)+1),
    "medicion": np.round(datos, 2),
    "grupo": "Lote_A"
})

df.to_csv("datos_test_estadistico.csv", index=False)
print("CSV generado")