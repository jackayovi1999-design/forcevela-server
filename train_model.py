import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import joblib

# Datos de ejemplo para entrenar el modelo (luego se actualizará con datos reales)
data = {
    'vela_tamaño': [0.0010, 0.0025, 0.0015, 0.0030, 0.0008],
    'volumen': [1500, 3200, 2100, 4500, 1200],
    'trend_dir': [1, 1, -1, -1, 1],  # 1 = alcista, -1 = bajista
    'fuerza': [2.3, 4.8, 2.9, 5.5, 1.9]
}

df = pd.DataFrame(data)

# Entrenar modelo
X = df[['vela_tamaño', 'volumen', 'trend_dir']]
y = df['fuerza']

model = LinearRegression()
model.fit(X, y)

# Guardar modelo
joblib.dump(model, 'fuerza_velas_modelo.pkl')
print("Modelo entrenado y guardado como 'fuerza_velas_modelo.pkl'")
