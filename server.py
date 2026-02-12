from fastapi import FastAPI
import joblib
import pandas as pd

app = FastAPI(title="ForceVela Server")

# Cargar modelo entrenado
model = joblib.load('fuerza_velas_modelo.pkl')

@app.get("/calcular-fuerza")
def calcular_fuerza(vela_tamaño: float, volumen: int, trend_dir: int):
    # Preparar datos para el modelo
    datos = pd.DataFrame({
        'vela_tamaño': [vela_tamaño],
        'volumen': [volumen],
        'trend_dir': [trend_dir]
    })
    
    # Calcular fuerza
    fuerza = model.predict(datos)[0]
    
    return {
        "vela_tamaño": vela_tamaño,
        "volumen": volumen,
        "trend_dir": trend_dir,
        "fuerza_vela": round(fuerza, 2)
    }

# Para correr el servidor: uvicorn server:app --host 0.0.0.0 --port 8000
