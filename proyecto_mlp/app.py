"""Servicio web (Flask) del MLP. Entrena si no existe el modelo y expone:
  GET  /          -> interfaz web
  GET  /health    -> estado
  GET  /metrics   -> metricas del set de prueba
  POST /predict   -> JSON con las 6 caracteristicas -> prediccion
  POST /train     -> reentrena con el CSV (opcional)
"""
import os, json, subprocess, sys
import joblib, pandas as pd
from flask import Flask, request, jsonify, send_from_directory

BASE = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE, "modelo_mlp.joblib")
FEATURES = ["Tareas", "Examenes", "Asistencia", "Estudio_Horas",
            "Sueno_Horas", "Extracurriculares"]
RANGOS = {"Tareas": (0, 100), "Examenes": (0, 100), "Asistencia": (0, 100),
          "Estudio_Horas": (0, 60), "Sueno_Horas": (0, 16), "Extracurriculares": (0, 1)}

def train():
    subprocess.run([sys.executable, "train.py"], cwd=BASE, check=True)

if not os.path.exists(MODEL_PATH):
    train()
model = joblib.load(MODEL_PATH)
app = Flask(__name__, static_folder=os.path.join(BASE, "static"))

@app.get("/")
def index():
    return send_from_directory(app.static_folder, "index.html")

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/metrics")
def metrics():
    return jsonify(json.load(open(os.path.join(BASE, "metricas.json"))))

@app.post("/predict")
def predict():
    data = request.get_json(silent=True) or {}
    row, errores = {}, []
    for f in FEATURES:
        v = data.get(f)
        if v is None or v == "":
            row[f] = float("nan")          # el imputador usara la mediana
            continue
        try:
            v = float(v)
        except (TypeError, ValueError):
            errores.append(f"{f} debe ser numerico"); continue
        lo, hi = RANGOS[f]
        if not lo <= v <= hi:
            errores.append(f"{f} fuera de rango [{lo}, {hi}]")
        row[f] = v
    if errores:
        return jsonify({"error": errores}), 400
    X = pd.DataFrame([row], columns=FEATURES)
    p = float(model.predict_proba(X)[0, 1])
    return jsonify({"aprobado": int(p >= 0.5), "prediccion": "Aprobado" if p >= 0.5 else "Reprobado",
                    "probabilidad_aprobar": round(p, 4)})

@app.post("/train")
def retrain():
    global model
    train(); model = joblib.load(MODEL_PATH)
    return jsonify({"status": "reentrenado"})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
