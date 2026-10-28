"""Entrenamiento del MLP para predecir Aprobado.
Uso: python train.py [ruta_csv]
Genera: modelo_mlp.joblib, metricas.json, matriz_confusion.png, curva_perdida.png, modelo_web.json
"""
import sys, json
import numpy as np, pandas as pd, joblib
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import (confusion_matrix, ConfusionMatrixDisplay,
                             classification_report, accuracy_score)

SEED = 42
CSV = sys.argv[1] if len(sys.argv) > 1 else "dataset_estudiantes_balanceado.csv"
FEATURES = ["Tareas", "Examenes", "Asistencia", "Estudio_Horas",
            "Sueno_Horas", "Extracurriculares"]
TARGET = "Aprobado"

# 1. Carga
df = pd.read_csv(CSV)
print("Registros:", len(df), "| Balance de clases:\n", df[TARGET].value_counts())
print("Nulos por columna:\n", df.isna().sum())

# NOTA: Calificacion_Final se EXCLUYE: Aprobado = Calificacion_Final > mediana
# (fuga de informacion / data leakage). Las 6 variables si estan disponibles
# antes de conocer la nota final.
X, y = df[FEATURES], df[TARGET]

# 2. Split estratificado 80/20 (set de prueba independiente)
X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.20, stratify=y, random_state=SEED)

# 3. Pipeline: imputacion por MEDIANA -> estandarizacion -> MLP
#    (el pipeline se ajusta solo con train: sin fuga desde el test)
pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
    ("mlp", MLPClassifier(activation="relu", solver="adam", max_iter=2000,
                          early_stopping=True, validation_fraction=0.15,
                          n_iter_no_change=30, random_state=SEED)),
])
grid = {
    "mlp__hidden_layer_sizes": [(8,), (16,), (32, 16), (64, 32), (32, 16, 8)],
    "mlp__alpha": [1e-4, 1e-3, 1e-2],          # regularizacion L2
    "mlp__learning_rate_init": [1e-3, 5e-3],
}
cv = StratifiedKFold(5, shuffle=True, random_state=SEED)
gs = GridSearchCV(pipe, grid, cv=cv, scoring="f1", n_jobs=-1).fit(X_tr, y_tr)
model = gs.best_estimator_
print("Mejores hiperparametros:", gs.best_params_, "| F1 CV: %.4f" % gs.best_score_)

# 4. Evaluacion en test independiente
pred = model.predict(X_te)
cm = confusion_matrix(y_te, pred)
rep = classification_report(y_te, pred, target_names=["Reprobado", "Aprobado"],
                            output_dict=True)
print(classification_report(y_te, pred, target_names=["Reprobado", "Aprobado"]))
print("Matriz de confusion:\n", cm)

ConfusionMatrixDisplay(cm, display_labels=["Reprobado", "Aprobado"]).plot(cmap="Blues")
plt.title("Matriz de confusion - set de prueba (n=%d)" % len(y_te))
plt.savefig("matriz_confusion.png", dpi=140, bbox_inches="tight"); plt.close()

mlp = model.named_steps["mlp"]
plt.plot(mlp.loss_curve_, label="Perdida entrenamiento")
plt.plot(1 - np.array(mlp.validation_scores_), label="1 - accuracy validacion")
plt.xlabel("Epoca"); plt.legend(); plt.title("Curva de aprendizaje")
plt.savefig("curva_perdida.png", dpi=140, bbox_inches="tight"); plt.close()

# 5. Persistencia
joblib.dump(model, "modelo_mlp.joblib")
json.dump({"mejores_parametros": {k: str(v) for k, v in gs.best_params_.items()},
           "f1_cv": gs.best_score_, "accuracy_test": accuracy_score(y_te, pred),
           "matriz_confusion": cm.tolist(), "reporte": rep},
          open("metricas.json", "w"), indent=2)

# 6. Exportar pesos para inferencia en navegador (demo web estatica)
json.dump({
    "features": FEATURES,
    "medianas": model.named_steps["imputer"].statistics_.tolist(),
    "mean": model.named_steps["scaler"].mean_.tolist(),
    "scale": model.named_steps["scaler"].scale_.tolist(),
    "W": [w.tolist() for w in mlp.coefs_],
    "b": [b.tolist() for b in mlp.intercepts_],
}, open("modelo_web.json", "w"))
print("Listo.")
