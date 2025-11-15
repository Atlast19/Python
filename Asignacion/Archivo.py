# hf_assignment.py
# Requisitos: pandas, numpy, scikit-learn, matplotlib, seaborn
# Instalación (si necesitas): pip install pandas numpy scikit-learn matplotlib seaborn

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import joblib

# ------------- 1) Cargar datos -------------
CSV_NAME = "heart_failure_clinical_records_dataset.csv"
if not os.path.exists(CSV_NAME):
    raise FileNotFoundError(f"No encontré el archivo '{CSV_NAME}' en la carpeta actual. "
                            "Coloca ahí el CSV y vuelve a ejecutar.")

df = pd.read_csv(CSV_NAME)
print("Datos cargados. Dimensiones:", df.shape)
print("Columnas:", df.columns.tolist())
print("\nPrimeras filas:\n", df.head())

# ------------- 2) Revisión y preprocesado simple -------------
print("\nDescripción rápida:")
print(df.describe(include='all'))

# Revisar nulos
print("\nNulos por columna:\n", df.isnull().sum())

# Columnas que esperamos (dataset estándar)
expected_cols = ['age', 'anaemia', 'creatinine_phosphokinase', 'diabetes',
                 'ejection_fraction', 'high_blood_pressure', 'platelets',
                 'serum_creatinine', 'serum_sodium', 'sex', 'smoking', 'time', 'DEATH_EVENT']

# Si el CSV tiene columnas en mayúsculas o diferentes, ajustar aquí.
# Para este script asumimos que las columnas existen según expected_cols.
missing = [c for c in expected_cols if c not in df.columns]
if missing:
    print("Advertencia: faltan columnas esperadas en el CSV:", missing)
else:
    print("Todas las columnas esperadas están presentes.")

# Comprobar tipos
print("\nTipos de datos:\n", df.dtypes)

# Si hubiese variables categóricas representadas por 0/1 ya están listas.
# Separar X, y
TARGET = 'DEATH_EVENT'
X = df.drop(columns=[TARGET])
y = df[TARGET]

# ------------- 3) Train / Test split-------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y)

print(f"\nTrain: {X_train.shape}, Test: {X_test.shape}")

# ------------- 4) Entrenar varios clasificadores y elegir el mejor -------------
# Pipelines con escalado donde convenga
models = {
    "LogisticRegression": Pipeline([('scaler', StandardScaler()), ('clf', LogisticRegression(max_iter=1000))]),
    "KNN": Pipeline([('scaler', StandardScaler()), ('clf', KNeighborsClassifier())]),
    "SVM": Pipeline([('scaler', StandardScaler()), ('clf', SVC(probability=True))]),
    "RandomForest": Pipeline([('clf', RandomForestClassifier(n_estimators=200, random_state=42))]),
    "GradientBoosting": Pipeline([('clf', GradientBoostingClassifier(n_estimators=200, random_state=42))])
}

results = []
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for name, model in models.items():
    cv_f1 = cross_val_score(model, X_train, y_train, cv=skf, scoring='f1')
    cv_acc = cross_val_score(model, X_train, y_train, cv=skf, scoring='accuracy')
    results.append({
        'model': name,
        'cv_f1_mean': cv_f1.mean(),
        'cv_f1_std': cv_f1.std(),
        'cv_acc_mean': cv_acc.mean()
    })
    print(f"{name}: CV F1 = {cv_f1.mean():.4f} ± {cv_f1.std():.4f} | CV Acc = {cv_acc.mean():.4f}")

results_df = pd.DataFrame(results).sort_values(by='cv_f1_mean', ascending=False)
print("\nRanking por F1 (CV):\n", results_df)

# Seleccionar mejor por F1
best_model_name = results_df.iloc[0]['model']
best_model = models[best_model_name]
print(f"\nMejor modelo seleccionado: {best_model_name}")

# Entrenar en todo el train
best_model.fit(X_train, y_train)

# ------------- 5) Evaluar en Test y generar matriz de confusión -------------
y_pred = best_model.predict(X_test)
print("\nResultados en Test:")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print("F1:", f1_score(y_test, y_pred))
print("\nClassification report:\n", classification_report(y_test, y_pred))

# Matriz de confusión
cm = confusion_matrix(y_test, y_pred)
print("Matriz de confusión:\n", cm)

# Graficar y guardar matriz de confusión
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=[0,1], yticklabels=[0,1])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title(f'Confusion Matrix - {best_model_name}')
plt.tight_layout()
cm_png = f'confusion_matrix_{best_model_name}.png'
plt.savefig(cm_png)
print(f"Matriz de confusión guardada en {cm_png}")
plt.show()

# Guardar modelo entrenado
joblib.dump(best_model, f'{best_model_name}_best_model.joblib')
print(f"Modelo guardado en {best_model_name}_best_model.joblib")

# ------------- 6) Agregar columna con predicción al dataset original (clasificado) -------------
# Predecir para todo el dataset (o usar X_test index si prefieres). Vamos a predecir sobre todo X:
df_with_pred = df.copy()
df_with_pred['predicted_DEATH_EVENT'] = best_model.predict(X)
# También probabilidad
if hasattr(best_model, "predict_proba"):
    df_with_pred['predicted_proba_death'] = best_model.predict_proba(X)[:, 1]
else:
    # algunos modelos (SVM sin probability) no tienen predict_proba; si es el caso, usar decision_function
    try:
        scores = best_model.decision_function(X)
        df_with_pred['predicted_proba_death'] = (scores - scores.min()) / (scores.max() - scores.min())
    except Exception:
        df_with_pred['predicted_proba_death'] = np.nan

out_all = 'data_with_predictions.csv'
df_with_pred.to_csv(out_all, index=False)
print(f"Data table completo (con predicciones) guardado en: {out_all}")

# ------------- 7) Filtrar en 2 data tables: positivos y negativos -------------
positive_df = df_with_pred[df_with_pred['predicted_DEATH_EVENT'] == 1].copy()
negative_df = df_with_pred[df_with_pred['predicted_DEATH_EVENT'] == 0].copy()

pos_file = 'predicted_death_positive.csv'
neg_file = 'predicted_death_negative.csv'
positive_df.to_csv(pos_file, index=False)
negative_df.to_csv(neg_file, index=False)
print(f"Casos positivos guardados en: {pos_file} ({positive_df.shape[0]} filas)")
print(f"Casos negativos guardados en: {neg_file} ({negative_df.shape[0]} filas)")

# ------------- 8) Visualización de distribución en función a la variable Anaemia (anaemia) -------------
# Crear conteos por anaemia y por DEATH_EVENT predicho
plt.figure(figsize=(7,5))
sns.countplot(x='anaemia', hue='predicted_DEATH_EVENT', data=df_with_pred)
plt.title('Distribución por Anemia (0=no,1=yes) vs Predicción de DEATH_EVENT')
plt.xlabel('Anaemia')
plt.ylabel('Cuenta')
plt.legend(title='Predicción DEATH_EVENT', labels=['Negativo','Positivo'])
anaemia_png = 'anaemia_distribution.png'
plt.tight_layout()
plt.savefig(anaemia_png)
print(f"Gráfica de distribución por anaemia guardada en: {anaemia_png}")
plt.show()

# También mostrar proporciones
anaemia_ct = pd.crosstab(df_with_pred['anaemia'], df_with_pred['predicted_DEATH_EVENT'], normalize='index') * 100
print("\nPorcentaje de predicciones por anaemia (por fila):\n", anaemia_ct.round(2))

# ------------- 9) Visualización de dispersión (x = age, y = time) -------------
plt.figure(figsize=(8,6))
# colorear por DEATH_EVENT real y por marca de predicción (usar shape o alpha)
sns.scatterplot(x='age', y='time', hue='DEATH_EVENT', style='predicted_DEATH_EVENT', data=df_with_pred, s=80)
plt.title('Dispersión: age (x) vs time (y) - coloreado por DEATH_EVENT real')
plt.xlabel('Age (años)')
plt.ylabel('Time (días)')
scatter_png = 'scatter_age_time.png'
plt.tight_layout()
plt.savefig(scatter_png)
print(f"Scatter guardado en: {scatter_png}")
plt.show()

# ------------- 10) Interpretación sugerida del scatter (impresa) -------------
print("\n--- INTERPRETACIÓN SUGERIDA DEL GRÁFICO age vs time ---\n")
print("""
1) Ejes:
   - x = edad del paciente (age)
   - y = tiempo de seguimiento en días (time)
   - color = DEATH_EVENT real (0=no, 1=si)
   - estilo del marcador = predicción (0/1) para ver concordancia/discordancia

2) Qué buscar:
   - Agrupaciones de puntos: ¿los pacientes más viejos (x mayor) tienden a tener times más cortos o más largos?
   - Concentraciones de muertes reales (color 1): si aparecen mayormente en edades altas y/o en tiempos cortos, podría indicar que pacientes viejos tienden a morir más rápido.
   - Casos donde la predicción difiere de la realidad (markers distintos del color): son errores del modelo (falsos positivos / falsos negativos).
   - Outliers: pacientes muy jóvenes con muerte temprana o pacientes viejos con tiempos muy largos.

3) Posibles conclusiones (dependen del gráfico real):
   - Si se observa que los puntos con DEATH_EVENT=1 se concentran en edades altas y tiempos pequeños, podríamos inferir que la edad es un factor de riesgo ligado a muerte más temprana en este set.
   - Si no hay un patrón claro, puede que la relación entre age y time no sea lineal ni fuerte por sí sola y se necesite combinar con otras variables (p. ej. ejection_fraction, serum_creatinine).
   - Si hay muchos falsos negativos entre pacientes de cierta edad, habría que revisar el modelo (features, balanceo, hiperparámetros).

4) Recomendación:
   - Calcular estadísticas adicionales (p. ej. medias de time por grupos de edad, test de correlación) y analizar variables conjuntas (pairplots, modelos explicables como SHAP).
""")

# Finalmente, guardar un pequeño reporte resumen
report = {
    'chosen_model': best_model_name,
    'test_accuracy': accuracy_score(y_test, y_pred),
    'test_f1': f1_score(y_test, y_pred),
    'confusion_matrix': cm.tolist()
}
report_df = pd.DataFrame([report])
report_df.to_csv('model_report_summary.csv', index=False)
print("Resumen del modelo guardado en model_report_summary.csv")

print("\n--- Script terminado ---")