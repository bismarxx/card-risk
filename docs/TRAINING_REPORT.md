# Informe: Modelo de Riesgo Cardiovascular — Entrenamiento y Puesta en Producción

## 1. Resumen del proyecto

Se desarrolló un sistema completo de predicción de riesgo cardiovascular compuesto por:

| Componente | Archivo | Descripción |
|---|---|---|
| Entrenamiento | `train.py` | Descarga datos, entrena y serializa el modelo |
| Interfaz web | `app.py` | Aplicación Streamlit (UI en inglés) |
| Pruebas | `test_app.py` | 11 pruebas con `pytest` |
| Contenedor | `Dockerfile`, `requirements.txt` | Despliegue en Render / Hugging Face Spaces |
| Modelo serializado | `model/cardio_risk_model.joblib` | Pipeline entrenado (scikit-learn) |
| Metadatos | `model/model_metadata.json` | Métricas e hiperparámetros generados automáticamente |

---

## 2. Dataset

**Fuente:** [UCI Heart Disease Dataset (Cleveland)](https://archive.ics.uci.edu/dataset/45/heart+disease), obtenido de un mirror público en GitHub:
`https://raw.githubusercontent.com/sharmaroshan/Heart-UCI-Dataset/master/heart.csv`

- **Tamaño:** 303 pacientes, 14 columnas originales.
- **Variable objetivo (`target`):** 1 = presencia de enfermedad cardiovascular, 0 = ausencia.
- **Variables clínicas originales:** edad, sexo, tipo de dolor torácico (`cp`), presión arterial en reposo (`trestbps`), colesterol sérico (`chol`), azúcar en sangre en ayunas (`fbs`), resultados del ECG en reposo (`restecg`), frecuencia cardíaca máxima alcanzada (`thalach`), angina inducida por ejercicio (`exang`), depresión del ST (`oldpeak`), pendiente del segmento ST (`slope`), número de vasos principales (`ca`) y resultado de la prueba de talasemia (`thal`).
- **Calidad de datos:** el dataset no presenta valores nulos; se aplicó de todos modos `drop_duplicates()` y `dropna()` como medida de robustez en `train.py`.

### Descarga automática
`train.py` descarga el CSV con `urllib.request.urlretrieve` a `data/heart.csv` la primera vez que se ejecuta; en corridas subsecuentes reutiliza el archivo local para evitar dependencias de red innecesarias.

---

## 3. Ingeniería de características

Sobre las 13 variables predictoras originales se construyeron 4 características derivadas, clínicamente motivadas:

| Feature nueva | Fórmula | Justificación |
|---|---|---|
| `bp_chol_ratio` | `trestbps / chol` | Relación entre presión arterial y colesterol |
| `age_group` | Bins: `<40`, `40–49`, `50–59`, `60+` | Categorización de riesgo por rango etario |
| `max_hr_expected` | `220 - age` | Frecuencia cardíaca máxima teórica (fórmula estándar) |
| `hr_reserve_deficit` | `max_hr_expected - thalach` | Déficit respecto a la frecuencia cardíaca teórica; una reserva baja puede indicar menor capacidad cardiovascular |

**Total de variables usadas por el modelo:** 8 numéricas + 9 categóricas (incluyendo `age_group`).

### Preprocesamiento
- Variables numéricas → `StandardScaler`.
- Variables categóricas → `OneHotEncoder(handle_unknown="ignore")`.
- Todo encapsulado en un `ColumnTransformer` dentro de un `Pipeline` de scikit-learn, de forma que el objeto serializado incluye el preprocesamiento completo (no se requiere replicarlo manualmente al hacer inferencia, salvo la construcción de las features derivadas, que se reutiliza desde `app.py` vía la función `build_feature_row`).

---

## 4. Optimización de hiperparámetros

Se evaluaron 3 modelos candidatos mediante `GridSearchCV` con validación cruzada estratificada de 5 folds (`StratifiedKFold`), optimizando **ROC-AUC** como métrica principal (apropiada para un problema de clasificación binaria con clases balanceadas y donde interesa el ranking de probabilidad, no solo la clase final).

| Modelo | Grilla de hiperparámetros explorada | Mejor ROC-AUC (CV) | Mejores hiperparámetros |
|---|---|---|---|
| Regresión Logística | `C ∈ {0.01, 0.1, 1, 10}`, `solver=lbfgs` | **0.9077** | `C=0.1` |
| Random Forest | `n_estimators ∈ {100,200,300}`, `max_depth ∈ {3,5,8,None}`, `min_samples_leaf ∈ {1,2,4}` | 0.9062 | `max_depth=3, min_samples_leaf=4, n_estimators=100` |
| Gradient Boosting | `n_estimators ∈ {100,200}`, `learning_rate ∈ {0.01,0.05,0.1}`, `max_depth ∈ {2,3,4}` | 0.8905 | `learning_rate=0.01, max_depth=2, n_estimators=200` |

**Modelo seleccionado:** Regresión Logística (`C=0.1`), por obtener el mejor ROC-AUC en validación cruzada y ser el modelo más simple e interpretable entre los de desempeño equivalente (criterio de parsimonia).

---

## 5. Evaluación final (conjunto de prueba, 20% hold-out, 61 pacientes)

| Métrica | Valor |
|---|---|
| Accuracy | 0.8525 |
| Precision | 0.8333 |
| Recall (sensibilidad) | 0.9091 |
| F1-score | 0.8696 |
| ROC-AUC | 0.8972 |

**Interpretación clínica:** el recall de 0.91 indica que el modelo identifica correctamente al ~91% de los pacientes que efectivamente presentan riesgo cardiovascular en el conjunto de prueba, lo cual es deseable en un contexto de cribado (screening), donde es preferible minimizar los falsos negativos. La precisión de 0.83 indica que, de los pacientes marcados como riesgo alto, el 83% efectivamente lo eran.

> Nota: el dataset es pequeño (303 pacientes), por lo que estas métricas deben interpretarse como una prueba de concepto y no como validación clínica. Antes de cualquier uso real se requeriría validación externa con cohortes más grandes y diversas.

---

## 6. Artefactos generados

- `model/cardio_risk_model.joblib` — pipeline completo (preprocesamiento + modelo) serializado con `joblib`.
- `model/model_metadata.json` — métricas, hiperparámetros y configuración del entrenamiento, generado automáticamente por `train.py` para trazabilidad y reproducibilidad.

---

## 7. Interfaz web (Streamlit)

`app.py` implementa una aplicación de una sola página, íntegramente en inglés (etiquetas, botones y resultados), que:

1. Carga el modelo serializado (`@st.cache_resource`, se carga una sola vez por sesión).
2. Recolecta las 13 variables clínicas mediante widgets (`number_input`, `selectbox`).
3. Reconstruye las mismas características derivadas usadas en el entrenamiento (`build_feature_row`), garantizando consistencia train/inferencia.
4. Al presionar **"Predict Cardiovascular Risk"**, muestra la probabilidad estimada, una etiqueta de riesgo (alto/bajo) y una barra de progreso, junto con un descargo de responsabilidad médica.

---

## 8. Pruebas automatizadas

`test_app.py` contiene **11 pruebas** con `pytest`, organizadas en 3 clases:

1. **`TestInputValidation`** (3 pruebas + 3 parametrizadas): valida el esquema de las features construidas y los rangos/códigos permitidos para los campos clínicos y categóricos.
2. **`TestModelPrediction`** (4 pruebas): valida que el modelo serializado exista, que las predicciones sean probabilidades válidas en `[0,1]`, que la etiqueta sea binaria y que la inferencia sea determinística.
3. **`TestStreamlitApp`** (2 pruebas): usa `streamlit.testing.v1.AppTest` para simular una sesión de usuario real —carga de la app sin excepciones y flujo completo de click en el botón de predicción, verificando que se devuelva un resultado—, cumpliendo el rol de una prueba "extremo a extremo" equivalente a validar la respuesta de un endpoint HTTP.

Resultado de la última ejecución: **11 passed**.

Ejecución:
```bash
pip install -r requirements.txt
python train.py          # genera model/cardio_risk_model.joblib
pytest test_app.py -v
```

---

## 9. Contenedorización

`Dockerfile`:
- Basado en `python:3.11-slim`.
- Instala dependencias desde `requirements.txt`.
- Ejecuta `train.py` en tiempo de build, de modo que la imagen queda autocontenida (incluye el modelo ya entrenado).
- Expone el puerto configurado en la variable de entorno `PORT` (por defecto `8501`, el puerto estándar de Streamlit).
- Incluye un `HEALTHCHECK` contra el endpoint interno de salud de Streamlit (`/_stcore/health`).
- `CMD`: `streamlit run app.py --server.port=${PORT} --server.address=0.0.0.0 --server.headless=true`.

> **Nota:** en el entorno donde se generó este proyecto no hay acceso a Docker Hub ni al daemon de Docker, por lo que el `Dockerfile` no pudo construirse ni ejecutarse dentro de este sandbox. Se recomienda validar el build (`docker build -t cardio-risk-app .` y `docker run -p 8501:8501 cardio-risk-app`) en un entorno local antes de desplegar a producción.

---

## 10. Pasos manuales de despliegue

### Opción A — Render (usando el Dockerfile)

1. Subir el proyecto completo a un repositorio de GitHub (incluyendo `Dockerfile`, `requirements.txt`, `train.py`, `app.py`, `test_app.py`).
2. En [Render](https://render.com), crear un nuevo **Web Service**.
3. Conectar el repositorio de GitHub.
4. En **Environment**, seleccionar **Docker** (Render detecta automáticamente el `Dockerfile`).
5. Render inyecta la variable `PORT` automáticamente; el `Dockerfile` ya está preparado para leerla (`--server.port=${PORT}`), no se requiere configuración adicional.
6. Definir el **Instance Type** (el plan gratuito es suficiente para una demo).
7. Desplegar. Render construirá la imagen (ejecutando `train.py` durante el build) y levantará el contenedor.
8. Verificar en la URL pública (`https://<nombre-servicio>.onrender.com`) que la app cargue y realice predicciones.

### Opción B — Hugging Face Spaces (usando SDK Docker)

1. Crear un nuevo Space en [huggingface.co/new-space](https://huggingface.co/new-space).
2. Elegir **Docker** como SDK (no "Streamlit" nativo, ya que se usa el `Dockerfile` propio).
3. Subir (o hacer `git push`) los archivos del proyecto al repositorio del Space: `app.py`, `train.py`, `test_app.py`, `requirements.txt`, `Dockerfile`, `.dockerignore`.
4. Hugging Face Spaces expone el contenedor en el puerto `7860` por convención; como el `Dockerfile` lee `PORT` desde el entorno, basta con definir la variable `PORT=7860` en la configuración del Space (Settings → Variables), o alternativamente añadir `ENV PORT=7860` antes del build si se desea fijarlo explícitamente para este destino.
5. El Space construirá la imagen automáticamente al detectar el `Dockerfile` y expondrá la aplicación en `https://huggingface.co/spaces/<usuario>/<nombre-space>`.

### Opción C — Despliegue directo sin Docker (alternativa simple)

Para plataformas con soporte nativo de Streamlit (por ejemplo Render como "Web Service" tipo *Python*, o Streamlit Community Cloud):

1. Subir el repositorio a GitHub.
2. Configurar el comando de arranque: `streamlit run app.py --server.port=$PORT --server.address=0.0.0.0`.
3. Configurar el comando de build/pre-arranque para entrenar el modelo si no está versionado: `pip install -r requirements.txt && python train.py`.
4. Desplegar.

---

## 11. Limitaciones y trabajo futuro

- El dataset (303 registros) es pequeño para uso clínico real; se recomienda reentrenar con datasets más grandes (p. ej. Kaggle "Cardiovascular Disease dataset", 70.000 registros) si se desea llevar el modelo a un contexto de producción real.
- No se realizó calibración de probabilidades (`CalibratedClassifierCV`); podría mejorar la interpretabilidad de las probabilidades reportadas en la interfaz.
- El modelo y la app son de carácter educativo/demostrativo y **no constituyen una herramienta de diagnóstico médico**.
