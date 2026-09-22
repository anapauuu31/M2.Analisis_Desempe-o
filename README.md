# Predicción de Colocación Laboral con Random Forest (Scikit-learn)

**Momento de Retroalimentación — Módulo 2: Análisis y Reporte sobre el desempeño del modelo**

## Descripción

Este proyecto implementa un modelo de clasificación con `RandomForestClassifier` de Scikit-learn para predecir si un estudiante universitario obtiene una colocación laboral (`Placement`), usando el College Student Placement Dataset.

El foco de esta entrega es un análisis completo del desempeño del modelo: separación en conjuntos de entrenamiento, validación y prueba; diagnóstico de bias y varianza; aplicación de regularización; y evaluación final sobre datos nunca antes vistos por el modelo.

## Dataset

- Observaciones: 10,000
- Variable objetivo: `Placement` (0 = No Placement, 1 = Placement)
- Distribución de clases: 83.41% No Placement / 16.59% Placement (desbalanceado)

## Preprocesamiento

- Se eliminó `College_ID` por ser únicamente un identificador.
- Se codificaron a formato numérico (0/1) las variables de texto `Internship_Experience` y `Placement`.
- No se aplicó normalización ni estandarización, ya que Random Forest divide por umbrales y no requiere que las variables estén en la misma escala.

## Separación de datos: Train / Validation / Test

A diferencia de un split simple de entrenamiento y prueba, aquí los datos se dividieron en **tres** conjuntos estratificados por clase:

| Conjunto | Proporción | Registros | % Placement |
|---|---|---|---|
| Entrenamiento | 60% | 5,999 | 16.59% |
| Validación | 20% | 1,999 | 16.56% |
| Prueba | 20% | 2,002 | 16.63% |

- **Entrenamiento**: se usa para ajustar los modelos.
- **Validación**: se usa para diagnosticar bias y varianza, y para comparar el modelo antes y después de regularizar, sin tocar el conjunto de prueba.
- **Prueba**: se reserva por completo y se evalúa una única vez, al final, como estimación no sesgada de la capacidad de generalización del modelo.

## Diagnóstico de bias, varianza y nivel de ajuste

Se entrenó primero un modelo **baseline sin regularizar** (sin límite de profundidad, `min_samples_split=2`, usando todas las variables en cada división), evaluado en entrenamiento y validación:

| Conjunto | Accuracy | F1-score |
|---|---|---|
| Entrenamiento | 1.0 | 1.0 |
| Validación | 1.0 | 1.0 |

Brecha train-validation: **0.0**

- **Bias: bajo** (accuracy/F1 en entrenamiento = 1.0)
- **Varianza: baja** (brecha train-validation = 0.0)
- **Nivel de ajuste: fit** (buen ajuste, sin señales de underfitting ni overfitting)

### Verificación de fuga de datos

Un resultado perfecto en ambos conjuntos es inusual, así que antes de aceptar el diagnóstico se revisó la importancia de variables del modelo baseline, para descartar que alguna columna funcionara como copia directa de la variable objetivo:

| Variable | Importancia |
|---|---|
| CGPA | 33% |
| Projects_Completed | 29% |
| IQ | 19% |
| Communication_Skills | 19% |
| Prev_Sem_Result, Academic_Performance, Internship_Experience, Extra_Curricular_Score | ~0% |

La importancia está repartida entre cuatro variables, sin que ninguna domine de forma aplastante, lo que descarta fuga de datos. El resultado perfecto se atribuye a que el dataset, al ser de tipo sintético, fue construido con una relación casi determinística entre esas cuatro variables y `Placement`. `Prev_Sem_Result` no aporta importancia adicional porque está correlacionada en 0.98 con `CGPA`.

## Aplicación de regularización

A partir del diagnóstico, se entrenó un segundo modelo aplicando regularización:

```python
model = RandomForestClassifier(
    n_estimators=200,
    max_depth=12,
    min_samples_split=10,
    max_features="sqrt",
    random_state=42,
)
```

| Hiperparámetro | Valor | Propósito |
|---|---|---|
| `n_estimators` | 200 | Número de árboles del bosque; más árboles → predicción más estable |
| `max_depth` | 12 | Limita la profundidad máxima de cada árbol |
| `min_samples_split` | 10 | Mínimo de observaciones por nodo para permitir una nueva división |
| `max_features` | `"sqrt"` | Introduce diversidad entre árboles al limitar variables por división |
| `random_state` | 42 | Reproducibilidad de los procesos aleatorios |

**Efecto de la regularización (conjunto de validación):**

| Modelo | Accuracy | F1-score | Brecha train-val |
|---|---|---|---|
| Baseline (sin regularizar) | 1.0 | 1.0 | 0.0 |
| Regularizado | 1.0 | 1.0 | 0.0 |

La regularización no modificó el desempeño, porque no había varianza que reducir: el modelo baseline ya generalizaba perfectamente. Aun así, se conservó el modelo regularizado como final por buena práctica: es más robusto ante datos que en el futuro no sean tan perfectamente separables como los de este dataset.

## Métricas de evaluación

Dado el desbalance de clases, se priorizaron **recall** y **F1-score** por encima de accuracy, ya que un falso negativo representa a un estudiante que sí obtuvo colocación pero el modelo no detecta.

## Resultados finales (conjunto de prueba)

Evaluado una sola vez, sobre datos nunca antes vistos por el modelo:

| Métrica | Valor |
|---|---|
| Accuracy | 99.90% |
| Precision | 100% |
| Recall | 99.40% |
| F1-score | 99.70% |

**Matriz de confusión:**

| | Predicho: No Placement | Predicho: Placement |
|---|---|---|
| **Real: No Placement** | 1,669 | 0 |
| **Real: Placement** | 2 | 331 |

## Interpretación

El modelo identifica de forma muy consistente ambas clases, sin falsos positivos y con solo 2 falsos negativos. Que el desempeño en prueba (99.90%) sea prácticamente igual al obtenido en entrenamiento y validación (1.0) confirma que el diagnóstico de bias bajo y varianza baja fue correcto: el modelo generaliza bien a datos nuevos y no estaba sobreajustado.

## Conclusión

El uso de Scikit-learn facilitó la construcción, el entrenamiento y la evaluación del modelo, así como el control de su comportamiento mediante hiperparámetros. El proceso de separación en train/validation/test, diagnóstico de bias-varianza y ajuste mediante regularización permitió construir un modelo con una estimación confiable de su capacidad de generalización, prácticamente igual de sólida en datos nuevos que en los de entrenamiento.

A diferencia de la entrega anterior (dataset de tarjetas de crédito), aquí el desbalance de clases no representó una limitación significativa para el desempeño del modelo.

---

**Autora:** Ana Paula Moreno Frías | A01751886
