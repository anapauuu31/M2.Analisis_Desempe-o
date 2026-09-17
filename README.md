# Predicción de Colocación Laboral con Random Forest (Scikit-learn)

Momento de Retroalimentación — Módulo 2: Análisis y Reporte sobre el desempeño del modelo

## Descripción

Este proyecto implementa un modelo de clasificación con `RandomForestClassifier` de
Scikit-learn para predecir si un estudiante universitario obtiene una colocación laboral
(`Placement`), usando el **College Student Placement Dataset**. El foco de esta entrega es
el uso y configuración correcta del algoritmo mediante el framework, y la evaluación de su
desempeño sobre datos no vistos durante el entrenamiento.

El análisis exploratorio y las decisiones de preprocesamiento se desarrollaron previamente
en la implementación de Random Forest sin framework, y se conservaron aquí para trabajar
bajo las mismas condiciones.

## Dataset

- **Observaciones:** 10,000
- **Variable objetivo:** `Placement` (0 = No Placement, 1 = Placement)
- **Distribución de clases:** 83.41% No Placement / 16.59% Placement (desbalanceado)

## Preprocesamiento

- Se eliminó `College_ID` por ser únicamente un identificador.
- Se codificaron a formato numérico (0/1) las variables de texto `Internship_Experience`
  y `Placement`.
- No se aplicó normalización/estandarización, ya que Random Forest divide por umbrales y
  no requiere que las variables estén en la misma escala.
- Separación estratificada: 70% entrenamiento / 30% prueba, manteniendo en ambos conjuntos
  la misma proporción de clases (~83.41% / 16.59%).

## Configuración del modelo

```python
RandomForestClassifier(
    n_estimators=200,
    max_depth=12,
    min_samples_split=10,
    max_features="sqrt",
    random_state=42,
)
```

| Hiperparámetro      | Valor  | Propósito                                                        |
|----------------------|--------|-------------------------------------------------------------------|
| `n_estimators`       | 200    | Número de árboles del bosque; más árboles → predicción más estable |
| `max_depth`          | 12     | Limita la profundidad máxima de cada árbol                        |
| `min_samples_split`  | 10     | Mínimo de observaciones por nodo para permitir una nueva división |
| `max_features`       | "sqrt" | Introduce diversidad entre árboles al limitar variables por split |
| `random_state`       | 42     | Reproducibilidad de los procesos aleatorios                       |

## Métricas de evaluación

Dado el desbalance de clases, se priorizaron **recall** y **F1-score** por encima de
accuracy, ya que un falso negativo representa a un estudiante que sí obtuvo colocación
pero el modelo no detecta.

## Resultados

| Métrica    | Valor  |
|------------|--------|
| Accuracy   | 99.90% |
| Precision  | 100%   |
| Recall     | 99.40% |
| F1-score   | 99.70% |

**Matriz de confusión:**

|                  | Predicho: No Placement | Predicho: Placement |
|------------------|:-----------------------:|:--------------------:|
| **Real: No Placement** | 2503                | 0                    |
| **Real: Placement**    | 3                   | 495                  |

## Interpretación

El modelo identifica de forma muy consistente ambas clases, sin falsos positivos y con
solo 3 falsos negativos. Sin embargo, un desempeño tan cercano al 100% también es señal
de que un subconjunto de variables académicas (`CGPA`, `Prev_Sem_Result`,
`Communication_Skills` — con los coeficientes de correlación más altos con la variable
objetivo, ~0.32) separan casi perfectamente a los estudiantes colocados de los no
colocados, lo que sugiere que el dataset fue construido de forma sintética con una
relación muy determinística entre esas características y la colocación.

## Conclusión

El uso de Scikit-learn facilitó la construcción, entrenamiento y evaluación del modelo,
así como el control de su comportamiento mediante hiperparámetros. A diferencia de la
entrega anterior (dataset de tarjetas de crédito), el desbalance de clases no representó
aquí una limitación significativa para el desempeño del modelo. La evaluación con recall
y F1-score permitió analizar con más detalle el comportamiento del modelo frente a ese
desbalance.

**Autora:** Ana Paula Moreno Frías 
