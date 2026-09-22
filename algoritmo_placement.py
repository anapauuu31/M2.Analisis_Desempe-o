from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix


def evaluar(modelo, X, y, nombre):
    # Calcula las cuatro métricas en una sola función para no repetir el código en cada evaluación
    y_pred = modelo.predict(X)
    metrics = {
        "conjunto": nombre,
        "accuracy": accuracy_score(y, y_pred),
        "precision": precision_score(y, y_pred),
        "recall": recall_score(y, y_pred),
        "f1": f1_score(y, y_pred),
    }
    print(metrics)
    return metrics


# Modelo sin restricciones, entrenado primero para observar su comportamiento
# antes de aplicar regularización
baseline_model = RandomForestClassifier(
    n_estimators=200,
    max_depth=None,
    min_samples_split=2,
    max_features=None,
    random_state=42,
)

baseline_model.fit(X_train, y_train)

baseline_train = evaluar(baseline_model, X_train, y_train, "train")
baseline_val = evaluar(baseline_model, X_val, y_val, "validation")

# Modelo con regularización aplicada (max_depth, min_samples_split, max_features);
# este es el que se conserva como modelo final
model = RandomForestClassifier(
    n_estimators=200,
    max_depth=12,
    min_samples_split=10,
    max_features="sqrt",
    random_state=42,
)

model.fit(X_train, y_train)

reg_train = evaluar(model, X_train, y_train, "train")
reg_val = evaluar(model, X_val, y_val, "validation")

# El conjunto de prueba se utiliza únicamente aquí, una sola vez, ya con el modelo final definido
y_pred = model.predict(X_test)
matrix = confusion_matrix(y_test, y_pred)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)