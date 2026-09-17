# Importar Random Forest y las métricas para evaluar el modelo
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)
import matplotlib.pyplot as plt
import seaborn as sns


# Crear el modelo Random Forest
# Se utilizan 200 árboles y se limita su profundidad para
# controlar la complejidad del modelo
model = RandomForestClassifier(
    n_estimators=200,
    max_depth=12,
    min_samples_split=10,
    max_features="sqrt",
    random_state=42
)

# Entrenar el modelo utilizando los datos balanceados
model.fit(X_train, y_train)


# Realizar las predicciones utilizando el conjunto de prueba original
# El conjunto de prueba no se balancea para mantener su distribución real
y_pred = model.predict(X_test)


# Crear la matriz de confusión para observar los aciertos
# y errores de clasificación en cada clase
matrix = confusion_matrix(y_test, y_pred)


# Calcular las métricas de evaluación
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)


# Mostrar los resultados obtenidos
print("Matriz de confusión:")
print(matrix)

print("Accuracy:", (accuracy))
print("Precision:", (precision))
print("Recall:", (recall))
print("F1-score:",(f1))

# Mostrar la matriz de confusión de forma gráfica
plt.figure(figsize=(6, 5))

sns.heatmap(
    matrix,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["No Default", "Default"],
    yticklabels=["No Default", "Default"]
)

plt.title("Matriz de confusión")
plt.xlabel("Predicción")
plt.ylabel("Valor real")

plt.show()