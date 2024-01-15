import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import pickle

# Charger les données
from sklearn.datasets import load_breast_cancer
data = load_breast_cancer()

# Convertir en DataFrame pour une meilleure manipulation
X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target)

# Diviser les données en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Créer et entraîner le modèle de régression logistique
model = LogisticRegression(max_iter=10000)
model.fit(X_train, y_train)

# Évaluer le modèle
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Précision du modèle : {accuracy:.2f}")

# Sauvegarder le modèle entraîné
with open('model.pkl', 'wb') as file:
    pickle.dump(model, file)

print("Modèle entraîné et sauvegardé.")