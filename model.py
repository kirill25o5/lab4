from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt

data = load_breast_cancer()

X = data.data
y = data.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

clf = RandomForestClassifier(n_estimators=50)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
confusion = confusion_matrix(y_test, y_pred)

correct_examples = X_test[y_test == y_pred]
incorrect_examples = X_test[y_test != y_pred]

plt.figure(figsize=(10, 8))

plt.subplot()
plt.scatter(correct_examples[:, 0], correct_examples[:, 1], c='green', label='Верные предсказания')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Поведение модели')
plt.legend()
plt.tight_layout()
plt.savefig('true.png', dpi=1200)


plt.figure(figsize=(10, 8))

plt.subplot()
plt.scatter(incorrect_examples[:, 0], incorrect_examples[:, 1], c='red', label='Неверные предсказания')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Поведение модели')
plt.legend()

plt.tight_layout()
plt.savefig('false.png', dpi=1200)
