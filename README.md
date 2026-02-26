📂 Dataset Information

Dataset Name: Iris Dataset

Target Column:

Species

Classes:

Iris-setosa

Iris-versicolor

Iris-virginica

Features:

SepalLengthCm

SepalWidthCm

PetalLengthCm

PetalWidthCm

1️⃣ Normalize Features
Objective

Scale numerical features for better performance.

Code
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
Explanation

Standardization formula:

Z = (X − mean) / standard deviation

After scaling:

Mean ≈ 0

Standard deviation ≈ 1

2️⃣ Use KNeighborsClassifier
Objective

Train KNN model to classify iris species.

Code
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
Explanation

KNN:

Finds K nearest neighbors

Uses majority voting

Predicts class

3️⃣ Experiment with Different K Values
Objective

Find optimal K value.

Code
for k in range(1, 21):
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    print(k, knn.score(X_test, y_test))
Observation

Small K → Overfitting

Large K → Underfitting

Medium K (3–7) → Best performance

4️⃣ Evaluate Model
Metrics Used

Accuracy

Confusion Matrix

Code
from sklearn.metrics import accuracy_score, confusion_matrix

y_pred = knn.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
Interpretation

Accuracy:

Percentage of correct predictions

Confusion Matrix:

Shows correct and incorrect classifications per class

5️⃣ Visualize Decision Boundaries
Objective

Visualize classification regions.

Code
plt.contourf(xx, yy, Z, alpha=0.3)
plt.scatter(X_train[:, 0], X_train[:, 1])
plt.show()
Explanation

Colored regions represent predicted classes

Points represent training samples

Boundary separates different species
