# Import libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Step 1: Load dataset
df = pd.read_csv("Iris.csv")

# Step 2: Drop unnecessary column
df = df.drop("Id", axis=1)

# Step 3: Separate features and target
X = df.drop("Species", axis=1)
y = df["Species"]

# Step 4: Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Step 5: Normalize features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Step 6: Create KNN model
knn = KNeighborsClassifier(n_neighbors=5)

# Step 7: Train model
knn.fit(X_train, y_train)

# Step 8: Make predictions
y_pred = knn.predict(X_test)

# Step 9: Check accuracy
accuracy = accuracy_score(y_test, y_pred)

print("KNN Accuracy:", accuracy)