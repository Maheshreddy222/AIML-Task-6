# Import libraries
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Step 1: Load dataset
df = pd.read_csv("Iris.csv")

# Step 2: Drop unnecessary column
df = df.drop("Id", axis=1)

# Step 3: Separate features and target
X = df.drop("Species", axis=1)
y = df["Species"]

# Step 4: Apply StandardScaler
scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

# Convert back to DataFrame (optional, for readability)
X_scaled = pd.DataFrame(X_scaled, columns=X.columns)

# Step 5: Check results
print("First 5 rows after normalization:")
print(X_scaled.head())

print("\nMean of features (should be ~0):")
print(X_scaled.mean())

print("\nStandard deviation (should be ~1):")
print(X_scaled.std())