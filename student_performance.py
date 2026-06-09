import pandas as pd

# Dataset Load
df = pd.read_csv("student_data.csv")

# Dataset Information
print("First 5 Rows:")
print(df.head())

print("\nShape:")
print(df.shape)

print("\nColumns:")
print(df.columns)

print("\nDataset Info:")
print(df.info())

print("\nMissing Values:")
print(df.isnull().sum())

# Features and Target
X = df.drop("G3", axis=1)
y = df["G3"]

print("\nFeatures Shape:", X.shape)
print("Target Shape:", y.shape)

# One-Hot Encoding
X = pd.get_dummies(X, drop_first=True)

print("\nNew Shape After Encoding:", X.shape)

# Train-Test Split
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.25,
    random_state=42
)

print("\nTraining Data:", X_train.shape)
print("Testing Data:", X_test.shape)

# Linear Regression Model
from sklearn.linear_model import LinearRegression

model = LinearRegression()

model.fit(X_train, y_train)

print("\nModel trained successfully!")

# Predictions
predictions = model.predict(X_test)

# Evaluation
from sklearn.metrics import r2_score

score = r2_score(y_test, predictions)

print("\nR2 Score:", score)
