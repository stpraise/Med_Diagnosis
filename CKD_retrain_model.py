import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix
import joblib

# Load dataset
df = pd.read_csv("Disease_Symptoms.csv")

# Check for missing values and handle them
if df.isnull().sum().any():
    print("Missing values found. Handling missing data...")
    df.fillna(df.mean(), inplace=True)  # Filling missing values with mean for simplicity

# Split dataset into features (X) and target (y)
X = df.drop(columns=[df.columns[0]])  # Features (all columns except the first one)
y = df.iloc[:, 0]  # Target (first column)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
decision_tree = DecisionTreeClassifier(random_state=42)
decision_tree.fit(X_train, y_train)

# Evaluate model performance
accuracy = decision_tree.score(X_test, y_test)
print(f"Model accuracy: {accuracy:.2f}")

# Confusion matrix and classification report
y_pred = decision_tree.predict(X_test)
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Save the trained model
joblib.dump(decision_tree, 'CKD_Model.joblib')
print("CKD model saved as 'CKD_Model.joblib'")
