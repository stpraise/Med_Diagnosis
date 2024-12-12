import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import joblib

# Assuming you have the dataset available as "Disease_Symptoms.csv"
df = pd.read_csv("Disease_Symptoms.csv")

# Split dataset into features (X) and target (y)
X = df.drop(columns=[df.columns[0]])
y = df.iloc[:, 0]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
decision_tree = DecisionTreeClassifier(random_state=42)
decision_tree.fit(X_train, y_train)

# Save the trained model
joblib.dump(decision_tree, 'HeartDisease.joblib')
print("HeartDisease saved as 'HeartDisease.joblib'")
