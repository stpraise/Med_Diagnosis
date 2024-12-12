import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import joblib

def retrain_decision_tree_model():
    dataset_path = "Disease_Symptoms.csv"
    model_path = ""

    # Load dataset
    df = pd.read_csv(dataset_path)
    
    # Split dataset into features (X) and target (y)
    X = df.drop(columns=[df.columns[0]])  # All columns except the first
    y = df.iloc[:, 0]  # The first column as target

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Train Decision Tree model
    decision_tree = DecisionTreeClassifier(random_state=42)
    decision_tree.fit(X_train, y_train)

    # Save the model using joblib
    joblib.dump(decision_tree, model_path)
    print(f"Model saved as '{model_path}'.")

if __name__ == "__main__":
    retrain_decision_tree_model()
