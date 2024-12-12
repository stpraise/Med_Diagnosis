import pandas as pd
import joblib
import os


def load_model(model_path: str):
    """
    Loads a saved Decision Tree model using joblib.
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file '{model_path}' does not exist.")
    return joblib.load(model_path)


def get_feature_template(dataset_path: str):
    """
    Creates a template of features based on the training dataset.
    """
    try:
        df = pd.read_csv(dataset_path)
        feature_names = df.columns[1:]  # Exclude the target column
        return {feature: 0 for feature in feature_names}
    except FileNotFoundError:
        raise FileNotFoundError(f"Dataset '{dataset_path}' not found.")


def predict_disease(symptoms, dataset_path="Disease_Symptoms.csv"):
    """
    Predicts the disease based on symptoms using a pre-trained model.
    """
    model_path = "DiseasePrediction_DecisionTree.joblib"
    try:
        # Load the model
        decision_tree = load_model(model_path)
        print("Model loaded successfully.")

        # Get feature template
        feature_template = get_feature_template(dataset_path)

        # Fill missing features with defaults
        for feature in feature_template:
            if feature not in symptoms:
                symptoms[feature] = 0  # Assign default value (e.g., 0)

        # Create input dataframe
        input_data = pd.DataFrame([symptoms], columns=feature_template.keys())

        # Validate input dimensions
        if input_data.shape[1] != len(feature_template):
            raise ValueError(
                f"Expected {len(feature_template)} features, but got {input_data.shape[1]}."
            )

        # Perform prediction
        prediction = decision_tree.predict(input_data)
        return prediction[0]  # Return the first prediction
    except Exception as e:
        print(f"Error: {e}")
        return None


if __name__ == "__main__":
    # Get feature template and print expected features
    dataset_path = "Disease_Symptoms.csv"
    feature_template = get_feature_template(dataset_path)
    print("Expected Features:", list(feature_template.keys()))

    # Example input with only a few features provided
    example_symptoms = {
        "Symptom1": 1,
        "Symptom2": 0,
        "Symptom3": 1,
    }

    # Predict disease
    prediction = predict_disease(example_symptoms, dataset_path=dataset_path)
    if prediction:
        print(f"Predicted Disease: {prediction}")
    else:
        print("Prediction failed.")
