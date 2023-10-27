
import sys
import json
import joblib
import numpy as np
import streamlit as st


# Define the directory where the trained models are saved
model_dir = "ai6-2-last/"


features_columns = ['التصنيف2', 'النوع2', 'area', 'الحي2', 'dd', 'mm', 'yyyy']
# Load the trained models
model_names = ["VotingRegressor2 copy"]
loaded_models = {model_name: joblib.load(
    f"{model_dir}{model_name}.pkl") for model_name in model_names}

# Function to test the input data with loaded models and return predictions


def test_input_with_loaded_models(input_data, loaded_models):
    input_data_array = np.array(list(input_data.values())).reshape(1, -1)
    predictions = {model_name: model.predict(
        input_data_array)[0] for model_name, model in loaded_models.items()}
    return predictions


def fff(input_data):
    predictions = test_input_with_loaded_models(input_data, loaded_models)
    for model_name, prediction in predictions.items():
        prediction = (f'{int(prediction)}')
    return prediction

if __name__ == "__main__":
    try:
        # Get the input data JSON string from the command line arguments
        input_data_json = sys.argv[1]

        # Use json.loads() to parse the JSON input data safely
        input_data = json.loads(input_data_json)

        # Call the function to get predictions
        predictions = fff(input_data)

        # Use json.dumps() to convert predictions to a JSON string
        predictions_json = json.dumps(predictions)

        # Print the predictions
        print(predictions_json)
    except Exception as e:
        print(f"An error occurred: {str(e)}", file=sys.stderr)
