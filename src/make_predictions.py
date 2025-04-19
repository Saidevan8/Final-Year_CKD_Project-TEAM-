import joblib
import pandas as pd
import numpy as np
import os  # For checking file existence
from tensorflow.keras.models import load_model

# Paths to all the models used in hyperparameter tuning
model_paths = {
    'XGBoost': r'models\m1\xgb_model.joblib',
    'CatBoost': r'models\m2\catboost_model.joblib',
    'LightGBM': r'models\m3\lgb_model.joblib',
    'RandomForest': r'models\m4\rf_model.joblib',
    'LSVM': r'models\m5\lsvm_model.joblib',
    'DNN': r'models\m6\dnn_model.keras',  # For Keras model (DNN)
    'LogisticRegression': r'models\m7\lr_model.joblib',
    'KNN': r'models\m8\knn_model.joblib',
    'GradientBoosting': r'models\m9\gb_model.joblib',
    'NaiveBayes': r'models\m10\nb_model.joblib',
    'Scaler': r'models\scaler.joblib'  # For scaling if used
}

# Load new data for predictions (update the path if necessary)
new_data = pd.read_csv("data/ckc.csv")  # Adjust the path as necessary

# Ensure you're selecting the correct features and the 'Diagnosis' column exists
X_new = pd.DataFrame(new_data.iloc[:, :-1].values, columns=new_data.columns[:-1])  # Add column names

# Replace 'Confidential' and 'Not Confidential' with 1 and 0 for the 'Diagnosis' column
y_new = new_data['Diagnosis'].replace({'Confidential': 1, 'Not Confidential': 0})  # Adjust as needed

# Load the scaler if applicable (assuming the scaler is used to standardize features)
scaler = joblib.load(model_paths['Scaler'])  # Assuming scaler is saved at the given path
X_new = scaler.transform(X_new)  # Scale the new data if needed

# Prepare a dictionary to store the predictions from each model
predictions_dict = {}

# Loop through all models, load them, and make predictions
for model_name, model_path in model_paths.items():
    try:
        # Skip scaler since it's not a predictive model
        if model_name == 'Scaler':
            continue

        # Check if the model file exists at the given path
        if not os.path.exists(model_path):
            print(f"Model file for {model_name} not found at {model_path}. Skipping.")
            continue

        # Load the model
        if model_name == 'DNN':  # Handle DNN separately if it's a Keras/TensorFlow model
            model = load_model(model_path)  # Use Keras' load_model for .keras format
        else:
            model = joblib.load(model_path)  # For other models (XGBoost, CatBoost, etc.)

        # Make predictions
        predictions = model.predict(X_new)

        # Adjust predictions for different models
        if model_name == 'DNN':
            predictions = (predictions.flatten() > 0.5).astype(int)  # Convert probabilities to binary labels
        elif model_name in ['XGBoost', 'CatBoost', 'LightGBM', 'RandomForest', 'GradientBoosting', 'LogisticRegression', 'KNN', 'NaiveBayes']:
            if predictions.ndim > 1:  # If predictions are probabilities, we take the class with highest probability
                predictions = np.argmax(predictions, axis=1)  # For multiclass, else for binary
            else:
                predictions = predictions.astype(int)  # In case they are already class labels

        # Save the predictions to the dictionary
        predictions_dict[model_name] = predictions

        # Print predictions for the current model
        print(f"Predictions from {model_name} model:")
        print(predictions)

    except Exception as e:
        print(f"An error occurred while loading or predicting with {model_name}: {e}")

# Optionally, save predictions to a CSV file
predictions_df = pd.DataFrame(predictions_dict)
predictions_df.to_csv('data/model_predictions.csv', index=False)
print("All predictions have been saved to 'model_predictions.csv'.")
