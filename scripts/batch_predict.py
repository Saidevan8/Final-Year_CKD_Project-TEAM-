import pandas as pd
import joblib
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
import numpy as np
import os

# Define paths for models and data
model_paths = {
    "XGBoost": "models/m1/xgb_model.joblib",
    "CatBoost": "models/m2/catboost_model.joblib",
    "LightGBM": "models/m3/lgb_model.joblib",
    "RandomForest": "models/m4/rf_model.joblib",
    "LSVM": "models/m5/lsvm_model.joblib",
    "DNN": "models/m6/dnn_model.keras",  # For Keras model (DNN)
    "LogisticRegression": "models/m7/lr_model.joblib",
    "KNN": "models/m8/knn_model.joblib",
    "GradientBoosting": "models/m9/gb_model.joblib",
    "NaiveBayes": "models/m10/nb_model.joblib",
    "Scaler": "models/scaler.joblib"  # Assuming you also saved the scaler
}
data_path = "data/selected_features.csv"

def load_model(path, model_type):
    """Load models from given paths based on model type."""
    try:
        if model_type == "DNN":
            return tf.keras.models.load_model(path)
        elif model_type == "Scaler":
            return joblib.load(path)
        else:
            return joblib.load(path)
    except Exception as e:
        print(f"Error loading model {model_type}: {e}")
        return None

def batch_predict(models, data_path):
    """Perform batch predictions on the dataset using multiple models."""
    # Load the dataset
    if not os.path.exists(data_path):
        print(f"Error: Data file not found at {data_path}")
        return
    
    data = pd.read_csv(data_path)
    
    # Ensure the dataset contains the necessary columns
    if 'Diagnosis' not in data.columns:
        print("Error: Dataset does not contain 'Diagnosis' column.")
        return
    
    # Separate features and target
    X = data.iloc[:, :-1]  
    data1=pd.read_csv("")# Assuming last column is the target ('Diagnosis')
    y = data1['Diagnosis']
    
    # Scale features using the loaded scaler
    if "Scaler" in models:
        scaler = models["Scaler"]
        X_scaled = scaler.transform(X)  # Use the scaler for transformation
    else:
        # If no scaler model is available, we will fit a new one (not recommended for real batch predictions)
        print("Warning: Scaler model not found, using default scaler.")
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
    
    # Make predictions using each model
    predictions = {}
    for model_name, model in models.items():
        if model_name == "Scaler":
            continue  # Skip scaler from predictions

        if model is None:
            print(f"Skipping {model_name} due to failed model loading.")
            continue

        try:
            if model_name == "DNN":
                preds = (model.predict(X_scaled) > 0.5).astype(int)  # DNN outputs probabilities
            else:
                preds = model.predict(X_scaled)
                if preds.ndim > 1 and preds.shape[1] > 1:  # For multiclass models, take the class with highest probability
                    preds = np.argmax(preds, axis=1)
                preds = preds.astype(int)
            
            predictions[model_name] = preds.flatten()
        except Exception as e:
            print(f"Error during prediction with {model_name}: {e}")
    
    # Save predictions to a DataFrame
    predictions_df = pd.DataFrame(predictions)
    predictions_df['Actual'] = y.reset_index(drop=True)
    
    # Save to CSV
    output_path = "data/batch_predictions.csv"
    predictions_df.to_csv(output_path, index=False)
    print(f"Batch predictions saved successfully to {output_path}.")

if __name__ == "__main__":
    # Load models
    models = {name: load_model(path, name) for name, path in model_paths.items()}
    
    # Perform batch predictions
    batch_predict(models, data_path)
