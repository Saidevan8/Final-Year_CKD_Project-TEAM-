import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import os

def evaluate_model(predictions_path, data_path, output_dir):
    # Load predictions
    predictions_df = pd.read_csv(predictions_path)
    
    # Load true labels (actual values)
    data = pd.read_csv(data_path)
    y_true = data['Diagnosis']  # Assuming 'Diagnosis' column contains the true labels
    
    # Align predictions with true labels
    model_names = predictions_df.columns[:-1]  # Skip the 'Actual' column in predictions.csv
    
    # Store evaluation metrics
    evaluation_metrics = {
        "Model": [],
        "Accuracy": [],
        "Precision": [],
        "Recall": [],
        "F1 Score": []
    }
    
    # Evaluate each model
    for model_name in model_names:
        y_pred = predictions_df[model_name]
        
        # Calculate metrics
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average='binary', pos_label=1)
        recall = recall_score(y_true, y_pred, average='binary', pos_label=1)
        f1 = f1_score(y_true, y_pred, average='binary', pos_label=1)
        
        # Append metrics
        evaluation_metrics["Model"].append(model_name)
        evaluation_metrics["Accuracy"].append(accuracy)
        evaluation_metrics["Precision"].append(precision)
        evaluation_metrics["Recall"].append(recall)
        evaluation_metrics["F1 Score"].append(f1)
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(6, 4))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, 
                    xticklabels=["Not CKD", "CKD"], yticklabels=["Not CKD", "CKD"])
        plt.title(f'Confusion Matrix for {model_name}')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        
        # Save confusion matrix plot
        cm_path = os.path.join(output_dir, f'{model_name}_confusion_matrix.png')
        plt.savefig(cm_path)
        plt.close()
    
    # Save evaluation metrics as a CSV file
    evaluation_df = pd.DataFrame(evaluation_metrics)
    evaluation_metrics_path = os.path.join(output_dir, 'evaluation_metrics.csv')
    evaluation_df.to_csv(evaluation_metrics_path, index=False)
    
    print("Evaluation completed and saved.")

if __name__ == "__main__":
    predictions_path = 'data/batch_predictions.csv'  # Path to batch predictions
    data_path = 'data/ckc.csv'  # Path to original dataset with true labels
    output_dir = 'evolution'  # Directory to save evaluation results
    
    evaluate_model(predictions_path, data_path, output_dir)
