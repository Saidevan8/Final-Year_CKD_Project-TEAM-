import os
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LassoCV, LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import (classification_report, confusion_matrix, roc_curve, auc,accuracy_score, precision_score, recall_score, f1_score)
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier
from sklearn.svm import LinearSVC
from catboost import CatBoostClassifier
import lightgbm as lgb
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import warnings

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

# Suppress warnings, particularly deprecation warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)

# Directory paths
models_dir = 'models'
plots_dir = 'plots'
os.makedirs(models_dir, exist_ok=True)
os.makedirs(plots_dir, exist_ok=True)

# Load the data
data = pd.read_csv(r'data\ckc.csv')
X = data.iloc[:, :-1] 
y = data['Diagnosis']

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model saving function
def save_model(model, model_name, model_dir):
    os.makedirs(model_dir, exist_ok=True)
    if isinstance(model, keras.Model):
        model.save(f'{model_dir}/{model_name}.keras')  # Save in the new Keras format
    else:
        joblib.dump(model, f'{model_dir}/{model_name}.joblib')

# Function to plot training history
def plot_training_history(history, model_name, save_dir):
    plt.figure(figsize=(12, 4))
    
    # Plot loss
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title(f'{model_name} Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    
    # Plot accuracy
    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title(f'{model_name} Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/{model_name}_training_history.png')
    plt.close()

# Function to plot heatmap of feature correlations
def plot_correlation_heatmap(data, save_dir):
    plt.figure(figsize=(12, 10))
    correlation_matrix = data.corr()
    sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='coolwarm', square=True)
    plt.title('Feature Correlation Heatmap')
    plt.savefig(f'{save_dir}/feature_correlation_heatmap.png')
    plt.close()


scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

lasso = LassoCV(alphas=np.logspace(-4, 4, 100), cv=5, random_state=42)
lasso.fit(X_scaled, y)

smote = SMOTE()
X_resampled, y_resampled = smote.fit_resample(X_scaled, y)
# Model training and saving
models = {}

# XGBoost - Hyperparameter tuning
xgb_param_grid = {
    'learning_rate': [0.01, 0.05, 0.1],
    'max_depth': [4, 6, 8],
    'n_estimators': [500, 1000, 1500]
}
xgb = XGBClassifier(objective = 'binary:logistic', learning_rate = 0.5, max_depth = 5, n_estimators = 150)
xgb.fit(X_train, y_train)
save_model(xgb, 'xgb_model', f'{models_dir}/m1')
models['XGBoost'] = xgb

# CatBoost - Hyperparameter tuning
catboost_model = CatBoostClassifier(iterations=3000, depth=10, learning_rate=0.01, verbose=0)
catboost_model.fit(X_train, y_train)
save_model(catboost_model, 'catboost_model', f'{models_dir}/m2')
models['CatBoost'] = catboost_model

# LightGBM - Hyperparameter tuning
lgb_model = lgb.LGBMClassifier(n_estimators=2000, learning_rate=0.01, num_leaves=64, random_state=42)
lgb_model.fit(X_train, y_train)
save_model(lgb_model, 'lgb_model', f'{models_dir}/m3')
models['LightGBM'] = lgb_model

# Random Forest - Hyperparameter tuning
rf_param_grid = {
    'n_estimators': [100, 200, 500],
    'max_depth': [10, 20, 30],
    'min_samples_split': [2, 5, 10]
}
rf_model = RandomForestClassifier(random_state=42)
grid_search_rf = GridSearchCV(estimator=rf_model, param_grid=rf_param_grid, cv=5, scoring='accuracy', n_jobs=-1)
grid_search_rf.fit(X_train, y_train)
best_rf_model = grid_search_rf.best_estimator_
save_model(best_rf_model, 'rf_model', f'{models_dir}/m4')
models['RandomForest'] = best_rf_model

# Linear SVC - No major changes
lsvm_model = LinearSVC(max_iter=10000)
lsvm_model.fit(X_train, y_train)
save_model(lsvm_model, 'lsvm_model', f'{models_dir}/m5')
models['LSVM'] = lsvm_model

# DNN (Keras) - Adjusted learning rate, more epochs
dnn_model = keras.Sequential([
    layers.InputLayer(shape=(X_train.shape[1],)),  # Changed input_shape to shape
    layers.Dense(128, activation='relu'),
    layers.Dense(64, activation='relu'),
    layers.Dense(32, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])
dnn_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
history = dnn_model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test))
save_model(dnn_model, 'dnn_model', f'{models_dir}/m6')
models['DNN'] = dnn_model
plot_training_history(history, 'DNN', plots_dir)  # Plot training history for DNN

# Logistic Regression - Hyperparameter tuning
lr_param_grid = {
    'C': [0.1, 1, 10],
    'solver': ['liblinear', 'saga']
}
lr_model = LogisticRegression()
grid_search_lr = GridSearchCV(estimator=lr_model, param_grid=lr_param_grid, cv=5, scoring='accuracy', n_jobs=-1)
grid_search_lr.fit(X_train, y_train)
best_lr_model = grid_search_lr.best_estimator_
save_model(best_lr_model, 'lr_model', f'{models_dir}/m7')
models['Logistic Regression'] = best_lr_model

# KNN - Hyperparameter tuning
knn_param_grid = {
    'n_neighbors': [3, 5, 7, 9],
    'weights': ['uniform', 'distance'],
    'metric': ['euclidean', 'manhattan']
}
knn_model = KNeighborsClassifier()
grid_search_knn = GridSearchCV(estimator=knn_model, param_grid=knn_param_grid, cv=5, scoring='accuracy', n_jobs=-1)
grid_search_knn.fit(X_train, y_train)
best_knn_model = grid_search_knn.best_estimator_
save_model(best_knn_model, 'knn_model', f'{models_dir}/m8')
models['KNN'] = best_knn_model

# Gradient Boosting - Hyperparameter tuning
gb_param_grid = {
    'n_estimators': [100, 200, 500],
    'learning_rate': [0.01, 0.05, 0.1],
    'max_depth': [3, 5, 7]
}
gb_model = GradientBoostingClassifier()
grid_search_gb = GridSearchCV(estimator=gb_model, param_grid=gb_param_grid, cv=5, scoring='accuracy', n_jobs=-1)
grid_search_gb.fit(X_train, y_train)
best_gb_model = grid_search_gb.best_estimator_
save_model(best_gb_model, 'gb_model', f'{models_dir}/m9')
models['Gradient Boosting'] = best_gb_model

# Naive Bayes
nb_model = GaussianNB()
nb_model.fit(X_train, y_train)
save_model(nb_model, 'nb_model', f'{models_dir}/m10')
models['Naive Bayes'] = nb_model

# Evaluation and saving model comparison
performance_metrics = {}
for model_name, model in models.items():
    if model_name != 'LSVM' and model_name != 'DNN':
        y_pred = model.predict(X_test)
        y_scores = model.predict_proba(X_test)[:, 1]
    elif model_name == 'DNN':
        y_pred = (model.predict(X_test) > 0.5).astype(int)
        y_scores = model.predict(X_test).ravel()
    else:
        y_pred = model.predict(X_test)
        y_scores = model.decision_function(X_test)

    # Metrics
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = auc(roc_curve(y_test, y_scores)[0], roc_curve(y_test, y_scores)[1])

    performance_metrics[model_name] = {
        'accuracy': acc, 'precision': prec, 'recall': rec, 'f1_score': f1, 'roc_auc': roc_auc
    }

# Save performance metrics
pd.DataFrame(performance_metrics).T.to_csv('data/model_training.csv')

# Plot confusion matrix
def plot_confusion_matrix(cm, model_name, save_dir):
    plt.figure(figsize=(6, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Negative', 'Positive'], yticklabels=['Negative', 'Positive'])
    plt.title(f'Confusion Matrix for {model_name}')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.savefig(f'{save_dir}/{model_name}_confusion_matrix.png')
    plt.close()

# Plot ROC curve
def plot_roc_curve(y_test, y_scores, model_name, save_dir):
    fpr, tpr, _ = roc_curve(y_test, y_scores)
    plt.figure()
    plt.plot(fpr, tpr, color='blue', label='ROC curve (area = %0.2f)' % auc(fpr, tpr))
    plt.plot([0, 1], [0, 1], color='red', linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve for {model_name}')
    plt.legend(loc='lower right')
    plt.savefig(f'{save_dir}/{model_name}_roc_curve.png')
    plt.close()

# Plot feature importance
def plot_feature_importance(model, feature_names, model_name, save_dir):
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
    elif hasattr(model, 'coef_'):
        importances = np.abs(model.coef_[0])
    else:
        return  # Model does not have feature importances

    indices = np.argsort(importances)[::-1]
    plt.figure()
    plt.title(f'Feature Importances for {model_name}')
    plt.bar(range(len(importances)), importances[indices], align='center')
    plt.xticks(range(len(importances)), np.array(feature_names)[indices], rotation=90)
    plt.xlim([-1, len(importances)])
    plt.tight_layout()
    plt.savefig(f'{save_dir}/{model_name}_feature_importance.png')
    plt.close()

# Confusion matrices and ROC curves
for model_name, model in models.items():
    if model_name != 'LSVM' and model_name != 'DNN':
        cm = confusion_matrix(y_test, model.predict(X_test))
        plot_confusion_matrix(cm, model_name, plots_dir)
        y_scores = model.predict_proba(X_test)[:, 1]
        plot_roc_curve(y_test, y_scores, model_name, plots_dir)
        plot_feature_importance(model, X, model_name, plots_dir)
    elif model_name == 'DNN':
        cm = confusion_matrix(y_test, (model.predict(X_test) > 0.5).astype(int))
        plot_confusion_matrix(cm, model_name, plots_dir)
        y_scores = model.predict(X_test).ravel()
        plot_roc_curve(y_test, y_scores, model_name, plots_dir)
        plot_training_history(history, model_name, plots_dir)  # Plot training history for DNN
    else:
        cm = confusion_matrix(y_test, model.predict(X_test))
        plot_confusion_matrix(cm, model_name, plots_dir)
        y_scores = model.decision_function(X_test)
        plot_roc_curve(y_test, y_scores, model_name, plots_dir)

# Plot heatmap of feature correlations
plot_correlation_heatmap(data, plots_dir)

# Final message
print("All models have been trained, saved, and evaluated. Performance metrics and visualizations saved.")